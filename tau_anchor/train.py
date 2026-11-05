"""Retrain Tau-Anchor with the paper's recipe (configs/tau_anchor.json).

    python -m tau_anchor.train embed    --manifest aigvd.csv --data-root DATA --cache runs/emb
    python -m tau_anchor.train probe    --manifest aigvd.csv --cache runs/emb --out runs/probe.json
    python -m tau_anchor.train finetune --manifest aigvd.csv --data-root DATA --cache runs/emb \\
                                        --probe runs/probe.json --fold 0 --out-dir runs/fold0
    python -m tau_anchor.train export   --checkpoint runs/fold0/checkpoint.pt --out tau_anchor.pt

Steps
-----
1. ``embed``: the frozen Delta-CLIP encoder embeds all 11 frames of every clip (float16 cache). These embeddings
   are the reference of the preservation term and the features of the probe.
2. ``probe``: per fold f, a logistic regression (C = 10, balanced classes) on the clip embeddings of the other folds
   (the normalized mean of the frame embeddings). Its weights start the head of fold f. Its scores on fold f, with
   the threshold at 5% false alarms on fold f's real clips, give the generator weights of the other folds' runs.
3. ``finetune``: adversarial fine-tuning of the last 4 blocks, the final norm, the projection and the head, for one
   test fold. Each step, on a batch of B clips with F random frames each (y = 1 generated, 0 real):
     delta  ~ U[-eps, eps], then ``pgd_steps`` steps
            delta <- clip(delta + eps/2 sign(grad_delta sum_i w_i CE_i), -eps, eps)
            on every clip ("both"; "generated" attacks generated clips only), with the model in eval mode;
     loss   = clean_weight mean_i w_i CE(s(x_i), y_i) + mean_i w_i CE(s(x_i + delta_i), y_i)
            + lambda mean_{i,t} |e_t(x_i) - e0_t(x_i)|^2
            + gamma mean_{i: y_i = 1} w_i softplus((kappa - rho_i) / kappa),
     rho_i  = (s(x_i + delta_i) - tau_hat) / |w_head|
   with w_i = (n_real / n_generated) x the generator weight for generated clips, 1 for real clips; e_t the
   normalized frame embeddings and e0_t the cached ones of the frozen encoder; tau_hat the 95th percentile of the
   scores of 256 real training clips (fixed frames, re-estimated every ``tau_every`` steps). AdamW (weight decay
   1e-4; learning rate 1e-5 for the encoder, 1e-3 for the head), 16-bit autocast with a gradient scaler on a GPU,
   an exponential moving average of the trained weights (decay 0.999) deployed at the end. A checkpoint every
   ``ckpt_every`` steps; a restarted run resumes from it. At the end the test fold is scored (all 11 frames) and the
   deployed threshold is set at 5% false alarms on its real clips (``result.json``).
4. ``export``: strips a training checkpoint to the release file (fine-tuned tensors, head, thresholds).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from tau_anchor import data as D

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "tau_anchor.json")


@dataclasses.dataclass
class Config:
    """The recipe. Defaults are the paper's values (configs/tau_anchor.json holds the same)."""
    blocks: int = 4                   # transformer blocks fine-tuned (plus the final norm, the projection, the head)
    frames: int = 2                   # random frames per clip in a training step (of 11)
    batch_size: int = 4               # clips per step
    epochs: float = 1.5               # steps = int(epochs * floor(n_train / batch_size))
    lr_encoder: float = 1e-5
    lr_head: float = 1e-3
    weight_decay: float = 1e-4        # AdamW, both parameter groups
    eps: float = 4.0                  # training budget, gray levels (l_inf)
    pgd_steps: int = 1                # sign steps of size eps / 2 after a uniform random start
    attack_side: str = "both"         # "both" | "generated"
    clean_weight: float = 1.0         # weight of the clean cross-entropy
    lambda_preserve: float = 1.0      # preservation term
    margin: bool = True               # margin term at the threshold estimate
    kappa: float = 0.15               # target margin, embedding units
    gamma: float = 1.0                # weight of the margin term
    tau_quantile: float = 0.95        # threshold estimate: this quantile of real training scores
    tau_clips: int = 256              # real training clips behind the estimate
    tau_every: int = 100              # steps between estimates
    generator_weights: str = "auto"   # "auto" (1 to 3, from the probe's out-of-fold margins) | "none"
    ema_decay: float = 0.999          # 0 = no weight averaging
    amp: bool = True                  # 16-bit autocast + gradient scaler (GPU only)
    fpr: float = 0.05                 # deployed threshold: false-alarm rate on the test fold's real clips
    strict_fpr: float = 0.01
    probe_c: float = 10.0             # logistic-regression probe: inverse regularization
    probe_max_iter: int = 3000
    seed: int = 0                     # 0: unseeded (as the paper's runs); > 0 seeds the shuffle and the attack starts
    workers: int = 7                  # data-loader processes
    ckpt_every: int = 200
    print_every: int = 500

    @classmethod
    def from_file(cls, path: Optional[str] = CONFIG, overrides: Optional[List[str]] = None) -> "Config":
        vals = {}
        if path:
            with open(path) as f:
                vals = {k: v for k, v in json.load(f)["recipe"].items()}
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(vals) - set(fields))
        if unknown:
            raise ValueError(f"unknown recipe keys {unknown}")
        for o in overrides or []:
            k, _, v = o.partition("=")
            if k not in fields:
                raise ValueError(f"unknown recipe key {k!r}; choose from {sorted(fields)}")
            typ = type(getattr(cls(), k))
            vals[k] = (v.lower() in ("1", "true", "yes")) if typ is bool else typ(v)
        return cls(**vals)


# ---- probe and generator weights -----------------------------------------------------------
def clip_features(emb: Dict[str, np.ndarray], ids: List[str]) -> np.ndarray:
    """Clip embeddings: the normalized mean of the cached frame embeddings, float32 ``[n, D]``."""
    p = np.stack([emb[c].astype(np.float32).mean(0) for c in ids])
    return p / np.linalg.norm(p, axis=1, keepdims=True).clip(1e-6)


def fit_probe(rows: List[dict], emb: Dict[str, np.ndarray], cfg: Config, log=print) -> dict:
    """One logistic-regression probe per fold (trained on the other folds) with its out-of-fold scores."""
    from sklearn.linear_model import LogisticRegression
    from civicbench.metrics import auc, threshold_at_fpr
    rows = [r for r in rows if r["clip_id"] in emb]
    ids = [r["clip_id"] for r in rows]
    feats = clip_features(emb, ids)
    y_all = np.array([r["label"] for r in rows])
    fold_all = np.array([r["fold"] for r in rows])
    out = {"format": "tau-anchor-probe/1", "fpr": cfg.fpr, "folds": {}}
    for fold in sorted(set(fold_all.tolist())):
        tr, te = fold_all != fold, fold_all == fold
        clf = LogisticRegression(C=cfg.probe_c, max_iter=cfg.probe_max_iter, class_weight="balanced")
        clf.fit(feats[tr], y_all[tr])
        s = feats[te] @ clf.coef_[0] + clf.intercept_[0]
        y = y_all[te]
        tau = float(threshold_at_fpr(s[y == 0], cfg.fpr))
        out["folds"][str(int(fold))] = {
            "probe_w": clf.coef_[0].astype(float).tolist(), "probe_b": float(clf.intercept_[0]), "tau": tau,
            "tau_strict": float(threshold_at_fpr(s[y == 0], cfg.strict_fpr)), "auc": auc(s, y),
            "tpr": float(np.mean(s[y == 1] >= tau)),
            "scores": {"clip_id": [ids[i] for i in np.flatnonzero(te)], "score": s.astype(float).tolist(),
                       "label": y.astype(int).tolist()}}
        log(f"probe fold={fold} auc={out['folds'][str(int(fold))]['auc']:.4f} tpr={np.mean(s[y == 1] >= tau):.4f}")
    return out


def generator_weights(probe: dict, fold: int, rows: List[dict], lo: float = 1.0, hi: float = 3.0) -> Dict[str, float]:
    """Per generator: (median margin of all generated training clips) / (median margin of its own), clipped to
    [lo, hi]. Margins are the probe's out-of-fold scores minus that fold's threshold, on the folds other than
    ``fold`` (the test fold's clips are never scored)."""
    gen = {r["clip_id"]: r["generator"] for r in rows}
    margins, by_gen = [], {}
    for fk, fr in probe["folds"].items():
        if int(fk) == fold:
            continue
        sc = fr["scores"]
        for cid, s, lab in zip(sc["clip_id"], sc["score"], sc["label"]):
            if int(lab) == 1 and cid in gen:
                m = float(s) - float(fr["tau"])
                margins.append(m)
                by_gen.setdefault(gen[cid], []).append(m)
    if not margins:
        raise ValueError("the probe has no out-of-fold scores of generated training clips")
    overall = float(np.median(margins))
    return {g: float(np.clip(overall / max(float(np.median(v)), 1e-3), lo, hi)) for g, v in sorted(by_gen.items())}


# ---- fine-tuning ---------------------------------------------------------------------------
def _grad_scaler(enabled: bool):
    import torch
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)        # torch < 2.3


def attack_batch(model, x, y, w, eps: float, steps: int, side: str, amp: bool, bce):
    """The training attack (model in eval mode): a uniform random start in the l_inf ball of radius ``eps``, then
    ``steps`` sign steps of size eps / 2 that raise the weighted cross-entropy, projected back into the ball. With
    ``side == "generated"`` real clips stay clean. Returns the attacked clips, clamped to [0, 1]."""
    import torch
    model.encoder.eval()
    gmask = (y > 0).float().view(-1, 1, 1, 1, 1) if side == "generated" else torch.ones_like(y).view(-1, 1, 1, 1, 1)
    d = (torch.rand_like(x) * 2 - 1) * eps * gmask                          # random start in the budget
    for _ in range(steps):
        d.requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lg, _ = model.logits((x + d).clamp(0, 1))
        g, = torch.autograd.grad((w * bce(lg.float(), y)).sum(), d)
        with torch.no_grad():
            d = (d + eps / 2 * g.sign()).clamp(-eps, eps) * gmask
    return (x + d.detach()).clamp(0, 1)


def loss_terms(model, x, xa, e_ref, y, w, tau_hat: float, cfg: Config, amp: bool, bce) -> dict:
    """Tau-Anchor's loss on clean clips ``x`` and attacked clips ``xa`` (model in train mode):
    clean (weighted CE on x), adv (weighted CE on xa), pres (mean squared distance of the normalized frame
    embeddings to the cached ones), margin (gamma x mean over generated clips of w softplus((kappa - rho) / kappa),
    rho = (s(xa) - tau_hat) / |w_head|), and their total ``loss``."""
    import torch
    model.encoder.train()
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        lc, ec = model.logits(x)
        la, _ = model.logits(xa)
    lc, la, ec = lc.float(), la.float(), ec.float()
    clean = (w * bce(lc, y)).mean()
    pres = ((ec - e_ref) ** 2).sum(-1).mean()                               # preservation
    if cfg.attack_side == "generated":
        adv = (w * bce(la, y) * (y > 0)).sum() / (y > 0).sum().clamp_min(1)
    else:
        adv = (w * bce(la, y)).mean()
    loss = cfg.clean_weight * clean + adv + cfg.lambda_preserve * pres
    margin = torch.zeros((), device=x.device)
    if cfg.margin and (y > 0).any():                                       # margin at the threshold estimate
        rad = (la[y > 0] - tau_hat) / model.head.weight.norm().clamp_min(1e-6)
        pen = torch.nn.functional.softplus((cfg.kappa - rad) / cfg.kappa)
        margin = cfg.gamma * (w[y > 0] * pen).mean()
        loss = loss + margin
    return {"loss": loss, "clean": clean, "adv": adv, "pres": pres, "margin": margin, "logit_clean": lc,
            "logit_adv": la}


def finetune(model, rows: List[dict], emb: Dict[str, np.ndarray], probe: dict, fold: int, cfg: Config,
             out_dir: str, device="cpu", log=print) -> dict:
    """Fine-tune ``model`` (a ``TauAnchor`` holding the frozen base encoder) on every fold but ``fold``; write
    ``out_dir/checkpoint.pt`` (resumable; the averaged weights at the end) and ``out_dir/result.json``. Returns
    the result dict (deployed threshold, AUC, TPR and scores on the test fold)."""
    import torch
    dev = torch.device(device)
    if cfg.seed:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)
    ck = os.path.join(out_dir, "checkpoint.pt")
    rows = [dict(r) for r in rows if r["clip_id"] in emb]
    model.to(dev)
    enc = model.encoder
    for q in enc.parameters():
        q.requires_grad_(False)
    train_params = []
    for mod in model.trainable_modules(cfg.blocks):
        for q in mod.parameters():
            q.requires_grad_(True)
            train_params.append(q)
    enc.proj.requires_grad_(True)
    train_params.append(enc.proj)
    trainable_names = {n for n, q in enc.named_parameters() if q.requires_grad}
    pf = probe["folds"][str(fold)]
    model.head = torch.nn.Linear(len(pf["probe_w"]), 1).to(dev)            # started from the fold's probe
    head = model.head
    with torch.no_grad():
        head.weight.copy_(torch.tensor(pf["probe_w"], device=dev)[None])
        head.bias.fill_(pf["probe_b"])
    opt = torch.optim.AdamW([{"params": train_params, "lr": cfg.lr_encoder},
                             {"params": head.parameters(), "lr": cfg.lr_head}], weight_decay=cfg.weight_decay)
    tr = [r for r in rows if r["fold"] != fold]
    gw = generator_weights(probe, fold, rows) if cfg.generator_weights == "auto" else {}
    for r in tr:
        r["gw"] = gw.get(r["generator"], 1.0) if r["label"] == 1 else 1.0
    if gw:
        order = sorted(gw.items(), key=lambda kv: kv[1])
        log("progress generator_weights " + " ".join(f"{k}={v:.2f}" for k, v in order))
    mp = dict(persistent_workers=True, multiprocessing_context="spawn") if cfg.workers > 0 else {}
    loader = torch.utils.data.DataLoader(D.ClipSet(tr, emb, cfg.frames, seed=fold * 100000 + cfg.seed * 7919),
                                         batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers,
                                         drop_last=True, **mp)
    total = int(cfg.epochs * len(loader))
    step = 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=True)
        enc.load_state_dict(st["enc"], strict=False)
        head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        log(f"progress resumed step={step}/{total}")
    amp = dev.type == "cuda" and cfg.amp
    scaler = _grad_scaler(amp)
    ema_src = train_params + list(head.parameters())
    ema = None
    if cfg.ema_decay > 0:
        st_e = torch.load(ck, map_location=dev, weights_only=True).get("ema") if os.path.exists(ck) else None
        ema = [t.to(dev) for t in st_e] if st_e else [q.detach().clone() for q in ema_src]
    eps = cfg.eps / 255.0
    pos = float(np.mean([r["label"] for r in tr]))
    pw = torch.tensor((1 - pos) / max(pos, 1e-6), device=dev)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    t0, run = time.time(), {"clean": 0.0, "adv": 0.0, "pres": 0.0, "n": 0}
    side = cfg.attack_side
    if side not in ("both", "generated"):
        raise ValueError(f"attack_side must be 'both' or 'generated', got {side!r}")

    def save():
        trainable = {k: v for k, v in enc.state_dict().items() if k in trainable_names}
        tmp = ck + ".part"
        torch.save({"enc": trainable, "head": head.state_dict(), "opt": opt.state_dict(), "step": step,
                    "blocks": cfg.blocks, "ema": [t.cpu() for t in ema] if ema else None,
                    "config": dataclasses.asdict(cfg)}, tmp)
        os.replace(tmp, ck)

    real_rows = [r for r in tr if r["label"] == 0]
    pick = np.random.RandomState(fold).choice(len(real_rows), size=min(cfg.tau_clips, len(real_rows)), replace=False)
    tau_set = D.ClipSet([real_rows[i] for i in pick], emb, cfg.frames, seed=7)
    tau_cache = [(tau_set[i][0] * 255).round().to(torch.uint8) for i in range(len(tau_set))] if cfg.margin else []

    def estimate_tau():
        """``tau_quantile`` of the current scores of the cached real training clips (the threshold estimate)."""
        enc.eval()
        sc = []
        with torch.no_grad():
            for j in range(0, len(tau_cache), 16):
                xb = torch.stack(tau_cache[j:j + 16]).to(dev).float() / 255
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    sc.append(model.logits(xb)[0].float())
        return float(torch.quantile(torch.cat(sc), cfg.tau_quantile))

    tau_hat = estimate_tau() if cfg.margin else 0.0
    log(f"EXP=tau_anchor_finetune fold={fold} train_clips={len(tr)} steps={total} device={dev} amp={amp} "
        f"tau_hat={tau_hat:.3f} out={out_dir}")
    while step < total:
        for x, e_ref, y, gwt in loader:
            if step >= total:
                break
            x, e_ref, y, gwt = x.to(dev), e_ref.to(dev), y.to(dev).float(), gwt.to(dev).float()
            w = torch.where(y > 0, pw, torch.ones_like(y)) * gwt
            xa = attack_batch(model, x, y, w, eps, cfg.pgd_steps, side, amp, bce)
            t = loss_terms(model, x, xa, e_ref, y, w, tau_hat, cfg, amp, bce)
            loss = t["loss"]
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                with torch.no_grad():
                    for e_, q_ in zip(ema, ema_src):
                        e_.mul_(cfg.ema_decay).add_(q_.detach(), alpha=1 - cfg.ema_decay)
            step += 1
            if cfg.margin and step % cfg.tau_every == 0:
                tau_hat = estimate_tau()
                enc.train()
            run["clean"] += float(t["clean"].detach())
            run["adv"] += float(t["adv"].detach())
            run["pres"] += float(t["pres"].detach())
            run["n"] += 1
            if step % cfg.print_every == 0:
                n = run["n"]
                log(f"progress step={step}/{total} pct={100 * step / total:.0f} tau_hat={tau_hat:.3f} "
                    f"clean_loss={run['clean'] / n:.4f} adv_loss={run['adv'] / n:.4f} pres={run['pres'] / n:.4f} "
                    f"elapsed_s={time.time() - t0:.0f}")
                run = {"clean": 0.0, "adv": 0.0, "pres": 0.0, "n": 0}
            if step % cfg.ckpt_every == 0:
                save()
    if ema is not None:                                               # deploy the averaged weights
        with torch.no_grad():
            for e_, q_ in zip(ema, ema_src):
                q_.copy_(e_)
    save()
    res = evaluate_fold(model, [r for r in rows if r["fold"] == fold], cfg, dev, amp)
    res.update({"fold": fold, "steps": total, "checkpoint": os.path.abspath(ck)})
    tmp = os.path.join(out_dir, "result.json.part")
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, os.path.join(out_dir, "result.json"))
    log(f"RESULT fold={fold} auc={res['auc']:.4f} tpr={res['tpr']:.4f} tau={res['tau']:.6g} "
        f"seconds={time.time() - t0:.0f}")
    return res


def evaluate_fold(model, rows: List[dict], cfg: Config, dev, amp: bool) -> dict:
    """Score every clip on all 11 frames (the training-time path, under autocast on a GPU) and set the deployed
    threshold at ``cfg.fpr`` false alarms on the real clips."""
    import torch
    from civicbench.metrics import auc, threshold_at_fpr
    model.encoder.eval()
    scores = []
    with torch.no_grad():
        for r in rows:
            x = D.to_tensor(D.read_clip(r["path"])).to(dev)[None]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                scores.append(float(model.logits(x)[0][0].float()))
    s, y = np.array(scores), np.array([r["label"] for r in rows])
    tau = float(threshold_at_fpr(s[y == 0], cfg.fpr))
    return {"tau": tau, "tau_strict": float(threshold_at_fpr(s[y == 0], cfg.strict_fpr)), "fpr": cfg.fpr,
            "auc": auc(s, y), "tpr": float(np.mean(s[y == 1] >= tau)), "n_real": int((y == 0).sum()),
            "n_generated": int((y == 1).sum()),
            "scores": {"clip_id": [r["clip_id"] for r in rows], "score": s.tolist(), "label": y.astype(int).tolist()}}


# ---- embedding cache -----------------------------------------------------------------------
def embed(model, rows: List[dict], cache: str, device="cpu", chunk: int = 1000, log=print) -> None:
    """Cache the frozen encoder's frame embeddings of every clip (chunks already written are skipped)."""
    import torch
    dev = torch.device(device)
    model.to(dev).eval()
    os.makedirs(cache, exist_ok=True)
    for k in range(0, (len(rows) + chunk - 1) // chunk):
        path = D.chunk_path(cache, k)
        if os.path.exists(path):
            continue
        part = rows[k * chunk:(k + 1) * chunk]
        emb = np.zeros((len(part), D.T, model.encoder.output_dim), np.float16)
        valid = np.zeros(len(part), np.uint8)
        t0 = time.time()
        for i, r in enumerate(part):
            try:
                x = D.to_tensor(D.read_clip(r["path"])).to(dev)
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    emb[i] = model.embed_frames(x).float().cpu().numpy()
                valid[i] = 1
            except Exception as e:                                       # recorded, excluded downstream
                log(f"warning clip={r['clip_id']} unreadable: {type(e).__name__}: {str(e)[:160]}")
            if (i + 1) % 100 == 0:
                log(f"progress chunk={k} {i + 1}/{len(part)} elapsed_s={time.time() - t0:.0f}")
        tmp = path + ".part.npz"
        np.savez(tmp, clip_id=np.array([r["clip_id"] for r in part]), emb=emb, valid=valid)
        os.replace(tmp, path)
        log(f"progress chunk={k} clips={len(part)} valid={int(valid.sum())} seconds={time.time() - t0:.0f}")


# ---- export --------------------------------------------------------------------------------
def export(checkpoint: str, out: str, tau: Optional[float] = None, tau_civicbench: Optional[float] = None,
           trained_on: Optional[str] = None, base: Optional[str] = None, base_revision: Optional[str] = None,
           heads: int = 16) -> dict:
    """Release file from a training checkpoint: the fine-tuned encoder tensors and the head (the averaged weights
    of a finished run), the thresholds and the base encoder they apply to. Optimizer state is dropped."""
    import torch
    from tau_anchor.model import BASE_FILE, BASE_REPO, BASE_REVISION
    st = torch.load(checkpoint, map_location="cpu", weights_only=True)
    res = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "result.json")
    if tau is None and os.path.exists(res):
        with open(res) as f:
            tau = json.load(f)["tau"]
    doc = {"format": "tau-anchor/1", "enc": {k: v.detach().clone().contiguous() for k, v in st["enc"].items()},
           "head": {k: v.detach().clone() for k, v in st["head"].items()}, "blocks": int(st.get("blocks", 4)),
           "base": {"repo": base or BASE_REPO, "revision": base_revision or BASE_REVISION, "file": BASE_FILE,
                    "heads": int(heads)}}
    for k, v in (("tau", tau), ("tau_civicbench", tau_civicbench), ("trained_on", trained_on)):
        if v is not None:
            doc[k] = float(v) if k != "trained_on" else str(v)
    tmp = out + ".part"
    torch.save(doc, tmp)
    os.replace(tmp, out)
    return doc


# ---- command line --------------------------------------------------------------------------
def _model(a):
    from tau_anchor.model import TauAnchor, load_base_encoder
    return TauAnchor(load_base_encoder(a.base, a.base_revision, "cpu", a.hf_cache))


def main(argv=None) -> int:
    from tau_anchor.model import BASE_REPO, BASE_REVISION
    p = argparse.ArgumentParser(prog="python -m tau_anchor.train", description="Retrain Tau-Anchor.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, data=True):
        q.add_argument("--manifest", required=True, help="CSV: clip_id, path, label, fold, generator")
        if data:
            q.add_argument("--data-root", default=None, help="prefix of relative clip paths")
        q.add_argument("--cache", required=True, help="directory of the embedding cache")
        q.add_argument("--config", default=CONFIG, help="recipe JSON (default: configs/tau_anchor.json)")
        q.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a recipe value")

    def encoder_args(q):
        q.add_argument("--device", default=None, help="default: cuda when available")
        q.add_argument("--base", default=BASE_REPO, help="base encoder: Hugging Face repo id or local open_clip file")
        q.add_argument("--base-revision", default=BASE_REVISION)
        q.add_argument("--hf-cache", default=None, help="Hugging Face cache directory")

    e = sub.add_parser("embed", help="cache the frozen encoder's frame embeddings")
    common(e)
    encoder_args(e)
    e.add_argument("--chunk", type=int, default=1000)
    q = sub.add_parser("probe", help="linear probe per fold (head initialization, generator weights)")
    common(q, data=False)
    q.add_argument("--out", required=True)
    f = sub.add_parser("finetune", help="adversarial fine-tuning of one test fold")
    common(f)
    encoder_args(f)
    f.add_argument("--probe", required=True, help="probe JSON written by `probe`")
    f.add_argument("--fold", type=int, required=True, help="test fold (the paper's released model: 0)")
    f.add_argument("--out-dir", required=True)
    x = sub.add_parser("export", help="release weights from a training checkpoint")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--tau", type=float, default=None, help="deployed threshold (default: result.json next to it)")
    x.add_argument("--tau-civicbench", type=float, default=None)
    x.add_argument("--trained-on", default=None, help="e.g. 'AIGVDBench folds 1-4'")
    x.add_argument("--base", default=None)
    x.add_argument("--base-revision", default=None)
    a = p.parse_args(argv)

    if a.cmd == "export":
        doc = export(a.checkpoint, a.out, a.tau, a.tau_civicbench, a.trained_on, a.base, a.base_revision)
        print(f"wrote {a.out}: {len(doc['enc'])} encoder tensors, head, tau={doc.get('tau')}")
        return 0
    import torch
    cfg = Config.from_file(a.config, a.set)
    if a.cmd == "probe":
        rows = D.read_manifest(a.manifest)
        doc = fit_probe(rows, D.read_embeddings(a.cache), cfg)
        tmp = a.out + ".part"
        with open(tmp, "w") as fh:
            json.dump(doc, fh)
        os.replace(tmp, a.out)
        return 0
    rows = D.read_manifest(a.manifest, a.data_root)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.cmd == "embed":
        embed(_model(a), rows, a.cache, device, a.chunk)
        return 0
    with open(a.probe) as fh:
        probe = json.load(fh)
    finetune(_model(a), rows, D.read_embeddings(a.cache), probe, a.fold, cfg, a.out_dir, device,
             log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
