"""Attack one clip (``attack``) and evaluate a detector on CivicBench (``evaluate``).

Protocol (the paper's CivicBench setting unless you change the arguments)
------------------------------------------------------------------------
* Threshold ``tau`` at 5% false alarms on the real clips; a clip is flagged when ``score >= tau``.
* Budgets 1, 2 and 4 gray levels (l_inf on the 0-255 grid); any integer levels 1..255 are allowed.
* Attacks PGD (100 steps), APGD (100 steps x 5 restarts) and Square (1,000 queries per budget). Each
  attack runs at the budgets in ascending order and stops at its first realized success.
* Evasion attacks flagged generated clips and succeeds at budget b when an 8-bit clip within b gray
  levels scores below ``tau``. TPR under evasion at b = clean TPR x share of the attacked flagged
  generated clips that no attack breaks at a budget <= b.
* Framing attacks real clips. The attacks aim at the stricter 1% false-alarm threshold (and stop
  there), but a clip counts as framed at budget b once any attack reached ``tau`` (5%) at b or
  below. A real clip flagged before any attack counts as framed at every budget. Attacked
  false-alarm rate at b = share of all attacked real clips framed at b.
* Per clip, success is made monotone in the budget.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from civicbench import frames as F
from civicbench._progress import Progress
from civicbench.attacks import DEFAULT_ATTACKS, FAMILIES, GRADIENT_ATTACKS
from civicbench.metrics import auc, threshold_at_fpr
from civicbench.sweep import check_budgets, cost_to_evade
from civicbench.target import Target

DEFAULT_BUDGETS = (1, 2, 4)
DIRECTIONS = ("evasion", "framing")


def seed_for(*parts) -> int:
    """The per-attack seed: the first 32 bits of sha1 over the ':'-joined parts."""
    return int(hashlib.sha1(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _seed_parts(seed) -> tuple:
    return tuple(seed) if isinstance(seed, (tuple, list)) else (seed,)


def _device(detector, device):
    import torch
    if device is not None:
        return torch.device(device)
    if isinstance(detector, torch.nn.Module):
        for p in itertools.chain(detector.parameters(), detector.buffers()):
            return p.device
    return torch.device("cpu")


def _check_detector(detector) -> None:
    import torch
    if isinstance(detector, torch.nn.Module) and detector.training:
        warnings.warn("the detector is an nn.Module in training mode; call .eval() so its score is deterministic",
                      stacklevel=3)


def score(detector, frames_u8, device=None) -> float:
    """The detector's score of one uint8 clip ``[T, H, W, 3]`` (no gradient, not counted as a query)."""
    t = Target(detector, 0.0, frames_u8, device=_device(detector, device))
    import torch
    from civicbench.target import _as_float
    with torch.no_grad():
        s = _as_float(detector(t.x0))
    if not math.isfinite(s):
        raise ValueError(f"the detector returned a non-finite score ({s}) on a clean clip")
    return s


def _check_attacks(attacks) -> tuple:
    attacks = (attacks,) if isinstance(attacks, str) else tuple(attacks)
    unknown = [a for a in attacks if a not in FAMILIES]
    if unknown or not attacks:
        raise ValueError(f"unknown attacks {unknown}; choose from {sorted(FAMILIES)}")
    return attacks


@dataclass
class AttackResult:
    """Outcome of ``attack`` on one clip. Scores are the detector's own (never negated)."""
    direction: str                  # "evasion" or "framing"
    tau: float                      # threshold at which success is counted
    stop_tau: float                 # threshold the attacks aimed at and stopped at
    clean_score: float
    budgets: tuple                  # gray levels, ascending
    status: str                     # "ok"; "not_flagged" (evasion on an unflagged clip) or
                                    # "already_flagged" (framing on a flagged real clip): nothing to attack
    success: dict                   # budget -> bool, monotone in the budget
    cost: float                     # lowest budget with success (inf if none)
    scores: dict                    # budget -> best score any attack reached there (min for evasion,
                                    # max for framing); None when no attack ran at that budget
    adversarial: dict = field(default_factory=dict, repr=False)   # budget -> uint8 [T, H, W, 3], best clip found
    per_attack: dict = field(default_factory=dict, repr=False)    # attack -> {cost, queries, total_queries,
                                                                  #            seconds, final_score, trace}
    clip_id: Optional[str] = None

    @property
    def frames_adv(self):
        """uint8 frames ``[T, H, W, 3]`` of the successful clip at the lowest successful budget, else None."""
        return self.adversarial.get(self.cost) if math.isfinite(self.cost) else None

    def to_record(self) -> dict:
        """JSON-friendly summary (no frames)."""
        fin = lambda v: None if v is None or not math.isfinite(v) else float(v)
        return {"clip_id": self.clip_id, "direction": self.direction, "tau": self.tau, "stop_tau": self.stop_tau,
                "clean_score": self.clean_score, "status": self.status, "budgets": list(self.budgets),
                "cost": fin(self.cost), "success": {int(b): bool(v) for b, v in self.success.items()},
                "scores": {int(b): fin(v) for b, v in self.scores.items()},
                "per_attack": {k: {**{kk: vv for kk, vv in v.items() if kk != "trace"},
                                   "cost": fin(v["cost"]), "trace": v["trace"]} for k, v in self.per_attack.items()}}


def attack(detector, frames_u8, tau: float, direction: str = "evasion", budgets=DEFAULT_BUDGETS,
           attacks=DEFAULT_ATTACKS, seed=0, stop_tau: Optional[float] = None, device=None,
           clip_id: Optional[str] = None, keep_frames: bool = True, quiet: bool = False,
           _progress=None) -> AttackResult:
    """Attack one clip at every budget.

    detector   callable: float32 tensor ``[T, 3, H, W]`` in [0, 1] -> scalar score, higher = generated.
    frames_u8  uint8 ``[T, H, W, 3]`` RGB (official input: ``[11, 512, 512, 3]``).
    tau        the detector's threshold (flagged when ``score >= tau``).
    direction  "evasion" (push a flagged clip below ``tau``) or "framing" (push a real clip to ``tau``).
    budgets    integer gray levels 1..255 (sorted and deduplicated); default (1, 2, 4).
    attacks    names from ``civicbench.attacks.FAMILIES``: "pgd", "apgd", "square", "simba".
    seed       int/str, or a tuple of parts; attack ``a`` uses ``numpy.random.default_rng(seed_for(*parts, a))``.
    stop_tau   threshold the attacks aim at and stop at (default ``tau``); evaluate() passes the 1%
               threshold for framing, as in the paper. Must not lie on the easy side of ``tau``.
    keep_frames  keep the best realized clip of every budget in ``result.adversarial``.
    quiet      no progress line (it appears only when the call lasts more than about 2 s).
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    budgets, attacks = check_budgets(budgets), _check_attacks(attacks)
    frames_u8 = F.check_frames(frames_u8)
    _check_detector(detector)
    dev = _device(detector, device)
    tau = float(tau)
    stop_tau = tau if stop_tau is None else float(stop_tau)
    framing = direction == "framing"
    if (framing and stop_tau < tau) or (not framing and stop_tau > tau):
        raise ValueError(f"stop_tau {stop_tau} lies on the easy side of tau {tau} for {direction}")
    sign = -1.0 if framing else 1.0
    clean = score(detector, frames_u8, dev)
    flagged = clean >= tau
    res = AttackResult(direction, tau, stop_tau, clean, budgets, "ok", {}, math.inf, {b: None for b in budgets},
                       clip_id=clip_id)
    if flagged != (not framing):              # nothing to attack: already on the attacker's side of tau
        res.status = "already_flagged" if framing else "not_flagged"
        res.success, res.cost = {b: True for b in budgets}, float(budgets[0])
        return res

    parts = _seed_parts(seed)
    best_frames, no_grad = {}, []
    prog = _progress if _progress is not None else Progress(len(attacks) * len(budgets), "attack", unit="steps",
                                                            quiet=quiet)
    try:
        for name in attacks:
            target = Target(detector, sign * stop_tau, frames_u8, sign=sign, device=dev)
            rng = np.random.default_rng(seed_for(*parts, name))
            kept = {}

            def run(t, eps, _name=name, _rng=rng, _kept=kept):
                b = int(round(eps * 255))
                prog.set_status(f"{direction} @{b} · {_name}")
                r = FAMILIES[_name](t, eps, rng=_rng)
                if keep_frames and r.x is not None:
                    _kept[b] = r.x
                if _progress is None:
                    prog.update(1)
                return r

            if name in GRADIENT_ATTACKS:
                g = target.grad(target.x0)
                if not bool((g != 0).any()):
                    no_grad.append(name)
            t0 = time.perf_counter()
            out = cost_to_evade(run, target, budgets)
            if _progress is None:
                prog.update(len(budgets) - len(out["trace"]))      # budgets skipped after a success
            trace = [{**t, "score": sign * t["score"]} for t in out["trace"]]
            res.per_attack[name] = {"cost": out["cte"], "queries": out["queries"], "total_queries": target.queries,
                                    "seconds": round(time.perf_counter() - t0, 3),
                                    "final_score": sign * min(t["score"] for t in out["trace"]), "trace": trace}
            for t in trace:
                b, s = t["budget"], t["score"]
                prev = res.scores[b]
                if prev is None or (s > prev if framing else s < prev):
                    res.scores[b] = s
                    if b in kept:
                        best_frames[b] = kept[b]
    finally:
        if _progress is None:
            prog.close()
    if no_grad:
        warnings.warn(f"the detector's score has no gradient at the clean clip; {', '.join(no_grad)} made no "
                      "gradient progress (use score-based attacks such as 'square' for such detectors)", stacklevel=2)

    hit = False
    for b in budgets:
        s = res.scores[b]
        crossed = s is not None and (s >= tau if framing else s < tau)
        hit = hit or crossed or any(a["cost"] <= b for a in res.per_attack.values())
        res.success[b] = hit
    res.cost = next((float(b) for b in budgets if res.success[b]), math.inf)
    if keep_frames:
        res.adversarial = {b: F.realize(x) for b, x in best_frames.items()}
    return res


# ---- evaluation ----------------------------------------------------------------------------
def _unpack(item):
    """A clip as (clip_id, frames, label, meta); label 0 = real, 1 = generated."""
    if hasattr(item, "clip_id") and hasattr(item, "frames"):
        cid, fr, lab, meta = item.clip_id, item.frames, item.label, dict(getattr(item, "meta", {}) or {})
    else:
        cid, fr, lab, *rest = item
        meta = dict(rest[0]) if rest else {}
    if isinstance(lab, str):
        if lab.lower() not in ("real", "generated", "fake"):
            raise ValueError(f"label must be 0/'real' or 1/'generated', got {lab!r}")
        lab = 0 if lab.lower() == "real" else 1
    lab = int(lab)
    if lab not in (0, 1):
        raise ValueError(f"label must be 0 (real) or 1 (generated), got {lab}")
    return str(cid), fr, lab, meta


def _nanmean(x):
    return float(np.mean(x)) if len(x) else float("nan")


def evaluate(detector, clips: Iterable, fpr: float = 0.05, strict_fpr: Optional[float] = 0.01,
             budgets=DEFAULT_BUDGETS, attacks=DEFAULT_ATTACKS, n_evasion: Optional[int] = None,
             n_framing: Optional[int] = None, evasion_per_generator: Optional[int] = None, seed=None,
             device=None, survival_budget: Optional[int] = None, survival_probe: Optional[int] = None,
             quiet: bool = False) -> dict:
    """Clean AUC and TPR at the ``fpr`` threshold set on the real clips, TPR under evasion and attacked
    false-alarm rate (framing) at every budget: the columns of the paper's CivicBench table.

    clips      re-iterable of clips, e.g. ``civicbench.load(...)``, or (clip_id, frames_u8, label[, meta])
               tuples with label 0/"real" or 1/"generated". A one-shot iterator is materialized.
    strict_fpr false-alarm rate of the threshold that framing attacks aim at (paper: 0.01); None = ``fpr``.
    budgets    integer gray levels 1..255, default (1, 2, 4); ``()`` skips the attacks (clean columns only).
    n_evasion / n_framing / evasion_per_generator  cap the attacked clips (in clip order); default all.
               The paper attacked up to 10 flagged generated clips per generator and 80 real clips.
               With caps, the attacked rates are estimates on a subset.
    seed       None reproduces the paper's per-clip seeds ("civic", 0, clip_id); an int s uses (s, clip_id).
    survival_budget / survival_probe  budgets of the margin-against-reach prediction (default 4 from 1,
               else the largest from the smallest requested budget).
    quiet      no progress line and no printed summary.
    Returns a dict (see README); prints a summary table unless ``quiet``.
    """
    budgets = check_budgets(budgets) if len(tuple(budgets)) else ()
    if budgets:
        attacks = _check_attacks(attacks)
    _check_detector(detector)
    dev = _device(detector, device)
    if iter(clips) is clips:
        clips = list(clips)

    t0 = time.perf_counter()
    rows = []
    with Progress(_len(clips), "scoring", unit="clips", quiet=quiet) as prog:
        for i, item in enumerate(clips):
            cid, fr, lab, meta = _unpack(item)
            rows.append({"clip_id": cid, "label": lab, "score": score(detector, F.check_frames(fr), dev),
                         "generator": meta.get("generator"), "index": i})
            prog.update(1, status=cid)
        prog.status = ""
    ids = [r["clip_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("clip ids must be unique")
    real = np.array([r["score"] for r in rows if r["label"] == 0])
    fake = np.array([r["score"] for r in rows if r["label"] == 1])
    if not len(real) or not len(fake):
        raise ValueError(f"need real and generated clips, got {len(real)} real and {len(fake)} generated")
    tau = threshold_at_fpr(real, fpr)
    tau_strict = threshold_at_fpr(real, strict_fpr) if strict_fpr is not None else tau
    tau_strict = max(tau_strict, tau)
    tpr = float(np.mean(fake >= tau))
    out = {"n_real": int(len(real)), "n_generated": int(len(fake)), "fpr": fpr, "strict_fpr": strict_fpr,
           "tau": tau, "tau_strict": tau_strict,
           "auc": auc(np.r_[real, fake], np.r_[np.zeros(len(real)), np.ones(len(fake))]), "tpr": tpr,
           "budgets": list(budgets), "attacks": list(attacks) if budgets else []}

    if budgets:
        ev, per_gen = [], {}
        for r in rows:
            if r["label"] == 1 and r["score"] >= tau:
                g = r["generator"]
                if evasion_per_generator is not None and per_gen.get(g, 0) >= evasion_per_generator:
                    continue
                per_gen[g] = per_gen.get(g, 0) + 1
                ev.append(r["index"])
        ev = ev[:n_evasion] if n_evasion is not None else ev
        fa = [r["index"] for r in rows if r["label"] == 0]
        fa = fa[:n_framing] if n_framing is not None else fa
        todo = {i: "evasion" for i in ev} | {i: "framing" for i in fa}
        records, top = [], budgets[-1]
        broken = {"evasion": [0, 0], "framing": [0, 0]}
        with Progress(len(todo), "attacking", unit="clips", quiet=quiet) as prog:
            for i, item in enumerate(clips):
                if i not in todo:
                    continue
                if len(records) == len(todo):
                    break
                cid, fr, _, _ = _unpack(item)
                direction = todo[i]
                parts = ("civic", 0, cid) if seed is None else (*_seed_parts(seed), cid)
                res = attack(detector, fr, tau, direction, budgets, attacks, seed=parts,
                             stop_tau=tau_strict if direction == "framing" else None, device=dev, clip_id=cid,
                             keep_frames=False, _progress=prog)
                records.append(res.to_record())
                broken[direction][0] += int(res.success[top])
                broken[direction][1] += 1
                prog.update(1, extra=f"evaded {broken['evasion'][0]}/{broken['evasion'][1]} · "
                                     f"framed {broken['framing'][0]}/{broken['framing'][1]} @{top}")
            prog.status = ""
        E = [r for r in records if r["direction"] == "evasion"]
        R = [r for r in records if r["direction"] == "framing"]
        out.update({"n_evasion": len(E), "n_framing": len(R),
                    "evasion_success": {b: _nanmean([r["success"][b] for r in E]) for b in budgets},
                    "evasion_tpr": {b: tpr * (1 - _nanmean([r["success"][b] for r in E])) for b in budgets},
                    "framing_rate": {b: _nanmean([r["success"][b] for r in R]) for b in budgets},
                    "framed_clean": _nanmean([r["status"] == "already_flagged" for r in R]),
                    "clips": records})
        if "pgd" in attacks:
            from civicbench.margin import predict_survival
            probe = survival_probe if survival_probe is not None else (1 if 1 in budgets else budgets[0])
            target = survival_budget if survival_budget is not None else (4 if 4 in budgets else budgets[-1])
            if target > probe and records:
                out["survival"] = {k: {kk: vv for kk, vv in v.items() if kk != "clips"}
                                   for k, v in predict_survival(records, budget=target, probe_budget=probe).items()}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    if not quiet:
        print(format_summary(out), flush=True)
    return out


def _len(x) -> Optional[int]:
    try:
        return len(x)
    except TypeError:
        return None


def format_summary(res: dict) -> str:
    """The evaluate() result as a small text table."""
    f = lambda v: "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:5.3f}"
    lines = [f"CivicBench: {res['n_real']} real, {res['n_generated']} generated clips; threshold tau = "
             f"{res['tau']:.6g} at {100 * res['fpr']:g}% false alarms on the real clips"]
    head, vals = ["  AUC", "  TPR"], [f(res["auc"]), f(res["tpr"])]
    budgets = res.get("budgets") or []
    if budgets and "evasion_tpr" in res:
        head += [f"e@{b}".rjust(5) for b in budgets] + [f"f@{b}".rjust(5) for b in budgets]
        vals += [f(res["evasion_tpr"][b]) for b in budgets] + [f(res["framing_rate"][b]) for b in budgets]
    lines.append("  ".join(head))
    lines.append("  ".join(vals))
    if budgets and "evasion_tpr" in res:
        lines.append(f"e@b: TPR under evasion at b gray levels ({res['n_evasion']} flagged generated clips attacked); "
                     f"f@b: false-alarm rate after framing at b ({res['n_framing']} real clips attacked, "
                     f"{f(res['framed_clean']).strip()} flagged before the attack); attacks: {', '.join(res['attacks'])}")
    for d, s in (res.get("survival") or {}).items():
        lines.append(f"margin against reach ({d}, PGD): {s['n']} clips; predicted from {s['probe_budget']} gray level(s), "
                     f"{f(s['predicted_keep']).strip()} keep their label at {s['budget']}; measured "
                     f"{f(s['measured_keep']).strip()}")
    return "\n".join(lines)


__all__ = ["attack", "evaluate", "score", "AttackResult", "format_summary", "seed_for", "DEFAULT_BUDGETS"]
