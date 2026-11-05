"""Tau-Anchor's training recipe on a tiny random ViT and tiny clips (no downloads, < 1 GB, CPU)."""
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root

from civicbench.metrics import threshold_at_fpr  # noqa: E402
from tau_anchor import data as D, train as TR  # noqa: E402
from tau_anchor.detector import TauAnchorDetector  # noqa: E402
from tau_anchor.model import TauAnchor, VisionTransformer  # noqa: E402

from test_tau_anchor_model import H, HEADS, OUT, T, open_clip_sd  # noqa: E402

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


def base_model(seed=0):
    m = TauAnchor(VisionTransformer.from_open_clip(open_clip_sd(seed), heads=HEADS))
    for q in m.encoder.parameters():
        q.requires_grad_(False)
    return m.eval()


@contextlib.contextmanager
def clip_size(h):
    old, D.SIZE = D.SIZE, h
    try:
        yield
    finally:
        D.SIZE = old


def write_data(d, n_folds=5, per_fold=8):
    """Tiny clips as .npy files and a manifest: per fold 5 real clips and 3 generated (two generators)."""
    rng = np.random.default_rng(0)
    rows = []
    for f in range(n_folds):
        for j in range(per_fold):
            gen = j >= 5
            x = np.clip(rng.integers(40, 200, size=(1, H, H, 3)) + rng.normal(0, 20 if gen else 6, size=(T, H, H, 3)),
                        0, 255).astype(np.uint8)
            cid = f"f{f}c{j}"
            np.save(os.path.join(d, cid + ".npy"), x)
            rows.append({"clip_id": cid, "path": cid + ".npy", "label": "generated" if gen else "real", "fold": f,
                         "generator": ("A" if j % 2 else "B") if gen else ""})
    path = os.path.join(d, "manifest.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def random_probe(rows, seed=0):
    """A probe document without scikit-learn: random weights, out-of-fold scores drawn at random."""
    rng = np.random.default_rng(seed)
    folds = {}
    for f in sorted({r["fold"] for r in rows}):
        te = [r for r in rows if r["fold"] == f]
        s = rng.normal(size=len(te)) + np.array([r["label"] for r in te])
        y = np.array([r["label"] for r in te])
        folds[str(f)] = {"probe_w": rng.normal(size=OUT).tolist(), "probe_b": 0.1,
                         "tau": float(threshold_at_fpr(s[y == 0], 0.05)),
                         "scores": {"clip_id": [r["clip_id"] for r in te], "score": s.tolist(), "label": y.tolist()}}
    return {"folds": folds}


class TestConfig(unittest.TestCase):
    def test_file_is_the_default_recipe(self):
        self.assertEqual(TR.Config.from_file(), TR.Config())
        with open(TR.CONFIG) as f:
            self.assertEqual(sorted(json.load(f)["recipe"]), sorted(TR.Config.__dataclass_fields__))
        c = TR.Config.from_file(overrides=["epochs=0.5", "margin=false", "blocks=2", "attack_side=generated"])
        self.assertEqual((c.epochs, c.margin, c.blocks, c.attack_side), (0.5, False, 2, "generated"))
        with self.assertRaises(ValueError):
            TR.Config.from_file(overrides=["nope=1"])

    def test_paper_values(self):
        c = TR.Config()
        self.assertEqual((c.blocks, c.frames, c.batch_size, c.epochs, c.eps, c.pgd_steps), (4, 2, 4, 1.5, 4.0, 1))
        self.assertEqual((c.lr_encoder, c.lr_head, c.weight_decay), (1e-5, 1e-3, 1e-4))
        self.assertEqual((c.kappa, c.gamma, c.lambda_preserve, c.tau_every, c.tau_quantile),
                         (0.15, 1.0, 1.0, 100, 0.95))
        self.assertEqual((c.ema_decay, c.generator_weights, c.attack_side, c.margin, c.amp),
                         (0.999, "auto", "both", True, True))


class TestGeneratorWeights(unittest.TestCase):
    def test_weights(self):
        rows = [{"clip_id": c, "generator": g} for c, g in
                (("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B"), ("c1", "C"), ("t1", "A"))]
        probe = {"folds": {
            "0": {"tau": 0.0, "scores": {"clip_id": ["t1"], "score": [-100.0], "label": [1]}},     # test fold: ignored
            "1": {"tau": 1.0, "scores": {"clip_id": ["a1", "b1", "r1", "zz"], "score": [3.0, 1.2, 9.0, -50.0],
                                         "label": [1, 1, 0, 1]}},                                # r1 real, zz unknown
            "2": {"tau": 0.5, "scores": {"clip_id": ["a2", "b2", "c1"], "score": [2.5, 0.9, -1.0],
                                         "label": [1, 1, 1]}}}}
        gw = TR.generator_weights(probe, 0, rows)
        # margins: A {2.0, 2.0}, B {0.2, 0.4}, C {-1.5}; overall median of [2, .2, 2, .4, -1.5] = 0.4
        self.assertAlmostEqual(gw["A"], 1.0)                 # 0.4 / 2.0 = 0.2 -> clipped up to 1
        self.assertAlmostEqual(gw["B"], 0.4 / 0.3 + 0.0)     # 1.333...
        self.assertAlmostEqual(gw["C"], 3.0)                 # negative median -> 1e-3 -> clipped down to 3
        self.assertEqual(sorted(gw), ["A", "B", "C"])


class TestStep(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = base_model()
        with torch.no_grad():
            self.m.head.weight.normal_()
        g = torch.Generator().manual_seed(1)
        self.x = torch.rand(4, 2, 3, H, H, generator=g)
        self.y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        self.w = torch.tensor([2.0, 1.0, 3.0, 1.0])
        self.e_ref = torch.randn(4, 2, OUT, generator=g)
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def test_attack(self):
        eps = 4 / 255
        for side in ("both", "generated"):
            torch.manual_seed(7)
            xa = TR.attack_batch(self.m, self.x, self.y, self.w, eps, 1, side, False, self.bce)
            self.assertLessEqual(float((xa - self.x).abs().max()), eps + 1e-6)
            self.assertTrue(bool(((xa >= 0) & (xa <= 1)).all()))
            moved = (xa != self.x).flatten(1).any(1)
            self.assertTrue(bool(moved[self.y > 0].all()))
            self.assertEqual(bool(moved[self.y == 0].any()), side == "both")
        torch.manual_seed(7)                                   # one step from the random start, by hand
        d = (torch.rand_like(self.x) * 2 - 1) * eps
        d.requires_grad_(True)
        lg, _ = self.m.logits((self.x + d).clamp(0, 1))
        g, = torch.autograd.grad((self.w * self.bce(lg, self.y)).sum(), d)
        want = (self.x + (d.detach() + eps / 2 * g.sign()).clamp(-eps, eps)).clamp(0, 1)
        torch.manual_seed(7)
        got = TR.attack_batch(self.m, self.x, self.y, self.w, eps, 1, "both", False, self.bce)
        self.assertTrue(torch.equal(got, want))

    def test_loss_terms(self):
        cfg, tau_hat = TR.Config(), 0.3
        xa = (self.x + 0.01).clamp(0, 1)
        t = TR.loss_terms(self.m, self.x, xa, self.e_ref, self.y, self.w, tau_hat, cfg, False, self.bce)
        t = {k: v.detach() for k, v in t.items()}
        with torch.no_grad():
            lc, ec = self.m.logits(self.x)
            la, _ = self.m.logits(xa)
        ce = lambda s, y: -(y * torch.log(torch.sigmoid(s)) + (1 - y) * torch.log(1 - torch.sigmoid(s)))
        self.assertAlmostEqual(float(t["clean"]), float((self.w * ce(lc, self.y)).mean()), places=5)
        self.assertAlmostEqual(float(t["adv"]), float((self.w * ce(la, self.y)).mean()), places=5)
        self.assertAlmostEqual(float(t["pres"]), float(((ec - self.e_ref) ** 2).sum(-1).mean()), places=5)
        rho = (la - tau_hat) / self.m.head.weight.norm()
        pen = torch.log1p(torch.exp((cfg.kappa - rho) / cfg.kappa))
        want_margin = float((self.w * pen)[self.y > 0].mean())
        self.assertAlmostEqual(float(t["margin"]), want_margin, places=4)
        self.assertAlmostEqual(float(t["loss"]), float(t["clean"] + t["adv"] + t["pres"]) + want_margin, places=4)
        # a fake far above the threshold estimate is barely pushed; one below it is pushed by at least gamma / (2 kappa)
        far = TR.Config(margin=True, lambda_preserve=0.0)
        t2 = TR.loss_terms(self.m, self.x, xa, self.e_ref, self.y, self.w, float(la.min()) - 50.0, far, False, self.bce)
        self.assertLess(float(t2["margin"].detach()), 1e-6)
        off = TR.loss_terms(self.m, self.x, xa, self.e_ref, self.y, self.w, tau_hat,
                            TR.Config(margin=False, attack_side="generated", clean_weight=0.5, lambda_preserve=2.0),
                            False, self.bce)
        off = {k: v.detach() for k, v in off.items()}
        self.assertEqual(float(off["margin"]), 0.0)
        adv_fakes = float((self.w * ce(la, self.y))[self.y > 0].sum() / 2)
        self.assertAlmostEqual(float(off["adv"]), adv_fakes, places=5)
        self.assertAlmostEqual(float(off["loss"]), 0.5 * float(off["clean"]) + adv_fakes + 2.0 * float(off["pres"]),
                               places=4)


class TestFinetune(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d, clip_size(H):
            rows = D.read_manifest(write_data(d), d)
            self.assertEqual(sum(r["label"] for r in rows), 15)
            cache = os.path.join(d, "emb")
            TR.embed(base_model(), rows, cache, "cpu", chunk=16, log=lambda s: None)
            emb = D.read_embeddings(cache)
            self.assertEqual(len(emb), len(rows))
            self.assertEqual(emb[rows[0]["clip_id"]].shape, (T, OUT))
            self.assertEqual(emb[rows[0]["clip_id"]].dtype, np.float16)
            cfg = TR.Config(blocks=1, epochs=0.5, workers=0, ckpt_every=2, tau_every=2, ema_decay=0.5, tau_clips=8,
                            lr_encoder=1e-3)
            probe = TR.fit_probe(rows, emb, cfg, log=lambda s: None) if HAVE_SKLEARN else random_probe(rows)
            self.assertEqual(sorted(probe["folds"]), ["0", "1", "2", "3", "4"])
            out = os.path.join(d, "fold0")
            logs = []
            model = base_model()
            frozen = model.encoder.blocks[0].fc.weight.clone()
            res = TR.finetune(model, rows, emb, probe, 0, cfg, out, "cpu", log=logs.append)
            st = torch.load(os.path.join(out, "checkpoint.pt"), weights_only=True)
            self.assertEqual(st["step"], 4)                      # 32 training clips / 4 per batch x 0.5 epochs
            self.assertEqual(sorted(st["enc"]), sorted(model.trainable_names(blocks=1)))
            src = [q for mod in model.trainable_modules(1) for q in mod.parameters()]
            src += [model.encoder.proj, model.head.weight, model.head.bias]
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(st["ema"], src)))   # the averaged weights deployed
            self.assertTrue(torch.equal(model.encoder.blocks[0].fc.weight, frozen))   # outside the tail: unchanged
            s, y = np.array(res["scores"]["score"]), np.array(res["scores"]["label"])
            self.assertEqual(len(s), 8)
            self.assertEqual(res["tau"], threshold_at_fpr(s[y == 0], 0.05))
            with open(os.path.join(out, "result.json")) as f:
                self.assertEqual(json.load(f)["tau"], res["tau"])
            self.assertTrue(any("generator_weights" in line for line in logs))
            # resume: a longer run continues from the checkpoint
            logs.clear()
            TR.finetune(base_model(), rows, emb, probe, 0, TR.Config(**{**cfg.__dict__, "epochs": 1.0}), out, "cpu",
                        log=logs.append)
            self.assertEqual(torch.load(os.path.join(out, "checkpoint.pt"), weights_only=True)["step"], 8)
            self.assertTrue(any("resumed step=4/8" in line for line in logs))
            # export, load, and score like the evaluation in training
            base = os.path.join(d, "base.bin")
            torch.save(open_clip_sd(0), base)
            TR.export(os.path.join(out, "checkpoint.pt"), os.path.join(d, "w.pt"), heads=HEADS)
            det = TauAnchorDetector.from_pretrained(os.path.join(d, "w.pt"), base=base)
            with open(os.path.join(out, "result.json")) as f:
                res2 = json.load(f)
            self.assertEqual(det.tau, res2["tau"])
            test = [r for r in rows if r["fold"] == 0]
            with torch.no_grad():
                got = [float(det(D.to_tensor(D.read_clip(r["path"])))) for r in test]
            np.testing.assert_allclose(got, res2["scores"]["score"], atol=1e-5)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            ck = os.path.join(d, "c.pt")
            torch.save({"enc": {}, "head": {"weight": torch.zeros(1, OUT), "bias": torch.zeros(1)}, "step": 1}, ck)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                argv = ["export", "--checkpoint", ck, "--out", os.path.join(d, "o.pt"), "--tau", "1.5"]
                self.assertEqual(TR.main(argv), 0)
            self.assertEqual(torch.load(os.path.join(d, "o.pt"), weights_only=True)["tau"], 1.5)
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    TR.main(["finetune"])


if __name__ == "__main__":
    unittest.main()
