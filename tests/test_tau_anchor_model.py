"""Tau-Anchor's model, score and detector on a tiny random ViT (no downloads, < 1 GB)."""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root

import civicbench  # noqa: E402
from tau_anchor import TauAnchorDetector  # noqa: E402
from tau_anchor.model import MEAN, STD, TauAnchor, VisionTransformer, load  # noqa: E402

W, LAYERS, HEADS, PATCH, SIZE, MLP, OUT, T, H = 32, 3, 4, 8, 16, 64, 16, 11, 24


def open_clip_sd(seed=0):
    """A random open_clip-layout CLIP state dict (vision tower + an unrelated text key)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=0.1: torch.randn(*s, generator=g) * sc
    sd = {"visual.conv1.weight": r(W, 3, PATCH, PATCH), "visual.class_embedding": r(W),
          "visual.positional_embedding": r((SIZE // PATCH) ** 2 + 1, W), "visual.ln_post.weight": 1 + r(W),
          "visual.ln_post.bias": r(W), "visual.proj": r(W, OUT, sc=0.3), "text_projection": r(4, 4)}
    for i in range(LAYERS):
        p = f"visual.transformer.resblocks.{i}."
        sd.update({p + "ln_1.weight": 1 + r(W), p + "ln_1.bias": r(W), p + "ln_2.weight": 1 + r(W),
                   p + "ln_2.bias": r(W), p + "attn.in_proj_weight": r(3 * W, W, sc=0.2),
                   p + "attn.in_proj_bias": r(3 * W),
                   p + "attn.out_proj.weight": r(W, W, sc=0.2), p + "attn.out_proj.bias": r(W),
                   p + "mlp.c_fc.weight": r(MLP, W, sc=0.2), p + "mlp.c_fc.bias": r(MLP),
                   p + "mlp.c_proj.weight": r(W, MLP, sc=0.2), p + "mlp.c_proj.bias": r(W)})
    return sd


def tiny_model(seed=0):
    m = TauAnchor(VisionTransformer.from_open_clip(open_clip_sd(seed), heads=HEADS))
    with torch.no_grad():
        m.head.weight.copy_(torch.randn(1, OUT, generator=torch.Generator().manual_seed(seed + 1)))
        m.head.bias.fill_(0.25)
    return m.eval()


def clip01(seed=0, t=T, h=H):
    return torch.rand(t, 3, h, h, generator=torch.Generator().manual_seed(seed))


class TestEncoder(unittest.TestCase):
    def test_open_clip_layout(self):
        sd = open_clip_sd()
        enc = VisionTransformer.from_open_clip(sd, heads=HEADS)
        self.assertEqual(enc.config, dict(width=W, layers=LAYERS, heads=HEADS, patch_size=PATCH, image_size=SIZE,
                                          mlp_width=MLP, output_dim=OUT, ln_pre=False))
        own = enc.state_dict()
        oc = "visual.transformer.resblocks."
        self.assertTrue(torch.equal(own["blocks.1.in_proj.weight"], sd[oc + "1.attn.in_proj_weight"]))
        self.assertTrue(torch.equal(own["blocks.2.fc.bias"], sd[oc + "2.mlp.c_fc.bias"]))
        self.assertTrue(torch.equal(own["blocks.0.proj.weight"], sd[oc + "0.mlp.c_proj.weight"]))
        self.assertTrue(torch.equal(own["proj"], sd["visual.proj"]))
        bad = dict(sd, **{"visual.transformer.resblocks.0.ls_1.gamma": torch.ones(W)})
        with self.assertRaises(ValueError):                        # layer scale is not part of this architecture
            VisionTransformer.from_open_clip(bad, heads=HEADS)

    def test_trainable_parts(self):
        m = tiny_model()
        names = m.trainable_names(blocks=2)
        want = {n for n, _ in m.encoder.named_parameters() if n.split(".")[0] in ("ln_post", "proj")
                or n.startswith(("blocks.1.", "blocks.2."))}
        self.assertEqual(set(names), want)
        self.assertEqual(len(names), 2 * 12 + 3)              # 12 tensors per block, ln_post weight and bias, proj


class TestScore(unittest.TestCase):
    def test_score_definition(self):
        """s(x) = w . normalize(mean_t normalize(f(norm(resize(x_t))))) + b."""
        m, x = tiny_model(), clip01()
        with torch.no_grad():
            z = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False).clamp(0, 1)
            z = (z - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
            e = m.encoder(z)
            e = e / e.norm(dim=-1, keepdim=True)
            c = e.mean(0)
            c = c / c.norm()
            want = float(c @ m.head.weight[0] + m.head.bias[0])
            got = m.score(x[None])
            self.assertEqual(tuple(got.shape), (1,))
            self.assertAlmostEqual(float(got[0]), want, places=5)
            self.assertAlmostEqual(float(m.logits(x[None])[0][0]), want, places=5)   # the training path
            perm = torch.randperm(T, generator=torch.Generator().manual_seed(5))
            self.assertAlmostEqual(float(m.score(x[perm][None])[0]), want, places=5)   # frame order is irrelevant

    def test_detector_contract(self):
        det = TauAnchorDetector(tiny_model(), tau=0.1, tau_civicbench=0.2)
        x = clip01().requires_grad_(True)
        s = det(x)
        self.assertEqual(s.dim(), 0)
        s.backward()
        self.assertTrue(bool((x.grad != 0).any()))
        self.assertTrue(all(not q.requires_grad for q in det.parameters()))
        xb = torch.stack([clip01(1), clip01(2)])
        with torch.no_grad():
            b = det(xb)
            self.assertEqual(tuple(b.shape), (2,))
            self.assertAlmostEqual(float(b[1]), float(det(xb[1])), places=5)
            self.assertEqual(float(det(xb[0].double())), float(det(xb[0])))
        self.assertEqual((det.tau, det.tau_civicbench), (0.1, 0.2))
        self.assertEqual(det.flagged(xb[0], tau=float(b[0])), True)

    def test_with_civicbench(self):
        det = TauAnchorDetector(tiny_model())
        frames = np.random.default_rng(0).integers(0, 256, size=(T, H, H, 3), dtype=np.uint8)
        s = civicbench.score(det, frames)
        with torch.no_grad():
            self.assertEqual(s, float(det(civicbench.to_tensor(frames))))
        r = civicbench.attack(det, frames, tau=s - 1e-3, direction="evasion", budgets=(1,), attacks=("pgd",),
                              quiet=True)
        self.assertEqual(r.status, "ok")
        self.assertIsNotNone(r.scores[1])
        self.assertIn(1, r.success)


class TestLoad(unittest.TestCase):
    def test_export_and_load(self):
        from tau_anchor.train import export
        m = tiny_model(seed=3)
        base_sd = open_clip_sd(seed=3)
        with torch.no_grad():                                  # "fine-tune" the tail
            for n, q in m.encoder.named_parameters():
                if n in m.trainable_names(blocks=1):
                    q.add_(0.01)
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "base.bin")
            torch.save(base_sd, base)
            names = set(m.trainable_names(blocks=1))
            ck = os.path.join(d, "checkpoint.pt")
            torch.save({"enc": {k: v for k, v in m.encoder.state_dict().items() if k in names},
                        "head": m.head.state_dict(), "opt": {}, "step": 7, "blocks": 1, "ema": None}, ck)
            out = os.path.join(d, "tau_anchor.pt")
            export(ck, out, tau=0.5, tau_civicbench=0.75, heads=HEADS)
            det = TauAnchorDetector.from_pretrained(out, base=base)
            raw = TauAnchorDetector(load(ck, base=base, heads=HEADS))          # a training checkpoint loads as is
            x = clip01(4)
            with torch.no_grad():
                want = float(m.score(x[None])[0])
                self.assertEqual(float(det(x)), want)
                self.assertEqual(float(raw(x)), want)
            self.assertEqual((det.tau, det.tau_civicbench), (0.5, 0.75))
            torch.save({"enc": {"blocks.0.fc.weight": torch.zeros(3, 3)}, "head": m.head.state_dict()}, ck)
            with self.assertRaises(ValueError):
                load(ck, base=base, heads=HEADS)


if __name__ == "__main__":
    unittest.main()
