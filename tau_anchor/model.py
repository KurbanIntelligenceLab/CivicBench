"""Tau-Anchor: a Delta-CLIP frame encoder, the clip embedding and a linear head.

Score of a clip ``x`` (frames ``[T, 3, H, W]`` in [0, 1], RGB):

    e_t  = f(normalize_imagenet(resize_336(x_t))) / |.|     per frame, same weights for every frame
    e(x) = mean_t e_t / |mean_t e_t|                        the clip embedding
    s(x) = w . e(x) + b                                     flagged when s(x) >= tau

* ``f`` is Delta-CLIP ViT-H/14 at 336 px (Hugging Face ``zw123/delta_clip_h14_336``, CC BY 4.0;
  Wang et al., arXiv:2501.09446) in its open_clip layout: patch embedding without bias, class token,
  no norm before the blocks, 32 pre-norm blocks (GELU MLP), average pooling over the patch tokens,
  final norm after pooling, projection to 1,024 dimensions.
* ``resize_336`` is bilinear with antialiasing (``align_corners=False``), clamped to [0, 1];
  ``normalize_imagenet`` uses the ImageNet mean and std (Delta-CLIP's preprocessing).
* Tau-Anchor fine-tunes the last 4 blocks, the final norm, the projection and the head. The
  released weights hold only these parts; ``load`` downloads the base encoder and puts them on top.
* ``TauAnchor.score`` is the inference score (``e @ w + b``). ``TauAnchor.logits`` is the training
  path (an ``nn.Linear`` head, as in training); both compute the same function, up to rounding.

Parameter names follow the original training code (``blocks.<i>.in_proj.weight``, ``ln_post.weight``,
``proj``, ...), so a training checkpoint loads without conversion.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_REPO = "zw123/delta_clip_h14_336"                       # Delta-CLIP ViT-H/14-336, CC BY 4.0
BASE_FILE = "open_clip_pytorch_model.bin"
BASE_REVISION = "724baea4b4f40c477a05a674e1b24cd4638f8796"   # main as of 2026-11 (last change 2025-04-09)
DELTA_CLIP = dict(width=1280, layers=32, heads=16, patch_size=14, image_size=336, mlp_width=5120, output_dim=1024)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)     # ImageNet normalization (Delta-CLIP preprocessing)


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln_1(x)), then x + mlp(ln_2(x)); MLP = fc, GELU, proj."""

    def __init__(self, width: int, heads: int, mlp_width: int):
        super().__init__()
        self.heads = heads
        self.ln_1, self.ln_2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, 3 * width)
        self.out_proj = nn.Linear(width, width)
        self.fc, self.proj = nn.Linear(width, mlp_width), nn.Linear(mlp_width, width)
        self.act = nn.GELU()

    def attn(self, x):
        b, n, w = x.shape
        q, k, v = self.in_proj(x).view(b, n, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(o.transpose(1, 2).reshape(b, n, w))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.proj(self.act(self.fc(self.ln_2(x))))


class VisionTransformer(nn.Module):
    """Normalized pixels ``[N, 3, S, S]`` -> pooled, projected embedding ``[N, D]`` (not normalized)."""

    def __init__(self, width: int, layers: int, heads: int, patch_size: int, image_size: int, mlp_width: int,
                 output_dim: int, ln_pre: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, patch_size, patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width) if ln_pre else None
        self.blocks = nn.ModuleList(Block(width, heads, mlp_width) for _ in range(layers))
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.zeros(width, output_dim))
        self.config = dict(width=width, layers=layers, heads=heads, patch_size=patch_size, image_size=image_size,
                           mlp_width=mlp_width, output_dim=output_dim, ln_pre=ln_pre)
        self.image_size, self.output_dim = image_size, output_dim

    def forward(self, x):
        x = self.conv1(x).flatten(2).transpose(1, 2)
        cls = self.class_embedding.view(1, 1, -1).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], 1) + self.positional_embedding
        if self.ln_pre is not None:
            x = self.ln_pre(x)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_post(x[:, 1:].mean(1)) @ self.proj         # average pooling, then the final norm

    @classmethod
    def from_open_clip(cls, state_dict: dict, heads: int) -> "VisionTransformer":
        """Build from an open_clip CLIP state dict (keys ``visual.*``; other keys are ignored)."""
        sd = {k[len("module."):] if k.startswith("module.") else k: t for k, t in state_dict.items()}
        v = {k[len("visual."):]: t for k, t in sd.items() if k.startswith("visual.")}
        if not v:
            raise ValueError("no 'visual.*' keys: not an open_clip CLIP state dict")
        width, patch = v["conv1.weight"].shape[0], v["conv1.weight"].shape[-1]
        layers = 1 + max(int(k.split(".")[2]) for k in v if k.startswith("transformer.resblocks."))
        n_tok = v["positional_embedding"].shape[0]
        enc = cls(width, layers, heads, patch, int(round((n_tok - 1) ** 0.5)) * patch,
                  v["transformer.resblocks.0.mlp.c_fc.weight"].shape[0], v["proj"].shape[1],
                  ln_pre="ln_pre.weight" in v)
        enc.load_state_dict(open_clip_to_encoder(v))
        return enc


def open_clip_to_encoder(v: dict) -> dict:
    """open_clip vision keys (``visual.`` stripped) -> ``VisionTransformer`` keys. Raises on keys this
    architecture does not have (layer scale, attention pooling, ...)."""
    out, known = {}, set()
    for a in ("conv1.weight", "class_embedding", "positional_embedding", "proj", "ln_post.weight", "ln_post.bias",
              "ln_pre.weight", "ln_pre.bias"):
        if a in v:
            out[a] = v[a]
            known.add(a)
    names = (("ln_1", "ln_1"), ("ln_2", "ln_2"), ("attn.out_proj", "out_proj"), ("mlp.c_fc", "fc"),
             ("mlp.c_proj", "proj"))
    i = 0
    while f"transformer.resblocks.{i}.ln_1.weight" in v:
        s, d = f"transformer.resblocks.{i}.", f"blocks.{i}."
        for a, b in names:
            for p in ("weight", "bias"):
                out[d + f"{b}.{p}"] = v[s + f"{a}.{p}"]
                known.add(s + f"{a}.{p}")
        out[d + "in_proj.weight"], out[d + "in_proj.bias"] = v[s + "attn.in_proj_weight"], v[s + "attn.in_proj_bias"]
        known |= {s + "attn.in_proj_weight", s + "attn.in_proj_bias"}
        i += 1
    extra = sorted(set(v) - known)
    if extra:
        raise ValueError(f"unsupported open_clip vision keys: {extra[:5]}")
    return out


class TauAnchor(nn.Module):
    """Encoder + linear head. ``score(x)``: clips ``[B, T, 3, H, W]`` in [0, 1] -> scores ``[B]``."""

    def __init__(self, encoder: VisionTransformer):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.output_dim, 1)

    def embed_frames(self, frames):
        """Frames ``[N, 3, H, W]`` in [0, 1] -> L2-normalized frame embeddings ``[N, D]`` (differentiable)."""
        s = self.encoder.image_size
        x = F.interpolate(frames, size=(s, s), mode="bilinear", antialias=True, align_corners=False).clamp(0, 1)
        m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
        sd = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
        e = self.encoder((x - m) / sd)
        return e / e.norm(dim=-1, keepdim=True)

    def clip_embedding(self, x):
        """Clips ``[B, T, 3, H, W]`` -> (clip embeddings ``[B, D]``, frame embeddings ``[B, T, D]``)."""
        b, t = x.shape[:2]
        f = self.embed_frames(x.reshape(b * t, *x.shape[2:])).view(b, t, -1)
        e = f.mean(1)
        return e / e.norm(dim=-1, keepdim=True), f

    def score(self, x):
        """Clips ``[B, T, 3, H, W]`` in [0, 1] -> scores ``[B]``, ``w . e(x) + b`` (the evaluation path)."""
        e, _ = self.clip_embedding(x)
        return e @ self.head.weight[0] + float(self.head.bias[0])

    def logits(self, x):
        """Training path: clips ``[B, F, 3, H, W]`` -> (scores ``[B]`` from the ``nn.Linear`` head, frame embeddings
        ``[B, F, D]``)."""
        b, f = x.shape[:2]
        e = self.embed_frames(x.reshape(b * f, *x.shape[2:])).view(b, f, -1)
        m = e.mean(1)
        return self.head(m / m.norm(dim=-1, keepdim=True))[:, 0], e

    def forward(self, x):
        return self.score(x)

    def trainable_modules(self, blocks: int = 4) -> list:
        """The parts Tau-Anchor fine-tunes, in training order: the last ``blocks`` blocks, the final norm (the
        projection is a bare parameter: ``self.encoder.proj``)."""
        return list(self.encoder.blocks[-blocks:]) + [self.encoder.ln_post]

    def trainable_names(self, blocks: int = 4) -> list:
        """Encoder state-dict keys of the fine-tuned parts (the ``enc`` entry of a checkpoint)."""
        ids = {id(q) for m in self.trainable_modules(blocks) for q in m.parameters()} | {id(self.encoder.proj)}
        return [n for n, q in self.encoder.named_parameters() if id(q) in ids]


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, TypeError, ValueError):     # legacy (non-zip) files cannot be memory-mapped
        return torch.load(path, map_location="cpu", weights_only=True)


def _hf_file(repo: str, filename: str, revision: Optional[str], cache_dir: Optional[str]) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("downloading weights needs huggingface_hub: pip install \".[tau-anchor]\"") from e
    return hf_hub_download(repo, filename, revision=revision, cache_dir=cache_dir)


def load_base_encoder(base: str = BASE_REPO, revision: Optional[str] = BASE_REVISION, device="cpu",
                      cache_dir: Optional[str] = None, heads: int = DELTA_CLIP["heads"]) -> VisionTransformer:
    """The frozen Delta-CLIP vision encoder (eval mode, no gradient on its parameters). ``base`` is a Hugging Face
    repo id or a local open_clip checkpoint file. Loading reads the full CLIP checkpoint (about 4 GB) on the CPU."""
    path = base if os.path.isfile(base) else _hf_file(base, BASE_FILE, revision, cache_dir)
    sd = _torch_load(path)
    sd = sd.get("state_dict", sd)
    enc = VisionTransformer.from_open_clip(sd, heads=heads)
    del sd
    enc.eval().to(device)
    for q in enc.parameters():
        q.requires_grad_(False)
    return enc


def load(weights: str, device="cpu", revision: Optional[str] = None, filename: Optional[str] = None,
         base: Optional[str] = None, base_revision: Optional[str] = None,
         cache_dir: Optional[str] = None, heads: Optional[int] = None) -> TauAnchor:
    """Tau-Anchor in eval mode with frozen parameters.

    weights   a local file (released weights, or a training checkpoint of ``tau_anchor.train``) or a Hugging Face
              repo id holding ``filename`` (default: ``tau_anchor.detector.WEIGHTS_FILE``).
    base      the base encoder (repo id or local open_clip file); default: the one the weights name, else Delta-CLIP.
    heads     attention heads of the base encoder; default: the weight file's, else Delta-CLIP's 16.
    The returned model carries ``meta``: the weight file's thresholds and provenance.
    """
    if not os.path.isfile(weights):
        from tau_anchor.detector import WEIGHTS_FILE
        weights = _hf_file(weights, filename or WEIGHTS_FILE, revision, cache_dir)
    st = torch.load(weights, map_location="cpu", weights_only=True)
    if "enc" not in st or "head" not in st:
        raise ValueError(f"{weights}: not a Tau-Anchor weight file (needs 'enc' and 'head')")
    info = st.get("base") or {}
    enc = load_base_encoder(base or info.get("repo", BASE_REPO),
                            base_revision or (info.get("revision", BASE_REVISION) if not base else None),
                            "cpu", cache_dir, heads=int(heads or info.get("heads", DELTA_CLIP["heads"])))
    own = enc.state_dict()
    bad = [k for k, t in st["enc"].items() if k not in own or tuple(own[k].shape) != tuple(t.shape)]
    if bad:
        raise ValueError(f"fine-tuned tensors that do not fit the base encoder: {bad[:5]}")
    enc.load_state_dict(st["enc"], strict=False)
    model = TauAnchor(enc)
    model.head.load_state_dict(st["head"])
    model.meta = {k: st[k] for k in ("tau", "tau_civicbench", "base", "blocks", "trained_on") if k in st}
    model.eval().to(device)
    for q in model.parameters():
        q.requires_grad_(False)
    return model


__all__ = ["TauAnchor", "VisionTransformer", "Block", "load", "load_base_encoder", "open_clip_to_encoder",
           "BASE_REPO", "BASE_FILE", "BASE_REVISION", "DELTA_CLIP", "MEAN", "STD"]
