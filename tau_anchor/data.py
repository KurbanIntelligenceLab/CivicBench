"""Training data for Tau-Anchor: the clip manifest, clip decoding and the cache of base-encoder embeddings.

Manifest (CSV, one row per clip of one benchmark)
--------------------------------------------------
* ``clip_id``    unique id;
* ``path``       the clip: a video file, a ``.npy`` array (uint8 ``[T, H, W, 3]``), a ``.npz`` with ``frames``, or a
                 directory of image frames (sorted by name); relative paths resolve against ``--data-root``;
* ``label``      0 = real, 1 = generated (``real`` / ``generated`` also accepted);
* ``fold``       integer fold (the paper: five folds grouped by prompt);
* ``generator``  the generator of a generated clip (needed for the generator weights; empty for real clips).
Other columns are ignored.

Clips
-----
A clip is uint8 ``[11, 512, 512, 3]`` (RGB): from a video, every k-th decoded frame from the first, k = the integer
nearest fps / 8 (at least 1), until 11 frames are read. The paper prepares every video first, with ffmpeg and x264 at
CRF 23 (see the README):

    ffmpeg -i in.mp4 -c:v libx264 -crf 23 -g 250 -profile:v main -an -vf scale=512:512 out.mp4

``read_clip`` refuses clips of another size instead of resizing them, so that no clip silently departs from that
preparation.

Embedding cache
---------------
``embed`` (train.py) writes ``emb_<k>.npz`` chunks: ``clip_id``, ``emb`` (float16 ``[n, 11, D]``, the L2-normalized
frame embeddings of the frozen base encoder) and ``valid`` (0 where the clip could not be read).
"""
from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List, Optional

import numpy as np

T = 11          # frames per clip
SIZE = 512      # frame height and width
FPS = 8         # target frame rate of the frame sample
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _label(v) -> int:
    s = str(v).strip().lower()
    if s in ("real", "generated", "fake"):
        return 0 if s == "real" else 1
    lab = int(float(s))
    if lab not in (0, 1):
        raise ValueError(f"label must be 0 (real) or 1 (generated), got {v!r}")
    return lab


def read_manifest(path: str, data_root: Optional[str] = None) -> List[dict]:
    """Rows ``{clip_id, path, label, fold, generator}`` in file order."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"clip_id", "path", "label", "fold"} - set(rows[0] if rows else {})
    if need:
        raise ValueError(f"{path}: missing columns {sorted(need)}")
    out, seen = [], set()
    for r in rows:
        cid = str(r["clip_id"])
        if cid in seen:
            raise ValueError(f"{path}: duplicate clip_id {cid}")
        seen.add(cid)
        p = r["path"]
        if data_root and not os.path.isabs(p):
            p = os.path.join(data_root, p)
        out.append({"clip_id": cid, "path": p, "label": _label(r["label"]), "fold": int(float(r["fold"])),
                    "generator": (r.get("generator") or "").strip()})
    return out


def stride(fps_src: float, fps: int = FPS) -> int:
    """Source frames between consecutive clip frames: round(fps_src / fps), at least 1."""
    return max(1, int(round(fps_src / fps)))


def _decode_video(path: str, t: int) -> np.ndarray:
    try:
        import av
    except ImportError as e:
        raise ImportError("decoding videos needs PyAV: pip install \".[tau-anchor-train]\"") from e
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if not stream.average_rate:
            raise ValueError(f"{path}: no frame rate")
        s = stride(float(stream.average_rate))
        for i, frame in enumerate(container.decode(stream)):
            if i % s == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) == t:
                    break
    if len(frames) < t:
        raise ValueError(f"{path}: {len(frames)} of {t} frames at stride {s} (the clip is too short)")
    return np.stack(frames)


def _read_images(path: str, t: int) -> np.ndarray:
    from PIL import Image
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXT))
    if len(files) < t:
        raise ValueError(f"{path}: {len(files)} image frames, need {t}")
    return np.stack([np.asarray(Image.open(os.path.join(path, f)).convert("RGB"), dtype=np.uint8) for f in files[:t]])


def read_clip(path: str, t: int = T, size: Optional[int] = None) -> np.ndarray:
    """uint8 ``[t, size, size, 3]`` (RGB) from a video, an array file or a directory of frames (``size``: default
    the module's ``SIZE``, 512)."""
    size = SIZE if size is None else size
    if os.path.isdir(path):
        a = _read_images(path, t)
    elif path.endswith(".npy"):
        a = np.load(path)[:t]
    elif path.endswith(".npz"):
        with np.load(path) as z:
            a = z["frames"][:t]
    else:
        a = _decode_video(path, t)
    a = np.asarray(a)
    if a.dtype != np.uint8 or a.ndim != 4 or a.shape[-1] != 3 or a.shape[0] < t:
        raise ValueError(f"{path}: expected uint8 [{t}, H, W, 3], got {a.dtype} {a.shape}")
    if a.shape[1:3] != (size, size):
        raise ValueError(f"{path}: frames are {a.shape[1]} x {a.shape[2]}, expected {size} x {size}; prepare the "
                         "clip as in the README (scale=512:512)")
    return np.ascontiguousarray(a, dtype=np.uint8)


def to_tensor(clip_u8: np.ndarray):
    """uint8 ``[T, H, W, 3]`` -> float32 torch tensor ``[T, 3, H, W]`` in [0, 1]."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(clip_u8)).permute(0, 3, 1, 2).float().div_(255.0)


class ClipSet:
    """Torch dataset of training clips: item i -> (frames float32 ``[F, 3, H, W]`` in [0, 1], cached base-encoder
    embeddings of the same frames ``[F, D]``, label, loss weight). The F frames of item i are a sorted draw without
    replacement from the 11, seeded by ``seed + i``, so an item shows the same frames in every epoch."""

    def __init__(self, rows: List[dict], emb: Dict[str, np.ndarray], frames: int, seed: int, t: int = T):
        self.rows, self.emb, self.frames, self.seed, self.t = list(rows), emb, frames, seed, t

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        r = self.rows[i]
        rng = np.random.default_rng(self.seed + i)
        idx = np.sort(rng.choice(self.t, self.frames, replace=False))
        x = to_tensor(read_clip(r["path"], self.t))[idx]
        return (x, torch.from_numpy(self.emb[r["clip_id"]][idx].astype(np.float32)), float(r["label"]),
                float(r.get("gw", 1.0)))


def chunk_path(cache: str, k: int) -> str:
    return os.path.join(cache, f"emb_{k:05d}.npz")


def read_embeddings(cache: str) -> Dict[str, np.ndarray]:
    """clip_id -> float16 frame embeddings ``[T, D]`` of every VALID clip in the cache."""
    files = sorted(glob.glob(os.path.join(cache, "emb_*.npz")))
    if not files:
        raise FileNotFoundError(f"no emb_*.npz in {cache}; run `python -m tau_anchor.train embed` first")
    out = {}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            for cid, e, ok in zip(z["clip_id"], z["emb"], z["valid"]):
                if ok:
                    out[str(cid)] = e
    return out


__all__ = ["read_manifest", "read_clip", "to_tensor", "ClipSet", "read_embeddings", "chunk_path", "stride", "T", "SIZE",
           "FPS"]
