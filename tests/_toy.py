"""Tiny shapes and a toy differentiable detector for the tests (no weights, no data, < 1 GB)."""
import numpy as np
import torch

T, H = 3, 16


def linear_detector(seed: int = 123):
    """score = sum over frames and channels of W[t, c] * mean intensity: a fixed linear map."""
    w = torch.tensor(np.random.default_rng(seed).normal(size=(T, 3)), dtype=torch.float64).reshape(-1)

    def det(x):
        return x.double().mean(dim=(-1, -2)).reshape(-1) @ w
    det.w = w
    return det


def unit_reach(det) -> float:
    """How far one gray level can move the linear detector's score (no clipping at 0 or 255)."""
    return float(det.w.abs().sum()) / 255


def clip(seed: int, t: int = T, h: int = H) -> np.ndarray:
    return np.random.default_rng(seed).integers(30, 225, size=(t, h, h, 3), dtype=np.uint8)


def dataset(n_real: int = 10, n_gen: int = 10, seed: int = 0):
    """(clip_id, frames, label, meta) tuples; generated clips are brighter where the detector's weights are positive."""
    det = linear_detector()
    w = det.w.reshape(T, 3).numpy()
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_real + n_gen):
        gen = i >= n_real
        x = rng.integers(60, 190, size=(T, H, H, 3)).astype(np.float64)
        x += (rng.normal(6.0, 4.0) if gen else rng.normal(0.0, 3.0)) * np.sign(w)[:, None, None, :]
        out.append((f"c{i:03d}", np.clip(np.rint(x), 0, 255).astype(np.uint8), int(gen),
                    {"generator": f"g{i % 2}" if gen else None}))
    return out
