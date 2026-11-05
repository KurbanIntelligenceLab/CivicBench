"""Frame layout shared by every part of the package.

* A clip is stored as uint8 frames ``[T, H, W, 3]`` (RGB). The official CivicBench test input is
  ``T = 11`` frames of ``512 x 512``.
* A detector receives a float32 torch tensor ``[T, 3, H, W]`` in [0, 1] (``to_tensor``).
* Attacks work on such float tensors. ``realize`` rounds them back to the 8-bit grid (round to
  nearest). Nothing is ever scored without being realized first, so every reported score is the
  score of a clip that can be saved as 8-bit frames.
"""
from __future__ import annotations

import numpy as np

T = 11          # frames per official test input
SIZE = 512      # height and width of the official test input


def check_frames(frames) -> np.ndarray:
    """Validate and return uint8 frames ``[T, H, W, 3]``."""
    a = np.asarray(frames)
    if a.dtype != np.uint8:
        raise TypeError(f"frames must be uint8 [T, H, W, 3], got dtype {a.dtype}")
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"frames must be uint8 [T, H, W, 3] (RGB), got shape {a.shape}")
    return np.ascontiguousarray(a)


def to_tensor(frames_u8: np.ndarray):
    """uint8 ``[T, H, W, 3]`` -> float32 torch tensor ``[T, 3, H, W]`` in [0, 1]."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(frames_u8)).permute(0, 3, 1, 2).float().div_(255.0)


def realize(x) -> np.ndarray:
    """Round a float clip (torch ``[T, 3, H, W]`` or numpy ``[T, H, W, 3]``) in [0, 1] to uint8
    ``[T, H, W, 3]``: the clip a detector actually receives."""
    import torch
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().permute(0, 2, 3, 1).numpy()
    x = np.asarray(x, dtype=np.float64)
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def linf_levels(a_u8: np.ndarray, b_u8: np.ndarray) -> int:
    """l_inf distance between two uint8 clips, in gray levels."""
    return int(np.abs(a_u8.astype(np.int16) - b_u8.astype(np.int16)).max())


__all__ = ["T", "SIZE", "check_frames", "to_tensor", "realize", "linf_levels"]
