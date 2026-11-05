"""What an attack may observe about one clip.

Contract
--------
* A detector is a callable ``detector(x) -> score``: ``x`` is a float32 torch tensor
  ``[T, 3, H, W]`` in [0, 1] (RGB; the official input is ``[11, 3, 512, 512]``), the score is a
  scalar (Python float, 0-d tensor or 1-element tensor), higher for generated clips. A clip is
  flagged when ``score >= tau``. Gradient attacks need the score to be differentiable in ``x``;
  score-based attacks (Square, SimBA) need only its value.
* The attacks always MINIMIZE ``sign * score`` toward ``attack_tau``. Evasion uses ``sign = +1``
  and the detector's threshold. Framing uses ``sign = -1`` and ``attack_tau = -tau``: lowering
  the negated score below ``-tau`` raises the real score above ``tau``.
* ``score(x)`` is ONE query: the float clip ``x`` is realized (rounded to 8 bits), converted back
  to a tensor and scored. A non-finite score counts as ``+inf`` (never a success). Every query
  increments ``queries``.
* ``evades(x) == score(x) < attack_tau``.
* ``verify(x)`` -> (realized score, realized l_inf distance to the clean clip in gray levels). It
  is not counted: it re-checks what an attack returns and never guides an attack.
* ``grad(x)`` is the gradient of the detector's score at the float (unrealized) ``x``. With
  ``eot > 0`` it averages over that many copies of ``x`` perturbed by uniform noise of half-width
  ``eot_sigma``. ``grad`` does not count as a query (first-order attacks are compared by budget,
  not queries). Where the score carries no gradient, ``grad`` is zero.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from civicbench import frames as F


def _as_float(s) -> float:
    import torch
    if isinstance(s, torch.Tensor):
        if s.numel() != 1:
            raise ValueError(f"the detector must return ONE score per clip, got a tensor of shape {tuple(s.shape)}")
        return float(s.detach().reshape(()).item())
    return float(s)


class Target:
    def __init__(self, detector, attack_tau: float, frames_u8: np.ndarray, sign: float = 1.0, device=None):
        import torch
        self.detector, self.tau, self.sign = detector, float(attack_tau), float(sign)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.x0_u8 = F.check_frames(frames_u8)
        self.x0 = F.to_tensor(self.x0_u8).to(self.device)
        self.queries = 0
        self._torch = torch

    # ---- queries ---------------------------------------------------------------------------
    def _score_u8(self, u8: np.ndarray) -> float:
        with self._torch.no_grad():
            s = _as_float(self.detector(F.to_tensor(u8).to(self.device)))
        return self.sign * s if math.isfinite(s) else math.inf

    def score(self, x) -> float:
        self.queries += 1
        return self._score_u8(F.realize(x))

    def evades(self, x) -> bool:
        return self.score(x) < self.tau

    def verify(self, x):
        u8 = F.realize(x)
        return self._score_u8(u8), F.linf_levels(u8, self.x0_u8)

    # ---- gradients -------------------------------------------------------------------------
    def differentiable_score(self, x):
        """``sign * detector`` on a batch ``[B, T, 3, H, W]`` -> ``[B]`` (one detector call per clip)."""
        torch = self._torch
        out = []
        for xi in x:
            s = self.detector(xi)
            if not isinstance(s, torch.Tensor):
                s = torch.as_tensor(float(s), dtype=torch.float64, device=xi.device)
            out.append(s.reshape(()))
        return self.sign * torch.stack(out)

    def grad(self, x, eot: int = 0, eot_sigma: float = 1 / 255, rng: Optional[np.random.Generator] = None):
        torch = self._torch
        if eot > 0:
            gen = torch.Generator(device="cpu").manual_seed(int((rng or np.random.default_rng()).integers(2**31)))
            noise = (torch.rand((eot,) + tuple(x.shape), generator=gen) * 2 - 1) * eot_sigma
            batch = (x.detach().unsqueeze(0) + noise.to(x.device)).clamp_(0, 1)
        else:
            batch = x.detach().unsqueeze(0)
        batch.requires_grad_(True)
        with torch.enable_grad():
            s = self.differentiable_score(batch)
            if not s.requires_grad:
                return torch.zeros_like(x)
            (g,) = torch.autograd.grad(s.sum(), batch, allow_unused=True)
        return torch.zeros_like(x) if g is None else g.mean(0).to(x.dtype)


__all__ = ["Target"]
