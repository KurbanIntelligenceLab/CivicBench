"""The attacks of the CivicBench protocol, each behind one signature ``run(target, eps, rng) -> Result``.

All attacks MINIMIZE the target's score (``Target`` negates it for framing) inside the l_inf ball
of radius ``eps`` (in [0, 1] units; ``eps = b / 255`` for a budget of ``b`` gray levels), and stop
at the first REALIZED success: a candidate rounded to 8 bits whose score is below the target's
threshold.

* PGD: 100 steps of size eps / 4 along the sign of the gradient, one restart from the clean clip,
  the realized score checked every 10 steps and after the last step (10 checks).
* APGD (Croce & Hein, ICML 2020): 100 iterations x 5 restarts, momentum 0.75, step 2 eps halved at
  the checkpoints when progress stalls (rho = 0.75), restart from the best point; the loss is the
  score itself. Restart 0 starts at the clean clip, later restarts uniformly at random in the
  ball. The realized score of the best iterate is checked every 10 iterations (50 checks).
* Square (Andriushchenko et al., ECCV 2020), video variant: each query sets a square patch of ONE
  random frame to x0 +/- eps (independent sign per channel) and keeps the change if the score
  drops; the patch area follows Square's schedule p(i) with p_init = 0.2. No stripe
  initialization. 1,000 queries per budget.
* SimBA-DCT (Guo et al., ICML 2019), l_inf variant: walks low-frequency 2-D DCT directions of
  single frames and channels (scaled to max-abs 1, step eps), trying -q then +q and keeping
  improvements. 1,000 queries per budget. Not part of the default set (the protocol uses it only
  for motion-vector detectors).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Result:
    evaded: bool
    score: float                # realized score of the returned clip (lowest seen if not successful)
    queries: int                # realized queries spent by this run
    x: Optional[object] = None  # float clip [T, 3, H, W] (the successful one, else the best seen)
    extra: dict = field(default_factory=dict)


def project(x, x0, eps: float):
    """Project onto the l_inf ball of radius eps around x0, intersected with [0, 1]."""
    return x.clamp(min=x0 - eps, max=x0 + eps).clamp_(0.0, 1.0)


# ---- first order ---------------------------------------------------------------------------
def _start(target, eps, restart, rng):
    x0 = target.x0
    if restart == 0:
        return x0.clone()
    torch = target._torch
    gen = torch.Generator(device="cpu").manual_seed(int(rng.integers(2**31)))
    u = (torch.rand(tuple(x0.shape), generator=gen) * 2 - 1).to(x0.device) * eps
    return project(x0 + u, x0, eps)


def pgd(target, eps: float, steps: int = 100, restarts: int = 1, step_size: float = None,
        check_every: int = 10, eot: int = 0, eot_sigma: float = 1 / 255, rng=None) -> Result:
    rng = rng or np.random.default_rng(0)
    alpha = step_size if step_size is not None else eps / 4
    q0, best = target.queries, Result(False, math.inf, 0)
    for r in range(restarts):
        x = _start(target, eps, r, rng)
        for i in range(1, steps + 1):
            g = target.grad(x, eot=eot, eot_sigma=eot_sigma, rng=rng)
            x = project(x - alpha * g.sign(), target.x0, eps)
            if i % check_every == 0 or i == steps:
                s = target.score(x)
                if s < best.score:
                    best = Result(False, s, 0, x.clone())
                if s < target.tau:
                    return Result(True, s, target.queries - q0, x.clone(), {"restart": r, "step": i})
    best.queries = target.queries - q0
    return best


def _checkpoints(steps: int):
    p = [0.0, 0.22]
    while p[-1] < 1:
        p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
    return sorted({int(math.ceil(q * steps)) for q in p[1:] if q <= 1})


def apgd(target, eps: float, steps: int = 100, restarts: int = 5, check_every: int = 10,
         eot: int = 0, eot_sigma: float = 1 / 255, rng=None, rho: float = 0.75,
         alpha: float = 0.75) -> Result:
    rng = rng or np.random.default_rng(0)
    torch = target._torch
    x0 = target.x0
    q0, best = target.queries, Result(False, math.inf, 0)

    def loss(x):
        with torch.no_grad():
            return float(target.differentiable_score(x.unsqueeze(0))[0])

    checks = _checkpoints(steps)
    for r in range(restarts):
        x_prev = _start(target, eps, r, rng)
        f_prev = loss(x_prev)
        eta = 2 * eps
        x = project(x_prev - eta * target.grad(x_prev, eot, eot_sigma, rng).sign(), x0, eps)
        f_x = loss(x)
        x_best, f_best = (x, f_x) if f_x < f_prev else (x_prev, f_prev)
        last_check, n_improved, eta_at_check, f_best_at_check = 0, int(f_x < f_prev), eta, f_best
        for k in range(1, steps):
            z = project(x - eta * target.grad(x, eot, eot_sigma, rng).sign(), x0, eps)
            x_new = project(x + alpha * (z - x) + (1 - alpha) * (x - x_prev), x0, eps)
            f_new = loss(x_new)
            n_improved += int(f_new < f_x)
            x_prev, x, f_x = x, x_new, f_new
            if f_new < f_best:
                x_best, f_best = x_new.clone(), f_new
            if (k + 1) % check_every == 0 or k == steps - 1:
                s = target.score(x_best)
                if s < best.score:
                    best = Result(False, s, 0, x_best.clone())
                if s < target.tau:
                    return Result(True, s, target.queries - q0, x_best.clone(), {"restart": r, "step": k + 1})
            if k + 1 in checks:
                window = k + 1 - last_check
                stalled = n_improved < rho * window
                frozen = eta == eta_at_check and f_best == f_best_at_check
                if stalled or frozen:
                    eta /= 2
                    x = x_best.clone()
                last_check, n_improved, eta_at_check, f_best_at_check = k + 1, 0, eta, f_best
    best.queries = target.queries - q0
    return best


# ---- zeroth order (scores only) ------------------------------------------------------------
def square_schedule(p_init: float, i: int, n_queries: int) -> float:
    it = int(i / max(n_queries, 1) * 10000)
    for bound, div in ((10, 1), (50, 2), (200, 4), (500, 8), (1000, 16), (2000, 32), (4000, 64),
                       (6000, 128), (8000, 256)):
        if it <= bound:
            return p_init / div
    return p_init / 512


def square(target, eps: float, n_queries: int = 1000, p_init: float = 0.2, rng=None) -> Result:
    rng = rng or np.random.default_rng(0)
    torch = target._torch
    x0 = target.x0
    t_len, ch, h, w = x0.shape
    q0 = target.queries
    x_best, s_best = x0.clone(), target.score(x0)
    if s_best < target.tau:
        return Result(True, s_best, target.queries - q0, x_best)
    while target.queries - q0 < n_queries:
        i = target.queries - q0
        side = int(round(math.sqrt(square_schedule(p_init, i, n_queries) * h * w)))
        side = min(max(side, 1), h - 1, w - 1)
        t = int(rng.integers(t_len))
        r, c = int(rng.integers(0, h - side)), int(rng.integers(0, w - side))
        signs = torch.from_numpy(rng.choice([-1.0, 1.0], size=(ch, 1, 1))).to(x0)
        x_new = x_best.clone()
        x_new[t, :, r:r + side, c:c + side] = (x0[t, :, r:r + side, c:c + side] + eps * signs).clamp(0, 1)
        s_new = target.score(x_new)
        if s_new < s_best:
            x_best, s_best = x_new, s_new
            if s_best < target.tau:
                return Result(True, s_best, target.queries - q0, x_best)
    return Result(False, s_best, target.queries - q0, x_best)


def dct_basis(h: int, w: int, u: int, v: int, device=None):
    import torch
    y = torch.arange(h, dtype=torch.float32, device=device)
    x = torch.arange(w, dtype=torch.float32, device=device)
    by = torch.cos(math.pi * (2 * y + 1) * u / (2 * h))
    bx = torch.cos(math.pi * (2 * x + 1) * v / (2 * w))
    return by[:, None] * bx[None, :]                     # max |.| = 1


def simba_dct(target, eps: float, n_queries: int = 1000, freq_frac: float = 1 / 8, rng=None) -> Result:
    rng = rng or np.random.default_rng(0)
    x0 = target.x0
    t_len, ch, h, w = x0.shape
    nu, nv = max(1, int(h * freq_frac)), max(1, int(w * freq_frac))
    q0 = target.queries
    x, s = x0.clone(), target.score(x0)
    if s < target.tau:
        return Result(True, s, target.queries - q0, x)
    order = rng.permutation(t_len * ch * nu * nv)
    k = 0
    while target.queries - q0 < n_queries:
        idx = int(order[k % len(order)])
        k += 1
        t, rem = divmod(idx, ch * nu * nv)
        c, rem = divmod(rem, nu * nv)
        u, v = divmod(rem, nv)
        q = dct_basis(h, w, u, v, device=x0.device) * eps
        for sgn in (-1.0, 1.0):
            if target.queries - q0 >= n_queries:
                break
            x_try = x.clone()
            x_try[t, c] = x_try[t, c] + sgn * q
            x_try = project(x_try, x0, eps)
            s_try = target.score(x_try)
            if s_try < s:
                x, s = x_try, s_try
                if s < target.tau:
                    return Result(True, s, target.queries - q0, x)
                break
    return Result(False, s, target.queries - q0, x)


# ---- the protocol's settings ---------------------------------------------------------------
FAMILIES = {
    "pgd": lambda target, eps, rng=None: pgd(target, eps, steps=100, restarts=1, rng=rng),
    "apgd": lambda target, eps, rng=None: apgd(target, eps, steps=100, restarts=5, rng=rng),
    "square": lambda target, eps, rng=None: square(target, eps, n_queries=1000, rng=rng),
    "simba": lambda target, eps, rng=None: simba_dct(target, eps, n_queries=1000, rng=rng),
}
GRADIENT_ATTACKS = ("pgd", "apgd")
DEFAULT_ATTACKS = ("pgd", "apgd", "square")

__all__ = ["Result", "project", "pgd", "apgd", "square", "square_schedule", "simba_dct", "dct_basis",
           "FAMILIES", "GRADIENT_ATTACKS", "DEFAULT_ATTACKS"]
