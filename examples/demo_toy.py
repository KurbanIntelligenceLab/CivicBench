#!/usr/bin/env python3
"""End-to-end demo with no downloads and no weights: 40 synthetic clips and a toy detector.

    python examples/demo_toy.py

Builds 20 "real" and 20 "generated" clips of 11 frames at 32 x 32 (generated clips are brighter and
flicker less), scores them with a small differentiable detector, then runs
``civicbench.evaluate`` with the paper's defaults: threshold at 5% false alarms on the real clips,
PGD, APGD and Square at 1, 2 and 4 gray levels, evasion and framing, and the margin-against-reach
prediction. Takes well under a minute on a laptop CPU and well under 1 GB of memory.
"""
import numpy as np
import torch

import civicbench

T, SIZE, N = 11, 32, 20


class ToyDetector(torch.nn.Module):
    """score = brightness - flicker + a colour term (higher = generated). Differentiable, no weights."""

    def forward(self, x):                      # x: [T, 3, H, W] in [0, 1]
        bright = x.mean()
        flicker = (x[1:] - x[:-1]).abs().mean()
        colour = torch.tanh(8.0 * (x[:, 0] - x[:, 2])).mean()
        return 30.0 * (bright - 0.5) - 20.0 * flicker + 2.0 * colour


def make_clips(seed: int = 0):
    """20 real and 20 generated uint8 clips [11, 32, 32, 3]; generated ones are brighter and flicker less."""
    rng = np.random.default_rng(seed)
    clips = []
    for i in range(2 * N):
        generated = i % 2 == 1
        base = rng.integers(70, 170, size=(1, SIZE, SIZE, 3)).astype(np.float64)
        drift = rng.normal(0, 0.5, size=(T, 1, 1, 1)).cumsum(0)
        noise = rng.normal(0, 2.0 if generated else 3.0, size=(T, SIZE, SIZE, 3))
        shift = rng.normal(5.0, 4.0) if generated else rng.normal(0.0, 2.0)
        frames = np.clip(np.rint(base + drift + noise + shift), 0, 255).astype(np.uint8)
        clips.append((f"toy_{'gen' if generated else 'real'}_{i:02d}", frames, int(generated),
                      {"generator": f"toygen{i % 3}" if generated else None}))
    return clips


def main():
    torch.set_num_threads(1)
    det = ToyDetector().eval()
    clips = make_clips()
    res = civicbench.evaluate(det, clips)          # default budgets (1, 2, 4) and attacks (pgd, apgd, square)
    s = res.get("survival", {})
    for d in ("evasion", "framing"):
        if d in s:
            print(f"{d}: median break budget {s[d]['break_budget_median']:.2f} gray levels; "
                  f"reach grows x{s[d]['reach_ratio_median'] or float('nan'):.2f} from 1 to 4 gray levels")


if __name__ == "__main__":
    main()
