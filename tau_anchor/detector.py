"""Tau-Anchor as a CivicBench detector.

    from tau_anchor import TauAnchorDetector
    det = TauAnchorDetector.from_pretrained(device="cuda")      # downloads the weights and the base encoder
    det(x)                                                      # x: float32 [T, 3, H, W] in [0, 1] -> 0-d score

or from the command line (run from the repository root):

    civicbench evaluate --detector tau_anchor.detector:build --factory-kwargs '{"device": "cuda"}' --data <civicbench>

Contract (``civicbench.attack`` / ``civicbench.evaluate``)
---------------------------------------------------------
* Input: float tensor ``[T, 3, H, W]`` in [0, 1], RGB (the official input is ``[11, 3, 512, 512]``), or a batch
  ``[B, T, 3, H, W]``. The detector resizes each frame to 336 x 336 itself, so attacks act on the pixels a
  viewer sees.
* Output: one score per clip (a 0-d tensor for one clip), higher = generated, differentiable in the input. A clip is
  flagged when ``score >= tau``. Every parameter is frozen; gradients flow to the input only.
* Thresholds: ``tau`` is the deployed threshold, set in training at 5% false alarms on the 2,000 real test clips of
  AIGVDBench fold 0. ``tau_civicbench`` is the threshold at 5% false alarms on CivicBench's real clips, the one
  behind the paper's CivicBench table. ``civicbench.evaluate`` sets its own threshold on the real clips it is given
  and does not read either value.
"""
from __future__ import annotations

from typing import Optional

import torch

from tau_anchor.model import TauAnchor, load

# TO BE CONFIRMED at release: the Hugging Face repo id and file name of the released weights.
DEFAULT_WEIGHTS = "KurbanIntelligenceLab/Tau-Anchor"
WEIGHTS_FILE = "tau_anchor_aigvd_fold0.pt"

# Thresholds of the released model (AIGVDBench fold 0). TAU_DEPLOYED was computed in training from scores under
# 16-bit mixed precision; TAU_CIVICBENCH from 32-bit scores, as ``TauAnchorDetector`` computes them.
TAU_DEPLOYED = 1.09765625
TAU_CIVICBENCH = 1.7751145362854004


class TauAnchorDetector(torch.nn.Module):
    """Callable detector for ``civicbench.attack`` and ``civicbench.evaluate``."""

    def __init__(self, model: TauAnchor, tau: float = TAU_DEPLOYED, tau_civicbench: Optional[float] = TAU_CIVICBENCH):
        super().__init__()
        self.model = model.eval()
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.tau = float(tau)
        self.tau_civicbench = None if tau_civicbench is None else float(tau_civicbench)
        self.eval()

    def forward(self, x):
        single = x.dim() == 4
        s = self.model.score((x[None] if single else x).to(torch.float32))
        return s[0] if single else s

    def flagged(self, x, tau: Optional[float] = None) -> bool:
        """Whether one clip is flagged at ``tau`` (default: the deployed threshold)."""
        with torch.no_grad():
            return float(self(x)) >= (self.tau if tau is None else float(tau))

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS, device="cpu", **kwargs) -> "TauAnchorDetector":
        """Load the weights (Hugging Face repo id or local file; see ``tau_anchor.model.load`` for ``kwargs``). The
        thresholds stored in the file win over the module constants."""
        model = load(weights, device=device, **kwargs)
        meta = getattr(model, "meta", {})
        return cls(model, meta.get("tau", TAU_DEPLOYED), meta.get("tau_civicbench", TAU_CIVICBENCH))


def build(weights: str = DEFAULT_WEIGHTS, device: Optional[str] = None, **kwargs) -> TauAnchorDetector:
    """Factory for ``civicbench evaluate --detector tau_anchor.detector:build`` (GPU when available)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return TauAnchorDetector.from_pretrained(weights, device=device, **kwargs)


__all__ = ["TauAnchorDetector", "build", "DEFAULT_WEIGHTS", "WEIGHTS_FILE", "TAU_DEPLOYED", "TAU_CIVICBENCH"]
