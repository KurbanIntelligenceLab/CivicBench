"""Tau-Anchor, the robust detector of the CivicBench paper: a Delta-CLIP frame encoder fine-tuned with attacks on
both classes, a margin at its deployed threshold, a preservation term, generator weights and weight averaging.

    from tau_anchor import TauAnchorDetector
    det = TauAnchorDetector.from_pretrained(device="cuda")
    score = det(x)                  # x: float32 [T, 3, H, W] in [0, 1]; flagged when score >= det.tau

Training: ``python -m tau_anchor.train --help`` and the README.
"""
from tau_anchor.detector import (DEFAULT_WEIGHTS, TAU_CIVICBENCH, TAU_DEPLOYED, WEIGHTS_FILE, TauAnchorDetector,
                                 build)
from tau_anchor.model import TauAnchor, VisionTransformer, load, load_base_encoder

__all__ = ["TauAnchorDetector", "TauAnchor", "VisionTransformer", "build", "load", "load_base_encoder",
           "DEFAULT_WEIGHTS", "WEIGHTS_FILE", "TAU_DEPLOYED", "TAU_CIVICBENCH"]
