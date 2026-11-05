"""CivicBench: load the benchmark and attack your own video detector with the paper's protocol.

A detector is any callable ``detector(x) -> score``:

* ``x``: float32 torch tensor ``[T, 3, H, W]`` in [0, 1], RGB; the official input is
  ``[11, 3, 512, 512]`` (11 frames of 512 x 512);
* ``score``: one scalar (float, 0-d or 1-element tensor), higher = more likely generated; a clip is
  flagged when ``score >= tau``.

Gradient attacks (PGD, APGD) need the score to be differentiable in ``x``; Square needs scores only.

    import civicbench
    clips = civicbench.load("path/to/civicbench/parquet")        # or a Hugging Face repo id
    table = civicbench.evaluate(my_detector, clips)                 # Table-2 columns, printed
    r = civicbench.attack(my_detector, frames_u8, tau, "evasion")   # one clip, budgets 1, 2, 4
    civicbench.predict_survival([r, ...], budget=4, probe_budget=1)
"""
from civicbench.attacks import DEFAULT_ATTACKS, FAMILIES
from civicbench.core import DEFAULT_BUDGETS, AttackResult, attack, evaluate, format_summary, score, seed_for
from civicbench.data import DEFAULT_REPO, Clip, ClipSet, load
from civicbench.frames import realize, to_tensor
from civicbench.margin import margin_reach, predict_survival
from civicbench.metrics import auc, threshold_at_fpr

__version__ = "0.1.0"

__all__ = ["load", "attack", "evaluate", "predict_survival", "margin_reach", "score", "threshold_at_fpr", "auc",
           "AttackResult", "Clip", "ClipSet", "DEFAULT_REPO", "DEFAULT_BUDGETS", "DEFAULT_ATTACKS", "FAMILIES",
           "format_summary", "seed_for", "realize", "to_tensor", "__version__"]
