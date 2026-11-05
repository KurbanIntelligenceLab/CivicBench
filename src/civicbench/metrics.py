"""Operating point and clean metrics (numpy only).

* ``threshold_at_fpr(real_scores, fpr)`` returns the smallest threshold whose false-alarm rate on
  the given real clips does not exceed ``fpr`` (the largest TPR at that rate). A clip is flagged
  when ``score >= tau``.
* ``auc(scores, labels)`` is the area under the ROC curve (ties count one half), the same value as
  scikit-learn's ``roc_auc_score``.
"""
from __future__ import annotations

import numpy as np


def threshold_at_fpr(real_scores, fpr: float) -> float:
    """Smallest tau with mean(real >= tau) <= fpr; above every real score if none qualifies."""
    s = np.sort(np.asarray(real_scores, float))[::-1]
    if s.size == 0 or not np.isfinite(s).all():
        raise ValueError("need finite real scores")
    allowed = int(np.floor(fpr * s.size + 1e-9))
    above = float(np.nextafter(s[0], np.inf))
    if allowed == 0:
        return above
    # candidate thresholds: distinct score values; keep the lowest one within budget
    vals, first = np.unique(s[::-1], return_index=True)          # ascending distinct values
    count_ge = s.size - first                                      # reals with score >= value
    ok = count_ge <= allowed
    return float(vals[ok].min()) if ok.any() else above


def auc(scores, labels) -> float:
    """Area under the ROC curve of ``scores`` for binary ``labels`` (1 = generated)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels).astype(int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("AUC needs both real (0) and generated (1) clips")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    while i < len(s):                                   # average ranks over ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


__all__ = ["threshold_at_fpr", "auc"]
