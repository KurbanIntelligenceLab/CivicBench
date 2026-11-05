"""Margin against reach: predict which clips keep their label at a larger budget from one small attack.

Score units, per attacked clip x at the success threshold tau (the 5% threshold in evaluate()):

    margin  m = s(x) - tau  (generated clip, evasion)     or   tau - s(x)  (real clip, framing)
    reach   r(b) = s(x) - s(x_b)  (evasion)                or   s(x_b) - s(x)  (framing)

where s(x_b) is the score of the clip the attack (PGD by default) reached at b gray levels: the
best realized score at that budget, or the successful one. A clip breaks at b when r(b) >= m.
Because the reach grows almost linearly in the budget, r(B) ~ (B / p) r(p), so a probe at budget p
predicts that a clip breaks at budget B when (B / p) r(p) >= m. With the paper's p = 1 and B = 4
this is 4 r(1) >= m. The break budget p m / r(p) (gray levels; m / r(1) for p = 1) is the budget
at which the line reaches the margin; a non-positive reach counts as a reach of 1e-12.

The "measured" share comes from the same attack's trace at B when B was run (or from an earlier
success, since an attack stops at its first success). The measured reach is a lower bound on the
true reach (an attack finds one reachable clip, not necessarily the worst), so the predicted share
that keeps its label is an upper bound.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


def _record(r) -> dict:
    return r.to_record() if hasattr(r, "to_record") else r


def margin_reach(record, attack: str = "pgd"):
    """(margin, {budget: reach}, first successful budget or None) of one attack record, or None
    when the clip was not attacked (status other than "ok") or the attack did not run."""
    r = _record(record)
    if r.get("status", "ok") != "ok" or attack not in r.get("per_attack", {}):
        return None
    pa = r["per_attack"][attack]
    s0, tau, framing = float(r["clean_score"]), float(r["tau"]), r["direction"] == "framing"
    m = (tau - s0) if framing else (s0 - tau)
    reach = {}
    for t in pa["trace"]:
        s = t["score"]
        if s is None or not math.isfinite(s):
            continue
        reach[int(t["budget"])] = (s - s0) if framing else (s0 - s)
    stopped = pa.get("cost")
    stopped = int(stopped) if stopped is not None and math.isfinite(stopped) else None
    return m, reach, stopped


def predict_survival(results: Iterable, budget: int = 4, probe_budget: int = 1, attack: str = "pgd") -> dict:
    """Predict, from each clip's margin and its reach at ``probe_budget``, the share of clips that keep
    their label at ``budget``; compare with the share measured at ``budget`` when that budget was run.

    results  AttackResult objects (from ``attack``), their ``to_record()`` dicts, or an ``evaluate()``
             result (its per-clip records are used).
    Returns ``{direction: summary}`` with, per direction: n, predicted_keep, measured_keep (None if
    ``budget`` was not run), error_pp (predicted minus measured share broken, in percentage points),
    break_budget_median, break_budget_q10, margin_median, reach_median (at the probe), reach_ratio_median
    (median r(budget) / r(probe) where both were measured) and the per-clip rows ("clips").
    """
    budget, probe_budget = int(budget), int(probe_budget)
    if not 1 <= probe_budget < budget <= 255:
        raise ValueError(f"need 1 <= probe_budget < budget <= 255, got probe {probe_budget}, budget {budget}")
    if isinstance(results, dict) and "clips" in results:
        results = results["clips"]
    by_dir = {}
    for rec in results:
        rec = _record(rec)
        mr = margin_reach(rec, attack)
        if mr is None:
            continue
        m, reach, stopped = mr
        p = probe_budget
        if p not in reach:
            if stopped is not None and stopped < p:      # broke before the probe: its own reach is the probe
                p = stopped
            else:
                continue
        rp = reach[p]
        pred_broken = budget * rp >= p * m                 # (B / p) r(p) >= m, the linear-growth prediction
        brk = p * m / max(rp, 1e-12)
        if budget in reach:
            meas = reach[budget] >= m
        elif stopped is not None and stopped < budget:     # stopped at an earlier success: broken
            meas = True
        else:
            meas = None
        ratio = reach[budget] / rp if (budget in reach and p == probe_budget and rp != 0) else None
        by_dir.setdefault(rec["direction"], []).append(
            {"clip_id": rec.get("clip_id"), "margin": m, "reach_probe": rp, "probe": p, "break_budget": brk,
             "predicted_broken": bool(pred_broken), "measured_broken": meas, "reach_ratio": ratio})
    if not by_dir:
        raise ValueError(f"no attacked clip has a {attack} trace at the probe budget {probe_budget}")
    out = {}
    for d, rows in by_dir.items():
        pred = float(np.mean([r["predicted_broken"] for r in rows]))
        meas_rows = [r["measured_broken"] for r in rows if r["measured_broken"] is not None]
        meas = float(np.mean(meas_rows)) if len(meas_rows) == len(rows) else None
        bb = np.array([r["break_budget"] for r in rows], float)
        ratios = [r["reach_ratio"] for r in rows if r["reach_ratio"] is not None]
        out[d] = {"direction": d, "attack": attack, "probe_budget": probe_budget, "budget": budget, "n": len(rows),
                  "predicted_keep": 1 - pred, "measured_keep": None if meas is None else 1 - meas,
                  "error_pp": None if meas is None else 100 * (pred - meas),
                  "break_budget_median": float(np.median(bb)), "break_budget_q10": float(np.quantile(bb, 0.1)),
                  "margin_median": float(np.median([r["margin"] for r in rows])),
                  "reach_median": float(np.median([r["reach_probe"] for r in rows])),
                  "reach_ratio_median": float(np.median(ratios)) if ratios else None,
                  "clips": rows}
    return out


__all__ = ["margin_reach", "predict_survival"]
