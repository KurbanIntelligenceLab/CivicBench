"""Cost to break a clip: the smallest budget at which an attack realizes a success.

``cost_to_evade(run, target, budgets)`` calls ``run(target, b / 255)`` at the budgets (gray levels,
ascending) and stops at the first realized success. Every claimed success is re-checked with
``target.verify`` (realized score below the attack threshold and realized l_inf distance within the
budget); a failed check raises, because it means an attack's bookkeeping is wrong. The result
holds the cost (inf when no budget succeeds), the queries spent at that budget, the successful
clip and a per-budget trace.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Sequence


def check_budgets(budgets) -> tuple:
    """Validate budgets: positive integer gray levels 1..255, returned sorted and deduplicated."""
    if isinstance(budgets, (int, float)) and not isinstance(budgets, bool):
        budgets = (budgets,)
    try:
        items = list(budgets)
    except TypeError:
        raise ValueError(f"budgets must be a sequence of integer gray levels, got {budgets!r}") from None
    out = set()
    for b in items:
        if isinstance(b, bool) or not hasattr(b, "__index__"):
            if isinstance(b, float) and b.is_integer():
                b = int(b)
            else:
                raise ValueError(f"budgets are integer gray levels on the 0-255 grid; got {b!r}")
        b = int(b)
        if not 1 <= b <= 255:
            raise ValueError(f"budgets must lie in 1..255 gray levels; got {b}")
        out.add(b)
    if not out:
        raise ValueError("budgets is empty; give at least one gray level, e.g. (1, 2, 4)")
    return tuple(sorted(out))


def cost_to_evade(run: Callable, target, budgets: Sequence[int]) -> dict:
    trace = []
    for b in check_budgets(budgets):
        t0 = time.perf_counter()
        r = run(target, b / 255.0)
        rec = {"budget": b, "evaded": bool(r.evaded), "score": float(r.score),
               "queries": int(r.queries), "seconds": round(time.perf_counter() - t0, 3)}
        trace.append(rec)
        if r.evaded:
            score, levels = target.verify(r.x)
            if not (score < target.tau and levels <= b):
                raise RuntimeError(f"claimed success at {b}/255 does not verify: realized score "
                                   f"{score:.6g} vs threshold {target.tau:.6g}, l_inf {levels} levels")
            rec["linf_levels"] = levels
            return {"cte": b, "queries": int(r.queries), "x": r.x, "trace": trace}
    return {"cte": math.inf, "queries": None, "x": None, "trace": trace}


__all__ = ["cost_to_evade", "check_budgets"]
