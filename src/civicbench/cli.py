"""Command line: ``civicbench evaluate --detector module:factory --data PATH_OR_REPO [...]``.

``module:factory`` names a callable that BUILDS the detector: it is imported (the current
directory is on the import path), called with the keyword arguments of ``--factory-kwargs``
(JSON), and must return the detector callable (see ``civicbench.attack``).
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys


def build_detector(spec: str, kwargs: dict):
    if ":" not in spec:
        raise SystemExit(f"--detector must look like module:factory, got {spec!r}")
    mod_name, attr = spec.split(":", 1)
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    det = obj(**kwargs)
    if not callable(det):
        raise SystemExit(f"{spec} returned {type(det).__name__}, which is not callable")
    if hasattr(det, "eval") and callable(det.eval):
        det.eval()
    return det


def _ints(s: str):
    try:
        return tuple(int(x) for x in s.split(",") if x.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {s!r}") from None


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return _jsonable(o.item())
    return o


def main(argv=None) -> int:
    import civicbench
    p = argparse.ArgumentParser(prog="civicbench", description="Evaluate a video detector on CivicBench.")
    p.add_argument("--version", action="version", version=f"civicbench {civicbench.__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="clean AUC/TPR and attacked rates (the paper's CivicBench table)")
    e.add_argument("--detector", required=True, help="module:factory returning the detector callable")
    e.add_argument("--factory-kwargs", default="{}", help="JSON keyword arguments for the factory")
    e.add_argument("--data", nargs="+", default=None,
                   help="Hugging Face repo id(s) or local parquet directories/files (real and generated)")
    e.add_argument("--split", default="test")
    e.add_argument("--limit", type=int, default=None, help="read at most this many clips (per run, for smoke tests)")
    e.add_argument("--fpr", type=float, default=0.05)
    e.add_argument("--strict-fpr", type=float, default=0.01, help="threshold framing attacks aim at")
    e.add_argument("--budgets", type=_ints, default=(1, 2, 4), help="gray levels, e.g. 1,2,4,8,16 (default 1,2,4)")
    e.add_argument("--attacks", default="pgd,apgd,square", help="comma list from: " + ",".join(civicbench.FAMILIES))
    e.add_argument("--n-evasion", type=int, default=None)
    e.add_argument("--n-framing", type=int, default=None)
    e.add_argument("--evasion-per-generator", type=int, default=None)
    e.add_argument("--seed", type=int, default=None, help="default: the paper's per-clip seeds")
    e.add_argument("--survival-budget", type=int, default=None)
    e.add_argument("--survival-probe", type=int, default=None)
    e.add_argument("--device", default=None)
    e.add_argument("--no-verify", action="store_true", help="skip the frames_sha256 check")
    e.add_argument("--out", default=None, help="write the full result (per-clip records included) as JSON")
    e.add_argument("--quiet", action="store_true", help="no progress line and no summary table")
    a = p.parse_args(argv)

    det = build_detector(a.detector, json.loads(a.factory_kwargs))
    clips = civicbench.load(a.data, split=a.split, limit=a.limit, verify=not a.no_verify, quiet=a.quiet)
    res = civicbench.evaluate(det, clips, fpr=a.fpr, strict_fpr=a.strict_fpr, budgets=a.budgets,
                              attacks=tuple(x for x in a.attacks.split(",") if x), n_evasion=a.n_evasion,
                              n_framing=a.n_framing, evasion_per_generator=a.evasion_per_generator, seed=a.seed,
                              device=a.device, survival_budget=a.survival_budget, survival_probe=a.survival_probe,
                              quiet=a.quiet)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(_jsonable(res), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
