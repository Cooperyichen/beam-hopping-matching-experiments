"""Compare ported numerical primitives with the existing experiment source.

This optional migration check requires the recipient's original experiment
directory. It does not require or execute the collaborator's raw scripts.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(original_root):
    original_root = Path(original_root).resolve()
    old_path = list(sys.path)
    sys.path.insert(0, str(original_root))
    try:
        old_allocation = _load(original_root / "allocation.py", "original_allocation")
        old_forecast = _load(original_root / "forecast.py", "original_forecast")
        old_traffic = _load(original_root / "traffic.py", "original_traffic")
        old_revision = _load(original_root / "revision_code/revision.py", "original_revision")
    finally:
        sys.path[:] = old_path
    from . import allocation, forecast, traffic
    from .revision_code import revision
    import torch
    torch.set_num_threads(2)
    checks = []

    def same(name, left, right):
        np.testing.assert_array_equal(left, right)
        checks.append(name)

    mu, sd = old_allocation.report_channel(cycle=7)
    same("channel mean", mu, allocation.report_channel(cycle=7)[0])
    same("channel standard deviation", sd, allocation.report_channel(cycle=7)[1])
    for lam in (0., .5, 2., 8.):
        target = np.linspace(2., 4., 8)
        a = old_allocation.matching_plan(mu, target, lam=lam)
        b = allocation.matching_plan(mu, target, lam=lam)
        same(f"historical matching lambda={lam}", a["bandwidth"], b["bandwidth"])
    rng = np.random.default_rng(41)
    residual = rng.normal(size=(400, 8))
    same("reserve rank", old_forecast.quantile_reserve(residual), forecast.quantile_reserve(residual))
    same("fractional weights", old_forecast.fractional_weights(.3, 40), forecast.fractional_weights(.3, 40))
    series = rng.normal(size=4096)
    a = old_forecast.fit_arfima(series[:3000], 1)
    b = forecast.fit_arfima(series[:3000], 1)
    for key in ("d", "phi", "mean", "scale", "objective"):
        same(f"Whittle fit {key}", a[key], b[key])
    weights = old_forecast.step_weights(a, 32, 12)
    same("covariance predictor", weights, forecast.step_weights(b, 32, 12))
    same("strictly past forecast", old_forecast.predict_at(series, [3100, 3200], weights.sum(0), a["mean"], 12),
         forecast.predict_at(series, [3100, 3200], weights.sum(0), b["mean"], 12))
    a = old_traffic.simulate_cell(8, 1000, .002, 1.6, .1, 1e6, 41, "cpu", batch=4)
    b = traffic.simulate_cell(8, 1000, .002, 1.6, .1, 1e6, 41, "cpu", batch=4)
    same("Pareto integrated traffic", a, b)
    for horizon in (12, 24, 48):
        a = old_revision.channel(horizon, 800084)
        b = revision.channel(horizon, 800084)
        same(f"revision channel mean T={horizon}", a[0], b[0])
        same(f"revision channel deviation T={horizon}", a[1], b[1])
    return {"seed": 41, "passed": len(checks), "checks": checks,
            "scope": "finite input numerical parity with the original experiment implementation; not model equivalence to the raw collaborator scripts"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.original_root), indent=2))


if __name__ == "__main__":
    main()
