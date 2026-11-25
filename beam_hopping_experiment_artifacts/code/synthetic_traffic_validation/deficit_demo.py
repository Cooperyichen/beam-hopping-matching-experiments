"""Execute consecutive deficit cycles; outputs are integration checks, not guarantees."""

import argparse
import json
from pathlib import Path

import numpy as np

from .deficit_matching import DeficitMatchingController


def run(cycles=8, traffic_path=None, origin=0, horizon=12, seed=20261124):
    if cycles < 1 or origin < 0 or horizon < 1:
        raise ValueError("cycles and horizon must be positive; origin must be nonnegative")
    rng = np.random.default_rng(seed)
    if traffic_path is None:
        controller = DeficitMatchingController(2)
        rows = []
        for _ in range(cycles):
            plan = controller.plan(np.full((1, 1, 2), 3.0), 1)
            rows.append(controller.observe(np.ones(2), plan.nominal_service_bits))
        return {"example": "archived two-cell counterexample", "units": "bit",
                "evidence": "deterministic integration example; no statistical assumption validation",
                "cycles": rows, "checkpoint": controller.state_dict()}

    with np.load(traffic_path, allow_pickle=False) as archive:
        traffic = np.asarray(archive["traffic_bits"], dtype=float)
    if traffic.ndim != 2 or traffic.shape[0] != 8 or origin + cycles * horizon > traffic.shape[1]:
        raise ValueError("the report fixture needs 8 cells and enough consecutive traffic slots")
    if not np.all(np.isfinite(traffic)) or np.any(traffic < 0):
        raise ValueError("traffic_bits must be finite and nonnegative")
    from .revision_code.revision import channel
    controller = DeficitMatchingController(8)
    rows = []
    k, total_bandwidth_hz, slot_duration_s = 3, 500e6, .002
    for j in range(cycles):
        start = origin + j * horizon
        mu, sd = channel(horizon, start)
        # Decision-time channel statistics only. Shape is (slot, beam, cell).
        nominal = np.broadcast_to((slot_duration_s * total_bandwidth_hz / k * mu.T)[:, None, :],
                                  (horizon, k, 8))
        plan = controller.plan(nominal, k)
        # Feedback is formed only after the plan. One realized channel stream
        # updates this controller; Monte Carlo replications need separate states.
        realized_se = mu + sd * rng.normal(size=mu.shape)
        service = np.maximum(np.einsum("tbi,it->i", plan.selected, realized_se)
                             * slot_duration_s * total_bandwidth_hz / k, 0.0)
        demand = traffic[:, start:start + horizon].sum(axis=1)
        rows.append(dict(controller.observe(demand, service), origin=start,
                         next_origin=start + horizon))
    return {"example": "consecutive existing-traffic adapter", "units": "bit", "seed": seed,
            "evidence": "integration replay; no proof of capacity slack or forecast calibration",
            "traffic_path": str(traffic_path), "cycles": rows, "checkpoint": controller.state_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--traffic-npz", type=Path)
    parser.add_argument("--origin", type=int, default=800000)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a fresh path")
    result = run(args.cycles, args.traffic_npz, args.origin, args.horizon, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
