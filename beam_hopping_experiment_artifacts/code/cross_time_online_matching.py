#!/usr/bin/env python3
"""Phase 4 cross-time beam hopping matching prototype.

The model keeps the original single-slot matching constraints and restores
the cross-time demand lower bound. The implemented policies are deliberately
simple and auditable:

1. throughput_greedy: solve each slot using raw edge weights.
2. urgency_primal_dual: increase cell prices when residual demand is high.
3. prediction_aware: blend predicted-demand urgency with realized-demand
   urgency using a trust weight that decays with observed prediction error.

These are prototypes for comparing cross-time coupling strategies. They are
not final literature-calibrated implementations of DRL, MADRL, or OCO papers.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from phase3_benchmarks import greedy_matching, selected_matrix
from single_slot_matching import Instance, generate_instance


@dataclass(frozen=True)
class HorizonInstance:
    instance_id: str
    slots: list[Instance]
    demand_rate: np.ndarray
    predicted_demand_rate: np.ndarray

    @property
    def T(self) -> int:
        return len(self.slots)

    @property
    def M(self) -> int:
        return self.slots[0].M

    @property
    def N(self) -> int:
        return self.slots[0].N

    @property
    def K(self) -> int:
        return self.slots[0].K

    @property
    def T_slot(self) -> float:
        return self.slots[0].T_slot

    @property
    def Q(self) -> np.ndarray:
        return self.T_slot * self.demand_rate.sum(axis=0)

    @property
    def Q_hat(self) -> np.ndarray:
        return self.T_slot * self.predicted_demand_rate.sum(axis=0)


def generate_horizon_instance(
    *,
    instance_id: str,
    T: int,
    M: int,
    N: int,
    K: int,
    topology: str,
    demand_profile: str,
    prediction_mode: str,
    edge_density: float,
    seed: int,
) -> HorizonInstance:
    rng = np.random.default_rng(seed)
    slots: list[Instance] = []
    demand_rates = []
    for t in range(T):
        inst = generate_instance(
            instance_id=f"{instance_id}_t{t+1}",
            M=M,
            N=N,
            K=K,
            topology=topology,
            edge_density=edge_density,
            demand_profile=demand_profile,
            seed=seed + 1009 * t,
        )
        slots.append(inst)
        demand_rates.append(inst.demand_rate)
    demand_rate = np.vstack(demand_rates)
    predicted = make_prediction(demand_rate, prediction_mode, rng)
    return HorizonInstance(instance_id, slots, demand_rate, predicted)


def make_prediction(demand_rate: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "accurate":
        noise = rng.normal(0.0, 0.03, size=demand_rate.shape)
        return np.maximum(demand_rate * (1.0 + noise), 0.0)
    if mode == "noisy":
        noise = rng.normal(0.0, 0.30, size=demand_rate.shape)
        return np.maximum(demand_rate * (1.0 + noise), 0.0)
    if mode == "adversarial":
        flipped = demand_rate[:, ::-1]
        noise = rng.normal(0.0, 0.20, size=demand_rate.shape)
        return np.maximum(flipped * (1.0 + noise), 0.0)
    raise ValueError(f"Unknown prediction_mode: {mode}")


def score_instance(instance: Instance, score: np.ndarray) -> Instance:
    return Instance(
        instance_id=instance.instance_id,
        M=instance.M,
        N=instance.N,
        K=instance.K,
        T_slot=instance.T_slot,
        B_tot=instance.B_tot,
        P_tot=instance.P_tot,
        sigma2=instance.sigma2,
        gain=(score / max(instance.T_slot, 1e-12)),
        demand_rate=instance.demand_rate,
        feasible_edges=instance.feasible_edges,
    )


def greedy_by_score(instance: Instance, score: np.ndarray) -> np.ndarray:
    # Reuse the feasible greedy routine by temporarily overriding weight
    # through a lightweight object with compatible attributes.
    proxy = _ProxyInstance(instance, score)
    return selected_matrix(instance, greedy_matching(proxy))


class _ProxyInstance:
    def __init__(self, base: Instance, weight: np.ndarray):
        self.M = base.M
        self.N = base.N
        self.K = base.K
        self.feasible_edges = base.feasible_edges
        self.weight = weight


def run_policy(horizon: HorizonInstance, policy: str, *, kappa: float = 2.0) -> dict:
    residual = horizon.Q.copy()
    predicted_residual = horizon.Q_hat.copy()
    delivered_total = np.zeros(horizon.N)
    total_throughput = 0.0
    cumulative_error = 0.0
    reoptimization_count = 0
    start = time.perf_counter()

    for t, slot in enumerate(horizon.slots):
        remaining = max(horizon.T - t, 1)
        base_weight = slot.weight
        if policy == "throughput_greedy":
            score = base_weight
        elif policy == "urgency_primal_dual":
            urgency = np.maximum(residual, 0.0) / remaining
            normalized = urgency / max(float(np.mean(urgency) + 1e-9), 1e-9)
            score = base_weight * (1.0 + normalized.reshape(1, horizon.N))
        elif policy == "prediction_aware":
            if t > 0:
                cumulative_error += float(np.sum(np.abs(horizon.demand_rate[t - 1] - horizon.predicted_demand_rate[t - 1])))
            scale = float(np.sum(horizon.demand_rate[: t + 1]) + 1e-9)
            theta = math.exp(-kappa * cumulative_error / scale)
            realized_urgency = np.maximum(residual, 0.0) / remaining
            predicted_urgency = np.maximum(predicted_residual, 0.0) / remaining
            realized_norm = realized_urgency / max(float(np.mean(realized_urgency) + 1e-9), 1e-9)
            predicted_norm = predicted_urgency / max(float(np.mean(predicted_urgency) + 1e-9), 1e-9)
            blended = (1.0 - theta) * realized_norm + theta * predicted_norm
            score = base_weight * (1.0 + blended.reshape(1, horizon.N))
            if theta < 0.5:
                reoptimization_count += 1
        else:
            raise ValueError(f"Unknown policy: {policy}")

        x = greedy_by_score(slot, score)
        delivered = (slot.T_slot * slot.rate * x).sum(axis=0)
        delivered_total += delivered
        total_throughput += float(np.sum(delivered))
        residual -= delivered
        predicted_residual -= delivered

    runtime = time.perf_counter() - start
    unmet = np.maximum(horizon.Q - delivered_total, 0.0)
    satisfaction = np.divide(
        delivered_total,
        horizon.Q,
        out=np.zeros_like(delivered_total),
        where=horizon.Q > 0,
    )
    prediction_error = float(np.sum(np.abs(horizon.demand_rate - horizon.predicted_demand_rate)))
    return {
        "instance_id": horizon.instance_id,
        "policy": policy,
        "T": horizon.T,
        "M": horizon.M,
        "N": horizon.N,
        "K": horizon.K,
        "total_throughput": total_throughput,
        "total_unmet_demand": float(np.sum(unmet)),
        "mean_demand_satisfaction_ratio": float(np.mean(np.minimum(satisfaction, 1.0))),
        "min_demand_satisfaction_ratio": float(np.min(np.minimum(satisfaction, 1.0))),
        "prediction_error": prediction_error,
        "runtime": runtime,
        "reoptimization_count": reoptimization_count,
    }


def run_experiment(args: argparse.Namespace) -> list[dict]:
    rows = []
    for prediction_mode in args.prediction_modes:
        for rep in range(args.repetitions):
            seed = args.seed + 100000 * rep + len(prediction_mode)
            horizon = generate_horizon_instance(
                instance_id=f"T{args.T}_M{args.M}_N{args.N}_K{args.K}_{args.topology}_{args.demand_profile}_{prediction_mode}_s{seed}",
                T=args.T,
                M=args.M,
                N=args.N,
                K=args.K,
                topology=args.topology,
                demand_profile=args.demand_profile,
                prediction_mode=prediction_mode,
                edge_density=args.edge_density,
                seed=seed,
            )
            for policy in ["throughput_greedy", "urgency_primal_dual", "prediction_aware"]:
                row = run_policy(horizon, policy, kappa=args.kappa)
                row["prediction_mode"] = prediction_mode
                rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--N", type=int, default=37)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--topology", default="sparse")
    parser.add_argument("--demand-profile", default="hotspot")
    parser.add_argument("--prediction-modes", nargs="+", default=["accurate", "noisy", "adversarial"])
    parser.add_argument("--edge-density", type=float, default=0.35)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=202607)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--output", default="results/phase_4_cross_time_online_matching.csv")
    args = parser.parse_args(list(argv) if argv is not None else None)
    rows = run_experiment(args)
    write_csv(rows, Path(args.output))
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
