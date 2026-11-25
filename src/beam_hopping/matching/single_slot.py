#!/usr/bin/env python3
"""Single-slot beam hopping matching experiments.

This module implements the Phase 1 experiment:
maximize single-slot throughput on a bipartite beam-cell graph.

Notation follows the source beam hopping formulation:
- M candidate beams, indexed by b = 1,...,M.
- N ground cells, indexed by c_i, i = 1,...,N.
- K active beam-cell pairs in one slot.
- x_t^{b,i} is relaxed to [0,1] for the Total Unimodularity experiment,
  or constrained to {0,1} for the Binary Integer Linear Programming benchmark.
- Edge weight w_t^{b,i}=T_slot * Rbar_t^{b,i}; demand is not an edge weight.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cvxpy as cp
except Exception as exc:  # pragma: no cover
    cp = None
    CVXPY_IMPORT_ERROR = exc
else:
    CVXPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class Instance:
    instance_id: str
    M: int
    N: int
    K: int
    T_slot: float
    B_tot: float
    P_tot: float
    sigma2: np.ndarray
    gain: np.ndarray
    demand_rate: np.ndarray
    feasible_edges: np.ndarray

    @property
    def B0(self) -> float:
        return self.B_tot / self.K

    @property
    def P0(self) -> float:
        return self.P_tot / self.K

    @property
    def rate(self) -> np.ndarray:
        snr = self.P0 * self.gain / self.sigma2.reshape(1, self.N)
        return self.B0 * np.log2(1.0 + snr)

    @property
    def weight(self) -> np.ndarray:
        return self.T_slot * self.rate * self.feasible_edges


def generate_instance(
    *,
    instance_id: str,
    M: int,
    N: int,
    K: int,
    topology: str,
    edge_density: float,
    demand_profile: str,
    seed: int,
    T_slot: float = 0.002,
    B_tot: float = 500e6,
    P_tot: float = 10 ** (33 / 10),
) -> Instance:
    """Generate one reproducible single-slot instance."""
    if not (1 <= K <= min(M, N)):
        raise ValueError("K must satisfy 1 <= K <= min(M, N).")
    rng = np.random.default_rng(seed)
    gain = rng.lognormal(mean=-3.0, sigma=0.7, size=(M, N))
    sigma2 = rng.uniform(1e-13, 3e-13, size=N)

    if demand_profile == "uniform":
        demand_rate = rng.uniform(100e6, 1000e6, size=N)
    elif demand_profile == "hotspot":
        demand_rate = rng.uniform(100e6, 300e6, size=N)
        hot_count = max(1, N // 5)
        hot_idx = rng.choice(N, size=hot_count, replace=False)
        demand_rate[hot_idx] = rng.uniform(1000e6, 3000e6, size=hot_count)
    elif demand_profile == "poisson":
        # This is a single-slot demand-rate sample, not a multi-slot arrival sequence.
        demand_rate = rng.poisson(lam=800, size=N).astype(float) * 1e6
    elif demand_profile == "heavy_tailed":
        # This is a single-slot demand-rate sample, not a multi-slot arrival sequence.
        demand_rate = (rng.pareto(a=2.5, size=N) + 1.0) * 250e6
    else:
        raise ValueError(f"Unknown demand_profile: {demand_profile}")

    feasible_edges = build_topology(M, N, topology, edge_density, rng)
    ensure_feasible_cardinality(feasible_edges, K)

    return Instance(
        instance_id=instance_id,
        M=M,
        N=N,
        K=K,
        T_slot=T_slot,
        B_tot=B_tot,
        P_tot=P_tot,
        sigma2=sigma2,
        gain=gain,
        demand_rate=demand_rate,
        feasible_edges=feasible_edges,
    )


def build_topology(
    M: int,
    N: int,
    topology: str,
    edge_density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a binary M by N feasible-edge matrix."""
    if topology == "complete":
        return np.ones((M, N), dtype=bool)
    if topology == "sparse":
        density = min(max(edge_density, 0.0), 1.0)
        mat = rng.random((M, N)) <= density
        for b in range(M):
            if not mat[b].any():
                mat[b, rng.integers(0, N)] = True
        for i in range(N):
            if not mat[:, i].any():
                mat[rng.integers(0, M), i] = True
        return mat
    if topology == "geometric":
        beams = rng.random((M, 2))
        cells = rng.random((N, 2))
        dist = np.linalg.norm(beams[:, None, :] - cells[None, :, :], axis=2)
        radius = math.sqrt(max(edge_density, 0.05) / math.pi)
        mat = dist <= radius
        for b in range(M):
            if not mat[b].any():
                mat[b, int(np.argmin(dist[b]))] = True
        for i in range(N):
            if not mat[:, i].any():
                mat[int(np.argmin(dist[:, i])), i] = True
        return mat
    raise ValueError(f"Unknown topology: {topology}")


def ensure_feasible_cardinality(feasible_edges: np.ndarray, K: int) -> None:
    """Basic feasibility guard for selecting K non-conflicting edges."""
    M, N = feasible_edges.shape
    if K > min(M, N):
        raise ValueError("K exceeds min(M,N).")
    if feasible_edges.sum() < K:
        raise ValueError("Not enough feasible edges for K selections.")


def solve_matching(instance: Instance, *, binary: bool, solver: str = "MOSEK") -> dict:
    """Solve single-slot matching as LP relaxation or BILP."""
    if cp is None:
        raise RuntimeError(f"cvxpy import failed: {CVXPY_IMPORT_ERROR}")
    if solver not in cp.installed_solvers():
        raise RuntimeError(
            f"Required solver {solver} is not installed. Installed solvers: {cp.installed_solvers()}"
        )

    M, N, K = instance.M, instance.N, instance.K
    x = cp.Variable((M, N), boolean=binary)
    constraints = [
        cp.sum(x, axis=1) <= 1,
        cp.sum(x, axis=0) <= 1,
        cp.sum(x) == K,
    ]
    if np.any(~instance.feasible_edges):
        constraints.append(x[~instance.feasible_edges] == 0)
    if not binary:
        constraints.extend([x >= 0, x <= 1])

    objective = cp.Maximize(cp.sum(cp.multiply(instance.weight, x)))
    problem = cp.Problem(objective, constraints)

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    problem.solve(solver=solver, verbose=False)
    wall_time = time.perf_counter() - wall_start
    cpu_time = time.process_time() - cpu_start

    if x.value is None:
        x_value = np.full((M, N), np.nan)
    else:
        x_value = np.asarray(x.value)

    delivered_by_cell = (instance.T_slot * instance.rate * x_value).sum(axis=0)
    demand_by_cell = instance.T_slot * instance.demand_rate
    integrality_violation = float(np.nanmax(np.minimum(np.abs(x_value), np.abs(1.0 - x_value))))
    selected = [
        {"b": int(b + 1), "i": int(i + 1), "x": float(x_value[b, i]), "weight": float(instance.weight[b, i])}
        for b in range(M)
        for i in range(N)
        if np.isfinite(x_value[b, i]) and x_value[b, i] > 1e-6
    ]

    return {
        "instance_id": instance.instance_id,
        "formulation": "BILP" if binary else "LP_relaxation",
        "solver": solver,
        "status": problem.status,
        "objective_value": float(problem.value) if problem.value is not None else float("nan"),
        "integrality_violation": integrality_violation,
        "solve_time_reported": float(problem.solver_stats.solve_time or 0.0),
        "wall_time": wall_time,
        "cpu_time": cpu_time,
        "peak_memory_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "M": M,
        "N": N,
        "K": K,
        "edge_count": int(instance.feasible_edges.sum()),
        "edge_density": float(instance.feasible_edges.mean()),
        "variable_count": int(M * N),
        "constraint_count_approx": int(M + N + 1 + (M * N - instance.feasible_edges.sum()) + (0 if binary else 2 * M * N)),
        "selected_edges": selected,
        "delivered_by_cell": delivered_by_cell.tolist(),
        "single_slot_demand_by_cell": demand_by_cell.tolist(),
    }


def brute_force_optimum(instance: Instance) -> float:
    """Small-instance exact check independent of cvxpy."""
    edges = [(b, i) for b in range(instance.M) for i in range(instance.N) if instance.feasible_edges[b, i]]
    best = -math.inf
    for combo in itertools.combinations(edges, instance.K):
        beams = {b for b, _ in combo}
        cells = {i for _, i in combo}
        if len(beams) != instance.K or len(cells) != instance.K:
            continue
        value = sum(instance.weight[b, i] for b, i in combo)
        best = max(best, value)
    return best


def run_experiment(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    for M, N, K in parse_sizes(args.sizes):
        for topology in args.topologies:
            for demand_profile in args.demand_profiles:
                for rep in range(args.repetitions):
                    seed = args.seed + 100000 * rep + 1000 * M + 10 * N + K
                    instance_id = f"M{M}_N{N}_K{K}_{topology}_{demand_profile}_s{seed}"
                    inst = generate_instance(
                        instance_id=instance_id,
                        M=M,
                        N=N,
                        K=K,
                        topology=topology,
                        edge_density=args.edge_density,
                        demand_profile=demand_profile,
                        seed=seed,
                    )
                    lp = solve_matching(inst, binary=False, solver=args.solver)
                    rows.append(flatten_result(lp))
                    if args.run_bilp:
                        bilp = solve_matching(inst, binary=True, solver=args.solver)
                        rows.append(flatten_result(bilp))
    return rows


def flatten_result(result: dict) -> dict:
    keep = [
        "instance_id",
        "formulation",
        "solver",
        "status",
        "objective_value",
        "integrality_violation",
        "solve_time_reported",
        "wall_time",
        "cpu_time",
        "peak_memory_kb",
        "M",
        "N",
        "K",
        "edge_count",
        "edge_density",
        "variable_count",
        "constraint_count_approx",
    ]
    return {k: result[k] for k in keep}


def parse_sizes(text: str) -> list[tuple[int, int, int]]:
    out = []
    for part in text.split(","):
        M, N, K = (int(x) for x in part.lower().split("x"))
        out.append((M, N, K))
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--sizes", default="4x19x4,10x37x10")
    parser.add_argument("--topologies", nargs="+", default=["complete", "sparse", "geometric"])
    parser.add_argument("--demand-profiles", nargs="+", default=["uniform", "hotspot"])
    parser.add_argument("--edge-density", type=float, default=0.35)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=202607)
    parser.add_argument("--run-bilp", action="store_true")
    parser.add_argument("--output", default="results/phase_1_single_slot_results.csv")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check_env:
        if cp is None:
            print(json.dumps({"cvxpy": False, "error": str(CVXPY_IMPORT_ERROR)}, indent=2))
            return 1
        installed = cp.installed_solvers()
        print(json.dumps({"cvxpy": True, "version": cp.__version__, "installed_solvers": installed}, indent=2))
        return 0 if args.solver in installed else 2

    rows = run_experiment(args)
    write_csv(rows, Path(args.output))
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
