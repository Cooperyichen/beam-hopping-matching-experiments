#!/usr/bin/env python3
"""Phase 3 single-slot benchmark algorithms.

All baselines use the same feasible edge set and the same edge weight
matrix w_t^{b,i}. Demand is recorded for evaluation, not used as a hard
single-slot constraint.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from single_slot_matching import (
    Instance,
    flatten_result,
    generate_instance,
    parse_sizes,
    solve_matching,
)


def selected_matrix(instance: Instance, edges: list[tuple[int, int]]) -> np.ndarray:
    x = np.zeros((instance.M, instance.N), dtype=float)
    for b, i in edges:
        x[b, i] = 1.0
    return x


def evaluate_edges(instance: Instance, edges: list[tuple[int, int]], method: str, elapsed: float) -> dict:
    x = selected_matrix(instance, edges)
    objective = float(np.sum(instance.weight * x))
    delivered_by_cell = (instance.T_slot * instance.rate * x).sum(axis=0)
    single_slot_demand = instance.T_slot * instance.demand_rate
    satisfaction = np.divide(
        delivered_by_cell,
        single_slot_demand,
        out=np.zeros_like(delivered_by_cell),
        where=single_slot_demand > 0,
    )
    return {
        "instance_id": instance.instance_id,
        "method": method,
        "status": "feasible" if len(edges) == instance.K else "infeasible",
        "objective_value": objective,
        "integrality_violation": 0.0,
        "wall_time": elapsed,
        "cpu_time": elapsed,
        "M": instance.M,
        "N": instance.N,
        "K": instance.K,
        "edge_count": int(instance.feasible_edges.sum()),
        "edge_density": float(instance.feasible_edges.mean()),
        "selected_count": len(edges),
        "mean_single_slot_satisfaction": float(np.mean(satisfaction)),
        "min_single_slot_satisfaction": float(np.min(satisfaction)),
    }


def greedy_matching(instance: Instance, *, score: np.ndarray | None = None) -> list[tuple[int, int]]:
    matrix = instance.weight if score is None else score
    order = sorted(
        [(float(matrix[b, i]), b, i) for b in range(instance.M) for i in range(instance.N) if instance.feasible_edges[b, i]],
        reverse=True,
    )
    used_b: set[int] = set()
    used_i: set[int] = set()
    edges: list[tuple[int, int]] = []
    for _, b, i in order:
        if b in used_b or i in used_i:
            continue
        edges.append((b, i))
        used_b.add(b)
        used_i.add(i)
        if len(edges) == instance.K:
            break
    return edges


def demand_priority_greedy(instance: Instance) -> list[tuple[int, int]]:
    demand_scale = instance.demand_rate / max(float(np.mean(instance.demand_rate)), 1.0)
    score = instance.weight * demand_scale.reshape(1, instance.N)
    return greedy_matching(instance, score=score)


def random_matching(instance: Instance, rng: random.Random) -> list[tuple[int, int]]:
    candidates = [(b, i) for b in range(instance.M) for i in range(instance.N) if instance.feasible_edges[b, i]]
    rng.shuffle(candidates)
    used_b: set[int] = set()
    used_i: set[int] = set()
    edges: list[tuple[int, int]] = []
    for b, i in candidates:
        if b in used_b or i in used_i:
            continue
        edges.append((b, i))
        used_b.add(b)
        used_i.add(i)
        if len(edges) == instance.K:
            break
    if len(edges) == instance.K:
        return edges
    return greedy_matching(instance)


def genetic_matching(
    instance: Instance,
    *,
    seed: int,
    population_size: int = 40,
    generations: int = 80,
    mutation_rate: float = 0.15,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    feasible_by_beam = [
        [i for i in range(instance.N) if instance.feasible_edges[b, i]]
        for b in range(instance.M)
    ]

    def make_individual() -> list[tuple[int, int]]:
        beams = list(range(instance.M))
        rng.shuffle(beams)
        used_i: set[int] = set()
        edges: list[tuple[int, int]] = []
        for b in beams:
            cells = feasible_by_beam[b][:]
            rng.shuffle(cells)
            for i in cells:
                if i not in used_i:
                    edges.append((b, i))
                    used_i.add(i)
                    break
            if len(edges) == instance.K:
                break
        if len(edges) < instance.K:
            return greedy_matching(instance)
        return edges

    def fitness(edges: list[tuple[int, int]]) -> float:
        if len(edges) != instance.K:
            return -1.0
        if len({b for b, _ in edges}) != instance.K:
            return -1.0
        if len({i for _, i in edges}) != instance.K:
            return -1.0
        return float(sum(instance.weight[b, i] for b, i in edges))

    def repair(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        cleaned: list[tuple[int, int]] = []
        used_b: set[int] = set()
        used_i: set[int] = set()
        for b, i in edges:
            if b in used_b or i in used_i or not instance.feasible_edges[b, i]:
                continue
            cleaned.append((b, i))
            used_b.add(b)
            used_i.add(i)
            if len(cleaned) == instance.K:
                return cleaned
        for _, b, i in sorted(
            [(float(instance.weight[b, i]), b, i) for b in range(instance.M) for i in range(instance.N) if instance.feasible_edges[b, i]],
            reverse=True,
        ):
            if b in used_b or i in used_i:
                continue
            cleaned.append((b, i))
            used_b.add(b)
            used_i.add(i)
            if len(cleaned) == instance.K:
                return cleaned
        return cleaned

    population = [make_individual() for _ in range(population_size)]
    for _ in range(generations):
        population.sort(key=fitness, reverse=True)
        elites = population[: max(2, population_size // 5)]
        new_population = elites[:]
        while len(new_population) < population_size:
            p1 = rng.choice(elites)
            p2 = rng.choice(population[: max(4, population_size // 2)])
            cut = rng.randint(0, instance.K)
            child = p1[:cut] + p2[cut:]
            if rng.random() < mutation_rate:
                child.append(rng.choice(make_individual()))
            new_population.append(repair(child))
        population = new_population
    population.sort(key=fitness, reverse=True)
    return population[0]


def time_baseline(name: str, fn) -> tuple[list[tuple[int, int]], float]:
    start = time.perf_counter()
    edges = fn()
    return edges, time.perf_counter() - start


def run_phase3(args: argparse.Namespace) -> list[dict]:
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
                    lp = flatten_result(solve_matching(inst, binary=False, solver=args.solver))
                    lp["method"] = "LP_relaxation_TU"
                    rows.append(lp)
                    if args.run_bilp:
                        bilp = flatten_result(solve_matching(inst, binary=True, solver=args.solver))
                        bilp["method"] = "BILP_benchmark"
                        rows.append(bilp)

                    rng = random.Random(seed)
                    for method_name, fn in [
                        ("random_feasible_matching", lambda inst=inst, rng=rng: random_matching(inst, rng)),
                        ("greedy_throughput", lambda inst=inst: greedy_matching(inst)),
                        ("demand_priority_greedy", lambda inst=inst: demand_priority_greedy(inst)),
                        ("genetic_algorithm", lambda inst=inst, seed=seed: genetic_matching(inst, seed=seed)),
                    ]:
                        edges, elapsed = time_baseline(method_name, fn)
                        rows.append(evaluate_edges(inst, edges, method_name, elapsed))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
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
    parser.add_argument("--output", default="results/phase_3_single_slot_comparison.csv")
    args = parser.parse_args(list(argv) if argv is not None else None)
    rows = run_phase3(args)
    write_csv(rows, Path(args.output))
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
