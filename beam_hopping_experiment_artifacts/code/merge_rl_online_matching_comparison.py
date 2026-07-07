#!/usr/bin/env python3
"""Merge trained reinforcement-learning results with online matching baselines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from drl_marl_reproduction import RLConfig, evaluate_policy, write_rows


ONLINE_MATCHING_METHODS = [
    "greedy_throughput",
    "urgency_primal_dual",
    "prediction_aware_accurate",
]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def normalize_rows(rows: list[dict]) -> list[dict]:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return [{key: row.get(key, "") for key in fieldnames} for row in rows]


def build_rows(args: argparse.Namespace) -> list[dict]:
    existing_rows = read_rows(args.rl_input)
    config = RLConfig(
        N=args.N,
        M=args.M,
        K=args.K,
        T_ttl=args.T_ttl,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )
    online_rows = [
        evaluate_policy(
            config,
            method=method,
            horizon=args.eval_horizon,
            repetitions=args.repetitions,
            seed=args.seed,
        )
        for method in ONLINE_MATCHING_METHODS
    ]
    for row in online_rows:
        row.update(
            {
                "training_episodes": 0,
                "training_horizon": 0,
                "hidden_dim": "",
                "replay_capacity": "",
                "batch_size": "",
                "gamma": "",
                "learning_rate": "",
                "epsilon_decay": "",
                "use_expanded_state": False,
                "reward_unmet_reduction_weight": "",
                "reward_unmet_backlog_weight": "",
                "final_training_reward": "",
                "final_training_loss": "",
            }
        )
    return normalize_rows(existing_rows + online_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-input", type=Path, default=Path("results/rl_improvement_500ep_mild.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/rl_online_matching_unified_comparison.csv"))
    parser.add_argument("--eval-horizon", type=int, default=120)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--N", type=int, default=19)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--T-ttl", dest="T_ttl", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_rows(args.output, rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
