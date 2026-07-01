#!/usr/bin/env python3
"""Paper-aligned DRL and CMADRL prototypes for beam hopping.

This module implements the Phase 3 reproduction layer for two selected
reinforcement learning papers:

* Deng et al., "Beam Hopping Scheduling Based on Deep Reinforcement
  Learning": a single-agent Deep Q-Network selects K illuminated cells.
* Lin et al., "Dynamic Beam Pattern and Bandwidth Allocation Based on
  Multi-Agent Deep Reinforcement Learning for Beam Hopping Satellite
  Systems": K cooperative illumination agents share the same global state
  and reward.

The bandwidth allocation group in Lin et al. is intentionally frozen here.
The current project compares illumination-pattern algorithms under fixed
B_0, P_0, g_t^{b,i}, and sigma_i^2, so this file reproduces the scheduling
logic while keeping the resource variables controlled.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RLConfig:
    """Simulation configuration using the selected papers' main scale."""

    N: int = 19
    K: int = 4
    M: int = 4
    T_ttl: int = 5
    slot_duration_s: float = 0.002
    bandwidth_hz: float = 500e6
    power_total_dbw: float = 33.0
    noise_density_dbm_hz: float = -174.0
    beta: float = 0.5
    train_lambda_min_mbps: float = 50.0
    train_lambda_max_mbps: float = 150.0
    test_lambda_max_mbps: float = 300.0
    link_rate_min_mbps: float = 180.0
    link_rate_max_mbps: float = 900.0
    seed: int = 7


class BeamHoppingRLEnv:
    """Queue-based beam hopping environment.

    The state is the queue matrix
    ``mathbf{D}_t in R^{N x T_ttl}``, flattened and normalized. Entry
    ``phi[n, l]`` stores traffic volume that has waited ``l`` slots.
    """

    def __init__(self, config: RLConfig, *, seed: int, lambda_max_mbps: float | None = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.lambda_max_mbps = lambda_max_mbps or config.train_lambda_max_mbps
        self.lambda_mbps = self.rng.uniform(config.train_lambda_min_mbps, self.lambda_max_mbps, size=config.N)
        self.rate_mbps = self.rng.uniform(config.link_rate_min_mbps, config.link_rate_max_mbps, size=(config.M, config.N))
        self.queue = np.zeros((config.N, config.T_ttl), dtype=np.float64)
        self.t = 0
        self.last_delivered = np.zeros(config.N, dtype=np.float64)
        self.last_arrival = np.zeros(config.N, dtype=np.float64)

    @property
    def state_dim(self) -> int:
        return self.config.N * self.config.T_ttl

    def reset(self) -> np.ndarray:
        self.lambda_mbps = self.rng.uniform(self.config.train_lambda_min_mbps, self.lambda_max_mbps, size=self.config.N)
        self.queue.fill(0.0)
        self.t = 0
        self.last_delivered.fill(0.0)
        self.last_arrival.fill(0.0)
        self._arrive()
        return self.state()

    def state(self) -> np.ndarray:
        scale = max(float(np.max(self.lambda_mbps) * self.config.slot_duration_s * self.config.T_ttl), 1e-9)
        return (self.queue / scale).astype(np.float32).reshape(-1)

    def queue_volume(self) -> np.ndarray:
        return self.queue.sum(axis=1)

    def average_delay(self) -> np.ndarray:
        weights = np.arange(1, self.config.T_ttl + 1, dtype=np.float64)
        volume = self.queue_volume()
        delay = np.divide(
            self.queue @ weights,
            volume,
            out=np.zeros_like(volume),
            where=volume > 1e-12,
        )
        return delay

    def _arrive(self) -> None:
        arrival_mbit = self.rng.poisson(self.lambda_mbps * self.config.slot_duration_s)
        self.queue[:, 0] += arrival_mbit
        self.last_arrival = arrival_mbit.astype(np.float64)

    def _age_queues(self) -> None:
        aged = np.zeros_like(self.queue)
        aged[:, 1:] = self.queue[:, :-1]
        self.queue = aged

    def _beam_assignment_capacity(self, selected_cells: Iterable[int]) -> dict[int, float]:
        selected = list(dict.fromkeys(int(i) for i in selected_cells))
        capacities: dict[int, float] = {}
        used_beams: set[int] = set()
        for i in sorted(selected, key=lambda cell: self.queue_volume()[cell], reverse=True):
            choices = [(self.rate_mbps[b, i], b) for b in range(self.config.M) if b not in used_beams]
            if not choices:
                break
            rate, b = max(choices)
            used_beams.add(b)
            capacities[i] = float(rate * self.config.slot_duration_s)
            if len(capacities) == self.config.K:
                break
        return capacities

    def step(self, selected_cells: Iterable[int]) -> tuple[np.ndarray, float, dict]:
        unique_cells = list(dict.fromkeys(int(i) for i in selected_cells if 0 <= int(i) < self.config.N))
        if len(unique_cells) < self.config.K:
            for i in np.argsort(-self.queue_volume()):
                if int(i) not in unique_cells:
                    unique_cells.append(int(i))
                if len(unique_cells) == self.config.K:
                    break
        unique_cells = unique_cells[: self.config.K]

        before_queue = self.queue.copy()
        capacities = self._beam_assignment_capacity(unique_cells)
        delivered = np.zeros(self.config.N, dtype=np.float64)
        for i, capacity in capacities.items():
            remaining = capacity
            for age in range(self.config.T_ttl - 1, -1, -1):
                served = min(self.queue[i, age], remaining)
                self.queue[i, age] -= served
                delivered[i] += served
                remaining -= served
                if remaining <= 1e-12:
                    break

        throughput = float(np.sum(delivered))
        demand = np.maximum(before_queue.sum(axis=1), 1e-9)
        satisfaction = np.minimum(delivered, demand) / demand
        delay = self.average_delay()
        delay_gap = float(np.max(delay) - np.min(delay))
        throughput_scale = max(float(self.config.K * np.max(self.rate_mbps) * self.config.slot_duration_s), 1e-9)
        reward = self.config.beta * throughput / throughput_scale - (1.0 - self.config.beta) * delay_gap / self.config.T_ttl

        self.last_delivered = delivered
        self._age_queues()
        self._arrive()
        self.t += 1

        info = {
            "selected_cells": tuple(unique_cells),
            "throughput_mbit": throughput,
            "mean_satisfaction": float(np.mean(satisfaction)),
            "min_satisfaction": float(np.min(satisfaction)),
            "delay_gap_slots": delay_gap,
        }
        return self.state(), float(reward), info


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.items: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, item: tuple[np.ndarray, int, float, np.ndarray]) -> None:
        self.items.append(item)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.rng.sample(list(self.items), batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.items)


def optimize_dqn(
    q_net: QNetwork,
    target_net: QNetwork,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    gamma: float,
) -> None:
    if len(buffer) < batch_size:
        return
    states, actions, rewards, next_states = buffer.sample(batch_size)
    q_values = q_net(states).gather(1, actions.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        next_actions = q_net(next_states).argmax(dim=1)
        next_q = target_net(next_states).gather(1, next_actions.view(-1, 1)).squeeze(1)
        target = rewards + gamma * next_q
    loss = nn.functional.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def all_cell_actions(N: int, K: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(N), K))


def select_epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: random.Random, mask: set[int] | None = None) -> int:
    if mask is None:
        valid = list(range(len(q_values)))
    else:
        valid = [i for i in range(len(q_values)) if i not in mask]
    if rng.random() < epsilon:
        return rng.choice(valid)
    return max(valid, key=lambda idx: float(q_values[idx]))


def train_single_agent_dqn(
    config: RLConfig,
    *,
    episodes: int,
    horizon: int,
    seed: int,
) -> tuple[QNetwork, list[tuple[int, ...]], list[float]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    env = BeamHoppingRLEnv(config, seed=seed)
    actions = all_cell_actions(config.N, config.K)
    q_net = QNetwork(env.state_dim, len(actions))
    target_net = QNetwork(env.state_dim, len(actions))
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=5000, seed=seed)
    rewards: list[float] = []
    epsilon = 1.0
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for step in range(horizon):
            with torch.no_grad():
                q = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            action_idx = select_epsilon_greedy(q, epsilon, rng)
            next_state, reward, _ = env.step(actions[action_idx])
            buffer.push((state, action_idx, reward, next_state))
            optimize_dqn(q_net, target_net, buffer, optimizer, batch_size=64, gamma=0.95)
            state = next_state
            total_reward += reward
            if step % 20 == 0:
                target_net.load_state_dict(q_net.state_dict())
        epsilon = max(0.05, epsilon * 0.94)
        rewards.append(total_reward)
    target_net.load_state_dict(q_net.state_dict())
    return q_net, actions, rewards


def train_cooperative_iql(
    config: RLConfig,
    *,
    episodes: int,
    horizon: int,
    seed: int,
) -> tuple[list[QNetwork], list[float]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    env = BeamHoppingRLEnv(config, seed=seed)
    agents = [QNetwork(env.state_dim, config.N) for _ in range(config.K)]
    targets = [QNetwork(env.state_dim, config.N) for _ in range(config.K)]
    for target, agent in zip(targets, agents):
        target.load_state_dict(agent.state_dict())
    optimizers = [torch.optim.Adam(agent.parameters(), lr=1e-3) for agent in agents]
    buffers = [ReplayBuffer(capacity=5000, seed=seed + k) for k in range(config.K)]
    rewards: list[float] = []
    epsilon = 1.0
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for step in range(horizon):
            selected: list[int] = []
            action_per_agent: list[int] = []
            for agent in agents:
                with torch.no_grad():
                    q = agent(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                action = select_epsilon_greedy(q, epsilon, rng, mask=set(selected))
                selected.append(action)
                action_per_agent.append(action)
            next_state, reward, _ = env.step(selected)
            for k, action in enumerate(action_per_agent):
                buffers[k].push((state, action, reward, next_state))
                optimize_dqn(agents[k], targets[k], buffers[k], optimizers[k], batch_size=64, gamma=0.95)
            state = next_state
            total_reward += reward
            if step % 20 == 0:
                for target, agent in zip(targets, agents):
                    target.load_state_dict(agent.state_dict())
        epsilon = max(0.05, epsilon * 0.94)
        rewards.append(total_reward)
    for target, agent in zip(targets, agents):
        target.load_state_dict(agent.state_dict())
    return agents, rewards


def policy_random(env: BeamHoppingRLEnv, rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.sample(range(env.config.N), env.config.K))


def policy_periodic(env: BeamHoppingRLEnv) -> tuple[int, ...]:
    start = (env.t * env.config.K) % env.config.N
    return tuple((start + offset) % env.config.N for offset in range(env.config.K))


def policy_greedy_demand(env: BeamHoppingRLEnv) -> tuple[int, ...]:
    return tuple(int(i) for i in np.argsort(-env.queue_volume())[: env.config.K])


def policy_delay_fairness(env: BeamHoppingRLEnv) -> tuple[int, ...]:
    return tuple(int(i) for i in np.argsort(-env.average_delay())[: env.config.K])


def evaluate_policy(
    config: RLConfig,
    *,
    method: str,
    horizon: int,
    repetitions: int,
    seed: int,
    q_net: QNetwork | None = None,
    actions: list[tuple[int, ...]] | None = None,
    agents: list[QNetwork] | None = None,
) -> dict:
    rng = random.Random(seed)
    metrics = []
    start = time.perf_counter()
    inference_time = 0.0
    for rep in range(repetitions):
        env = BeamHoppingRLEnv(config, seed=seed + 1000 * rep, lambda_max_mbps=config.test_lambda_max_mbps)
        state = env.reset()
        for _ in range(horizon):
            t0 = time.perf_counter()
            if method == "random":
                selected = policy_random(env, rng)
            elif method == "periodic":
                selected = policy_periodic(env)
            elif method == "greedy_demand":
                selected = policy_greedy_demand(env)
            elif method == "delay_fairness":
                selected = policy_delay_fairness(env)
            elif method == "single_agent_dqn":
                assert q_net is not None and actions is not None
                with torch.no_grad():
                    idx = int(q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax(dim=1).item())
                selected = actions[idx]
            elif method == "cmadrl_iql":
                assert agents is not None
                selected_list: list[int] = []
                for agent in agents:
                    with torch.no_grad():
                        q = agent(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                    selected_list.append(select_epsilon_greedy(q, 0.0, rng, mask=set(selected_list)))
                selected = tuple(selected_list)
            else:
                raise ValueError(f"Unknown method: {method}")
            inference_time += time.perf_counter() - t0
            state, _, info = env.step(selected)
            metrics.append(info)
    wall_time = time.perf_counter() - start
    return {
        "method": method,
        "N": config.N,
        "M": config.M,
        "K": config.K,
        "horizon": horizon,
        "repetitions": repetitions,
        "mean_throughput_mbit": float(np.mean([m["throughput_mbit"] for m in metrics])),
        "mean_satisfaction": float(np.mean([m["mean_satisfaction"] for m in metrics])),
        "min_satisfaction": float(np.min([m["min_satisfaction"] for m in metrics])),
        "mean_delay_gap_slots": float(np.mean([m["delay_gap_slots"] for m in metrics])),
        "wall_time_s": wall_time,
        "mean_inference_time_s": inference_time / max(horizon * repetitions, 1),
    }


def run_experiment(args: argparse.Namespace) -> list[dict]:
    config = RLConfig(seed=args.seed, beta=args.beta)
    single_agent, single_actions, single_rewards = train_single_agent_dqn(
        config,
        episodes=args.episodes,
        horizon=args.horizon,
        seed=args.seed,
    )
    cmadrl_agents, cmadrl_rewards = train_cooperative_iql(
        config,
        episodes=args.episodes,
        horizon=args.horizon,
        seed=args.seed + 17,
    )
    rows = [
        evaluate_policy(config, method="random", horizon=args.eval_horizon, repetitions=args.repetitions, seed=args.seed),
        evaluate_policy(config, method="periodic", horizon=args.eval_horizon, repetitions=args.repetitions, seed=args.seed),
        evaluate_policy(config, method="greedy_demand", horizon=args.eval_horizon, repetitions=args.repetitions, seed=args.seed),
        evaluate_policy(config, method="delay_fairness", horizon=args.eval_horizon, repetitions=args.repetitions, seed=args.seed),
        evaluate_policy(
            config,
            method="single_agent_dqn",
            horizon=args.eval_horizon,
            repetitions=args.repetitions,
            seed=args.seed,
            q_net=single_agent,
            actions=single_actions,
        ),
        evaluate_policy(
            config,
            method="cmadrl_iql",
            horizon=args.eval_horizon,
            repetitions=args.repetitions,
            seed=args.seed,
            agents=cmadrl_agents,
        ),
    ]
    for row in rows:
        row["training_episodes"] = args.episodes if row["method"] in {"single_agent_dqn", "cmadrl_iql"} else 0
        row["training_horizon"] = args.horizon if row["method"] in {"single_agent_dqn", "cmadrl_iql"} else 0
        row["final_training_reward"] = (
            float(single_rewards[-1])
            if row["method"] == "single_agent_dqn"
            else float(cmadrl_rewards[-1])
            if row["method"] == "cmadrl_iql"
            else ""
        )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--eval-horizon", type=int, default=80)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/phase_3_drl_marl_reproduction.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_experiment(args)
    write_rows(args.output, rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
