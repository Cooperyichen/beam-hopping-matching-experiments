# Beam Hopping Matching Experiments

This code implements the numerical experiments for the beam hopping matching project.

## Phase 1

Run environment check:

```bash
python3 single_slot_matching.py --check-env --solver MOSEK
```

Run a small Phase 1 experiment:

```bash
python3 single_slot_matching.py \
  --solver MOSEK \
  --sizes 4x19x4,10x37x10 \
  --topologies complete sparse geometric \
  --demand-profiles uniform hotspot \
  --repetitions 3 \
  --run-bilp \
  --output results/phase_1_single_slot_results.csv
```

Run unit tests:

```bash
python3 -m pytest -q
```

## Phase 3

Run single-slot comparison baselines:

```bash
python3 phase3_benchmarks.py \
  --solver MOSEK \
  --sizes 4x19x4,10x37x10 \
  --topologies complete sparse geometric \
  --demand-profiles uniform hotspot \
  --repetitions 3 \
  --run-bilp \
  --output results/phase_3_single_slot_comparison.csv
```

Run the paper-aligned Deep Reinforcement Learning and Cooperative Multi-Agent Deep Reinforcement Learning reproduction prototype:

```bash
python3 drl_marl_reproduction.py \
  --episodes 30 \
  --horizon 60 \
  --eval-horizon 80 \
  --repetitions 5 \
  --output results/phase_3_drl_marl_reproduction.csv
```

The reproduction prototype freezes bandwidth allocation and focuses on the illumination-pattern decision layer, so it remains comparable with the fixed $B_0$, $P_0$, $g_t^{b,i}$, and $\sigma_i^2$ setting used by the matching experiments.

## Phase 4

Run the cross-time online matching prototype:

```bash
python3 cross_time_online_matching.py \
  --T 20 \
  --M 10 \
  --N 37 \
  --K 10 \
  --topology sparse \
  --demand-profile hotspot \
  --prediction-modes accurate noisy adversarial \
  --repetitions 5 \
  --output results/phase_4_cross_time_online_matching.csv
```

## Modeling rule

The edge weight is the theoretical deliverable traffic volume:

$$
w_t^{b,i}=T_{\mathrm{slot}}\bar R_t^{b,i}.
$$

Demand is a cross-time lower bound, not an edge weight.
