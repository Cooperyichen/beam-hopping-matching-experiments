# Strategy Catalogue

For names and import locations, run `beam-hopping strategies` or add `--json`. Source paths below are relative to `src/`.

## Forecast aware matching from the Pareto results PDF

All five variants share [`allocation.matching_plan()`](../src/synthetic_traffic_validation/allocation.py). Their dispatch and targets are in [`revision.evaluate()`](../src/synthetic_traffic_validation/revision_code/revision.py).

| Display name | Code identifier | Demand target |
|---|---|---|
| Throughput matching | `throughput_matching` | No urgency term |
| Historical demand matching | `urgency_no_prediction` | Training mean cumulative demand |
| Prediction matching | `prediction_matching` | Predicted cumulative demand |
| Reserve matching | `reserve_matching` | Prediction plus calibrated reserve |
| Static prediction trust | `static_trust_matching` | Prediction plus trust weighted reserve |

For cell $c_i$, cycle $j$ and slot $t$, define remaining target $g_{i,j,t}=[D^{\mathrm{tar}}_{i,j}-S^{<t}_{i,j}]^+$, where $D^{\mathrm{tar}}_{i,j}$ is the chosen target and $S^{<t}_{i,j}$ is capacity already planned. With $N$ cells, base edge service $w_{j,t}^{b,i}$ and urgency coefficient $\lambda$, the score is

$$
\widetilde w_{j,t}^{b,i}=w_{j,t}^{b,i}\left(1+\lambda\frac{g_{i,j,t}}{N^{-1}\sum_{\ell=1}^{N}g_{\ell,j,t}}\right).
$$

When all gaps vanish, added priority is zero. Throughput matching uses $\lambda=0$. These heuristics do not impose a hard demand constraint.

## Other planners

| Method | Entry | Boundary |
|---|---|---|
| Matching with checking and repair | `revision.hybrid_plan()`; `certified_static_matching` | Checks margins, attempts bandwidth or local pattern repair, then full optimization |
| Nominal horizon optimizer | `allocation.solve_allocation(chance=False)` | Omits demand reserve and channel allowance |
| Chance horizon optimizer | `allocation.solve_allocation(chance=True)` | Includes both protections under the stated model |
| Persistent deficit matching | [`DeficitMatchingController`](../src/synthetic_traffic_validation/deficit_matching.py) | Different score and consecutive feedback; see [contract](deficit.md) |
| Single slot LP and binary benchmark | [`solve_matching()`](../src/beam_hopping/matching/single_slot.py) | Fixed resource bipartite matching |
| Greedy, demand priority, random and genetic baselines | [Heuristics](../src/beam_hopping/matching/heuristics.py) | No exact optimality guarantee |
| Earlier urgency and prediction policies | [`run_policy()`](../src/beam_hopping/scheduling/online.py) | Earlier synthetic online experiment, not the PDF pipeline |
| Deep Q Network and cooperative independent Q learning | [Learning prototype](../src/beam_hopping/learning/reinforcement.py) | Separate queue model; not a complete paper reproduction |

The PDF fixture uses channel arrays `(N, T)` and identical beam opportunities. Its allocation helpers return bandwidth fractions and interpret numerical service as Mbit only for the fixed 2 ms and 500 MHz scale. The deficit API instead takes explicit service in bits with shape `(T, M, N)`; these interfaces are not interchangeable.

An unresolved optimizer call is not proof of infeasibility. Standalone chance optimization falls back to Reserve matching; the hybrid falls back to its initial trust plan. A fallback is not automatically protected. See [experiment boundaries](experiments.md).
