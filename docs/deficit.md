# Deficit Controller Reuse

Import `DeficitMatchingController` from `synthetic_traffic_validation.deficit_matching`. A runnable example is [deficit_quickstart.py](../examples/deficit_quickstart.py).

## Inputs and outputs

| Item | Contract |
|---|---|
| `nominal_edge_service_bits` | Finite, nonnegative array `(T, M, N)` in bits; available at planning time |
| `cardinality` | Integer $K$ between 1 and the smaller party size |
| `feasible_edges` | Optional Boolean array of the same shape; omitted means complete graph |
| `plan.selected` | Binary `(T, M, N)` array; exactly $K$ pairs per slot, without repeated beam or cell |
| `plan.nominal_service_bits` | Planned capacity per cell, `(N,)`, in bits |
| `plan.score_bits_squared` | Weighted score, not traffic; units $\mathrm{bit}^2$ |
| `observe(demand_bits, service_capacity_bits)` | Observed `(N,)` arrays after execution, in bits |
| `state_dict()` / `from_state_dict()` | Checkpoint and restore only at a completed cycle boundary |

Here $T$ is slots per cycle, $M$ is beams, and $N$ is cells. Use one controller for each consecutive realized trajectory.

```python
from synthetic_traffic_validation.deficit_matching import DeficitMatchingController

controller = DeficitMatchingController(cells=N)
plan = controller.plan(edge_service_bits, cardinality=K)
# Execute plan.selected in your simulator before providing actual feedback.
receipt = controller.observe(actual_demand_bits, realized_capacity_bits)
checkpoint = controller.state_dict()
```

The application supplies `N`, `K` and the three arrays. It is responsible for their units, information availability and correspondence to the executed plan. Convert Mbit to bits explicitly.

## Policy and limits

Let $Z_i^j$ denote cell $c_i$'s virtual deficit at cycle $j$, $w_{j,t}^{b,i}$ its planning edge service, $D_{i,j}$ observed demand, and $S_{i,j}$ realized allocated capacity. The controller freezes $Z_i^j$ throughout the cycle, uses edge score $Z_i^jw_{j,t}^{b,i}$, then applies

$$Z_i^{j+1}=[Z_i^j+D_{i,j}-S_{i,j}]^+.$$

Provide feedback exactly once per plan. Invalid feedback leaves the plan pending. Do not substitute forecasts, reset debt between cycles, skip physical cycles, or average alternative service realizations into one controller. Capacity feedback is not capped at current demand; this is virtual cross cycle compensation, not physical packet departures. Zero deficits create tied primary scores, without an added throughput objective.

Fixed resources and independent slot feasibility permit exact slot matching. The archived [conditional theorem](research_notes/2026-11-10_deficit_weighted_matching_theorem_original.md) additionally requires valid conditional service estimates, moment bounds and strict service slack. The [optional error extension](research_notes/2026-11-12_deficit_error_tolerance_progress.md) remains background. These assumptions have not been validated for the workload. No per-cycle coverage, deadline, chance constraint or raw throughput guarantee is claimed. No generalized Benders decomposition is included.

For adjacent existing traffic windows, install the `experiments` extra and use `beam-hopping deficit --help`. The toy example needs only base dependencies. Keep each controller's checkpoint if execution spans jobs.
