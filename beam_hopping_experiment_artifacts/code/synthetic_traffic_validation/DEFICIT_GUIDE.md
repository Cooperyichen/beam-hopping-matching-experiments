# Integrating Fixed Resource Deficit Matching

## Policy and scope

This controller implements the policy described in [the original theorem](theory/2026-11-10_deficit_weighted_matching_theorem_original.md). The [error tolerance progress note](theory/2026-11-12_deficit_error_tolerance_progress.md) is optional theoretical context, not a different implemented policy. The package contains byte-identical mirrors of the original archive so these definitions accompany an installed distribution. There is no generalized Benders decomposition and no power or bandwidth optimization in this controller.

For cell $c_i$ and consecutive cycle $j$, let $Z_i^j$ be its nonnegative deficit in bits. Let $w_{j,t}^{b,i}$ be the nominal service capacity in bits for selecting beam $b$ and cell $c_i$ in slot $t$. Choose one matching of cardinality $K$ per slot, maximizing

$$
F_j(\mathbf u)=\sum_{t,b,i}Z_i^j w_{j,t}^{b,i}x_{j,t}^{b,i}.
$$

The feasible set has no additional constraint connecting slots. The same $Z_i^j$ is used for all slots in that cycle. Nominal service is the unweighted sum of selected edge capacity. No forecast target, normalization, added constant or per-slot progress update is applied to the deficits.

After execution, use true demand $D_{i,j}$ and realized allocated capacity $S_{i,j}$ to update

$$
Z_i^{j+1}=[Z_i^j+D_{i,j}-S_{i,j}]^+.
$$

The service capacity is not capped at current cycle demand. This is cumulative virtual accounting that permits later surplus capacity to compensate earlier shortfall. It is not a physical packet queue, a deadline guarantee, or permission to count unavailable packets as actual delivered traffic.

## Minimal use from another project

Install the package from the code directory, then import it without copying generic filenames such as `traffic.py` into the other project's module search path:

```python
import numpy as np
from synthetic_traffic_validation.deficit_matching import DeficitMatchingController

controller = DeficitMatchingController(cells=3, initial_deficit_bits=[2., 5., 1.])
nominal_bits = np.array([
    [[9., 8., 0.], [8., 0., 1.]],
    [[2., 4., 3.], [3., 1., 2.]],
])  # shape (T=2, M=2, N=3); decision-time capacity, in bits
plan = controller.plan(nominal_bits, cardinality=2)

# Execute plan.selected in the external simulator, then supply observed values.
receipt = controller.observe(
    demand_bits=np.array([3., 8., 2.]),
    service_capacity_bits=np.array([8., 7., 0.]),
)
checkpoint = controller.state_dict()
restored = DeficitMatchingController.from_state_dict(checkpoint)
```

The example feedback is illustrative. A real integration must obtain it from the chosen plan's execution. The planner accepts no true upcoming demand or realized channel argument. Information availability remains the caller's responsibility: a numeric capacity array cannot prove that its producer avoided future observations.

## API contract

| Item | Shape or behavior |
|---|---|
| `nominal_edge_service_bits` | `(T, M, N)`, finite and nonnegative |
| `feasible_edges` | Optional Boolean array of the same shape; omission means complete graph |
| `cardinality` | Integer $1\le K\le\min\{M,N\}$; infeasible graphs raise an error |
| `plan.selected` | Integral `(T, M, N)` decisions, exactly $K$ pairs each slot, no repeated beam or cell |
| `plan.nominal_service_bits` | `(N,)`, capacity summed over selected edges |
| `plan.frozen_deficit_bits` | `(N,)`, start-of-cycle values |
| `plan.score_bits_squared` | Scalar weighted score, in $\mathrm{bit}^2$ |
| `observe()` | Exactly once after a successful plan; uses `(N,)` demand and realized capacity |
| `state_dict()` | JSON-compatible checkpoint at a completed cycle boundary |

Pending cycles cannot be overwritten or checkpointed. Invalid feedback leaves state unchanged and the plan pending. Each controller owns one consecutive stream. Each Monte Carlo trajectory must have a distinct controller; do not update a single deficit state using averaged service from several alternative realizations. Save and restore checkpoints if execution is split across jobs. The API labels cycles consecutively but cannot detect physical periods omitted by an external caller.

When all deficits are zero, all primary scores are zero. The exact assignment implementation resolves ties; it does not add raw throughput or demand to the objective. Tied optimal plans need not match the assignments returned by another exact solver.

## Connecting the existing dataset

The adapter reads `traffic_bits` from the original generator and processes adjacent windows. Unlike the PDF's static protocol, it does not skip intervening cycles:

```bash
python -m synthetic_traffic_validation.deficit_demo \
  --traffic-npz /path/to/traffic.npz --origin 800000 \
  --horizon 12 --cycles 20 --seed 20261124 \
  --output /tmp/deficit_replay.json
```

This adapter uses the historical eight-cell channel fixture, three beam opportunities, 500 MHz and 2 ms slots. It converts channel statistics to edge capacity in bits explicitly. The plan is constructed before demand feedback and the random channel realization. It uses one Gaussian channel stream and clips negative aggregate cell capacity at zero, as in the historical metrics. It is an integration example, not a newly calibrated experiment or validation of stability conditions. The core API itself is independent of this eight-cell fixture and does not require forecasts.

## What the conditional theorem requires

The original theorem assumes a valid conditional expected service lower bound, uniformly bounded conditional second moments, strict conditional service slack under a comparison mixture of legal plans, and bounded additive error for this same frozen score. Exact per-slot matching establishes the optimization component in exact arithmetic for the uncoupled fixed resource model. Floating point solver tolerance and score magnitude still matter in implementation.

The extension permits bounded model overestimation $\rho_{\mathrm{model}}$ and score error $C+\rho_{\mathrm{opt}}\sum_iZ_i^j$, provided $\varepsilon-\rho_{\mathrm{model}}-\rho_{\mathrm{opt}}>0$. None of these statistical constants is estimated or certified by the controller. The archived documents have no independent mathematical review or Lean verification.

The output does not establish per-cycle demand satisfaction, a chance constraint, packet latency, raw throughput optimality, or the theorem's applicability to the existing Pareto traces. The bounded-deficit but frequent-violation counterexample is included as a regression test and a runnable demonstration specifically to keep that distinction visible.

## Checks included in this PR

Tests compare small cardinality matchings with enumeration and MOSEK LP objectives, exercise distinct beam qualities and forbidden edges, verify frozen cycle scores and realized-capacity feedback, check state persistence and rejected duplicate updates, test absence of future-traffic influence, and execute the original counterexample. These are implementation checks, not statistical assumption verification or a proof by simulation.
