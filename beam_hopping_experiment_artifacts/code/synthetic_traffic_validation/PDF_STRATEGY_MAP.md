# PDF Strategy and Implementation Map

Source document: *Pareto_Forecasting_Experimental_Results.pdf*, 21 pages, supplied by the project owner. SHA-256: `ea732d056fec11da94faf7bc5b66809e5b5d3c4e4dd79dbea8bfbc7b4c0953bd`. The PDF itself is not redistributed. The original evaluation is `revision_20261026`; it is separate from the newly implemented deficit controller.

## Strategy entries

| PDF location and displayed strategy | Code identifier and entry | Meaning |
|---|---|---|
| Page 5, Throughput matching | `throughput_matching`; `revision.evaluate()` and `allocation.matching_plan()` | Base link capacity, urgency coefficient zero |
| Page 5, Historical demand matching | `urgency_no_prediction` | Training mean cycle demand as target |
| Page 5, Prediction matching | `prediction_matching` | ARFIMA cumulative forecast as target |
| Page 5, Reserve matching | `reserve_matching` | Forecast plus cell and horizon specific residual reserve |
| Page 5, Static prediction trust | `static_trust_matching` | Forecast plus a trust dependent share of the reserve |
| Page 6, Nominal optimizer | `nominal_horizon`; `allocation.solve_allocation(chance=False)` | Joint horizon illumination and bandwidth with forecast demand constraints |
| Page 6, Chance optimizer | `chance_horizon`; `solve_allocation(chance=True)` | Gaussian service allowance and calibrated demand reserve |
| Page 12, Plan checking and repair | `certified_static_matching`; `revision.hybrid_plan()` | New trust matching plan, margin check, restricted bandwidth repair, local pattern repair, then full solve |

Here `revision` denotes `synthetic_traffic_validation.revision_code.revision`. The PDF's exact protocol is in this module, not the earlier `experiments.compare()` implementation alone. `forecast.fit_arfima()`, `forecast.step_weights()`, `forecast.predict_at()` and `forecast.quantile_reserve()` provide the fitting, direct cumulative prediction and reserve primitives. `revision.prepare()` extends them to horizons of 12, 24 and 48 slots.

## Historical scoring and units

For cell $c_i$, cycle $j$ and slot $t$, let $D^{\mathrm{tar}}_{i,j}$ be the selected target and $S^{<t}_{i,j}$ the nominal service already planned. Define the remaining target

$$
g_{i,j,t}=[D^{\mathrm{tar}}_{i,j}-S^{<t}_{i,j}]^+.
$$

For $N$ cells, base edge service $w_{j,t}^{b,i}$, and urgency coefficient $\lambda$, the historical score is

$$
\widetilde w_{j,t}^{b,i}=w_{j,t}^{b,i}
\left(1+\lambda\frac{g_{i,j,t}}{N^{-1}\sum_{\ell=1}^{N}g_{\ell,j,t}}\right).
$$

The implementation gives zero additional priority when all gaps vanish, using a numerical denominator floor. Its allocation helpers omit the common service scale $T_{\mathrm{slot}}B_{\mathrm{tot}}/10^6$, which is exactly one for the historical fixture (0.002 seconds and 500 MHz). Their numerical service therefore matches Mbit only under this configuration. They are not general physical-unit APIs for arbitrary bandwidth or slot duration. For a different configuration, explicit scale conversion must be supplied consistently to targets and returned service. The new deficit API instead accepts explicitly computed edge service in bits. The common remaining-slot divisor cancels under mean normalization. These scores are heuristic and are not the new frozen deficit weights $Z_i^j w_{j,t}^{b,i}$.

The mean spectral efficiency array has shape `(8, T)`. The model repeats each cell's channel quality across three beam opportunities, so matching is exact but beam identities have equal link qualities in this fixture. The reusable deficit API supports distinct beam qualities and any feasible cardinality up to the smaller party size.

## Protected and fallback execution

The nominal optimizer omits both the channel allowance and the demand reserve. It is not a comparison that isolates channel protection alone.

If the standalone chance optimizer has no accepted incumbent, the evaluator executes Reserve matching and keeps the failed optimizer attempt in the record. An unresolved call is not a proof of infeasibility. The hybrid instead executes its original trust matching plan if full optimization fails. The hybrid's unprotected path is explicitly labeled. No fallback receives a theoretical chance certificate merely by being executed.

## Experimental conditions and limits

The PDF uses 8 cells, 3 active opportunities, 2 ms slots, 500 MHz total bandwidth, per-cell bandwidth fraction at most 0.6 for full optimization, and matching share 1/3. Forecast horizons are 12, 24 and 48 slots. Demand multipliers are 0.75, 1, 1.25, 1.50 and 1.75. Baseline and two-hotspot scenarios use the same total mean load approximately. Pareto shape is 1.6, minimum complete ON/OFF duration 100 ms, and active user rate 1 Mbit/s at base load.

Training, model selection, residual calibration and testing occupy consecutive fractions 60%, 10%, 10% and 20%. The PDF's channel standard deviation is 20% of its known mean with independent slot errors. Demand and channel risk parameters are each 0.025; dependent residual calibration does not establish unconditional joint 95% protection.

The source run uses CVXPY 1.9.2, MOSEK 11.2.2, two solver threads, three seconds of mixed integer search and relative gap tolerance $10^{-4}$. This differs from the earlier allocation function's five second default; `revision.solve_record()` supplies the three second protocol value.

Page 8 uses 30 sampled traffic windows, with 512 common channel draws per window. Long horizons use smaller window samples. Static window sampling is not consecutive deficit replay, and the channel draws do not create independent traffic histories. PDF coverage caps each cell's service at its actual demand; raw capacity does not.

The archived `static_results.csv`, `static_attempts.csv` and `summary.csv` preserve all final executions and attempts. `python -m synthetic_traffic_validation.verify_evidence` checks their integrity, record accounting and the six displayed page 8 coverage values. It does not rerun the experiments or establish the new theorem's assumptions.
