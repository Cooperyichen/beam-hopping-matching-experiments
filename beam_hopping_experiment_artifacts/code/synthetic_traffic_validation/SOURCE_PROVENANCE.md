# Source Attribution and Modification History

## Original collaborator contribution

The project owner identifies **Yekta Demirci** as the collaborator who developed and supplied the original traffic generation and forecasting scripts. This credit is based on the owner's statement. The public account [YektaDemirci](https://github.com/YektaDemirci) has the matching name, links his [personal site](https://yektademirci.me/), and hosts the relevant satellite forecasting projects.

The public [leoForecast repository](https://github.com/YektaDemirci/leoForecast) was inspected at commit [`3dc4cc24cbb35a91ea89849a277cb0ea8564b562`](https://github.com/YektaDemirci/leoForecast/tree/3dc4cc24cbb35a91ea89849a277cb0ea8564b562) on 2026-11-24. Relevant public files are:

- [`farima_f.py`](https://github.com/YektaDemirci/leoForecast/blob/3dc4cc24cbb35a91ea89849a277cb0ea8564b562/farima_f.py), SHA-256 `0376f3335c23520c6b8c5bb2ee5134f7ead1e47281daf6352f7e0954772f2fad`.
- [`beamHop/gen_onoff_pareto.py`](https://github.com/YektaDemirci/leoForecast/blob/3dc4cc24cbb35a91ea89849a277cb0ea8564b562/beamHop/gen_onoff_pareto.py), SHA-256 `6f4409ffbe7aefebefa27eaa2dfa7de066198df18dae4416f1f6430a666094ef`.
- [`syntheticData_flows/gen_onoff_pareto.py`](https://github.com/YektaDemirci/leoForecast/blob/3dc4cc24cbb35a91ea89849a277cb0ea8564b562/syntheticData_flows/gen_onoff_pareto.py), SHA-256 `963cce5421f6fa227b7039cdd37f028c7358d829a22b704951ae97dc5c26e006`.

These public files differ byte-for-byte from the locally supplied snapshots. They are related public references, not verified exact upstream revisions of the supplied files. The supplied snapshots' exact upstream commits and intermediate edit history remain unknown. No matching public revision was established for `pareto_old.py` or `farima.py` in this audit. This is not a claim that such revisions do not exist.

## Supplied snapshot inventory

The original local reference directory preserves four scripts. Numeric prefixes are local inventory labels:

| Supplied file | SHA-256 | Relation to the experiment implementation |
|---|---|---|
| `1-pareto_old.py` | `fc35877c9925c30633424f12ce520ffec2400533b18f66ac72961e12cb84b624` | Earlier discrete traffic generator; retained as historical reference |
| `2-farima_f.py` | `67ca4c1d66f56f56db0682a1e1be4de03a80a11e83ab72d2fe5721192640e3b7` | Fit-once ARFIMA and theoretical covariance prediction design |
| `3-gen_onoff_pareto.py` | `c886d4ecfb6d8e29e820eb484afc1e7a284d9e10137715530975ef8e4f800399` | Continuous Pareto sources, equilibrium residual initialization and interval binning design |
| `4-farima.py` | `a67a34d89c9f399ece2b24b8ef8a8577a41885c9cdb311e4e5f9b289d39e846f` | Earlier R based forecasting script, not the estimator executed in the PDF experiments |

The raw reference scripts are not redistributed in this PR. Their supplied versions remain in the existing ThinkPad reference directory. Public reference links do not grant a new license, and this PR does not assign or replace Yekta Demirci's rights. Confirm the applicable upstream permissions before redistributing original source outside the project.

## Adaptations already present before this PR

The following describes the existing experiment implementation, not newly invented changes in this PR. Its precise pre-port fingerprints are in `source_provenance.json`. Attribution to the original scripts does not imply their author developed every later adaptation, matching policy, or theorem.

| Component | Original design retained | Existing adaptation used by the PDF pipeline |
|---|---|---|
| Pareto source process | Superposition of independent ON/OFF users, Pareto durations, stationary residual life, exact occupied-time integration | Per-cell users and independent seeds, eight-cell load calibration, baseline and hotspot profiles, 2 ms bins, NumPy output in bits, memory-aware CPU/GPU batching, generation diagnostics |
| ARFIMA model | Fit once on training observations, fractional differencing and finite-history direct prediction using model covariance | Pure Python profile Whittle fitting with weighted spectral bins; SciPy Toeplitz/Cholesky prediction; stationarity parameterization and fixed parameter bounds; no R/rpy2 dependency |
| Forecast target | Direct cumulative future sum rather than conflating it with independent per-step outputs | Cycle targets for 12, 24 and 48 slots; chronological fitting, selection, reserve calibration and evaluation; cell-specific residual quantiles |
| Evaluation | Traffic used as demand input | Added exact matching and MOSEK references, trust scoring, repair/fallback accounting and paired channel draws |

In particular, the Python Whittle estimator is a reimplementation with a different objective from the supplied R likelihood fitting. The PR does not assert equal parameters or prediction errors between those estimators. The earlier discrete generator and the stationary continuous generator also must not be described as identical models.

## Changes made by this PR

1. The existing experiment modules are placed in the `synthetic_traffic_validation` package. Internal imports become relative; legacy scripts are invoked with `python -m`.
2. Revision input/output roots can be supplied through `BEAM_HOPPING_DATA_ROOT` and `BEAM_HOPPING_REVISION_ROOT`, avoiding a dependency on the author's machine directory.
3. CVXPY is imported when an optimizer is called, rather than when exact matching is imported. MOSEK remains explicit for optimization; no solver substitution is introduced.
4. GPU telemetry searches for `nvidia-smi` before using the historical WSL path. The RSS conversion handles macOS units as well as Linux units.
5. A new deficit controller, exact arbitrary-cardinality matcher, checkpoint contract, examples and tests are added separately. Existing historical scoring is not replaced by this policy.
6. Historical result tables are archived unchanged with fingerprints. A new read-only verifier checks their integrity and displayed coverage values without rewriting them.

This PR does not change the traffic sampling law, Whittle objective, predictor filters, reserve rank, historical urgency formula, calibration grids, historical solver budgets or recorded results. Ported behavior is checked against the existing implementation on fixed inputs. `source_port_changes.patch` provides the exact original-to-ported source changes; `source_provenance.json` records original and final hashes.

## Evidence and provenance limits

The PDF's historical evidence is `reference_results/revision_20261026/`, with its original protocol, source hashes, environment, summary, per-window results and optimizer attempts. Those historical source hashes refer to the original experiment files, not the mechanically ported files. They must remain unchanged. The archive manifest separately identifies the exact bytes distributed here.

The archived theorem documents originate from local commit `aba1466` and retain their existing review status. The new tests establish implementation behavior only. No previous experiment is relabeled as a deficit experiment, and no original collaborator code is credited with the new theoretical guarantees.
