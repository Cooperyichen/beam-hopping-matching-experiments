# Project Journal

This journal starts with the actual project work on 2026-11-24. Earlier project activity is described in the existing status and phase reports; it is not reconstructed here. Dates and claims in historical artifacts remain attached to those artifacts. This project shares one journal across its code branches and compute nodes.

## 2026-11-24 — Forecasting policies and deficit controller supplement

### Purpose and authorization

Prepare one supplement pull request for collaborators to reuse the policies in *Pareto_Forecasting_Experimental_Results.pdf* and the fixed resource deficit theorem policy, without generalized Benders decomposition. The owner confirmed new controller implementation and requested provenance for the collaborator's traffic and forecasting code. The owner identifies that collaborator as Yekta Demirci.

### Starting state and isolation

GitHub `main` was `eb4e78136646cb8e4d72039aa1627d362b694918`. GitHub had no open pull request; `aba1466` is an existing local theory archive commit, not an unmerged PR. Its three Markdown records are reused with their status unchanged. Development occurs in an independent ThinkPad clone at `/home/coope/beam_hopping_pr_forecast_deficit_20261124`, on `codex/forecast-deficit-strategies`. The existing experiment directory and the Mac working tree's unrelated modifications are preserved.

### Work completed

- Located the PDF's actual evaluation in `synthetic_traffic_validation/revision_code/revision.py` and its allocation primitives, rather than substituting the earlier Phase 4 prototype.
- Ported the existing Pareto, ARFIMA, allocation and experiment modules into a namespaced package under the established `beam_hopping_experiment_artifacts/code/` directory.
- Added a separate deficit controller with frozen cycle weights, exact cardinality matching, observed capacity feedback, and boundary checkpoint support.
- Added policy mapping, model and unit contracts, migration examples, source attribution, fingerprints, historical reference results, and a read-only evidence verifier.
- Added a lightweight PR test workflow and package metadata. Optimizer tests continue to use MOSEK.

### Sources, parameters and evidence

The 21-page supplied PDF has SHA-256 `ea732d056fec11da94faf7bc5b66809e5b5d3c4e4dd79dbea8bfbc7b4c0953bd`. Its existing `revision_20261026` evidence contains 2728 policy records and 998 optimizer attempts. The historical fixture uses 8 cells, 3 active opportunities, 2 ms slots, 500 MHz, horizons 12/24/48, Pareto shape 1.6 and load multipliers 0.75/1/1.25/1.50/1.75. These historical experiments are not rerun or retuned for the PR.

`source_provenance.json`, `SOURCE_PROVENANCE.md` and `source_port_changes.patch` distinguish supplied collaborator snapshots, already-existing experiment adaptations, public related versions, and this PR's portability changes. Public related source was located in `YektaDemirci/leoForecast` at `3dc4cc24cbb35a91ea89849a277cb0ea8564b562`; checked public files differ from the supplied local snapshots. Exact upstream revisions for the supplied snapshots remain unverified.

### Validation and corrections during development

The first remote integration run passed 31 tests, including matching enumeration, MOSEK LP checks, source generator checks, original repair behavior, deficit state transitions and the bounded-deficit counterexample. The test environment was Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0, CVXPY 1.9.2 and PyTorch 2.12.1+cu130. Final validation evidence is appended below after the complete package and provenance checks.

The source allocation fixture omits an explicit service conversion factor. Its Mbit interpretation depends on the historical factor $T_{\mathrm{slot}}B_{\mathrm{tot}}/10^6=1$. The guide was corrected to state that condition instead of suggesting an unimplemented conversion. The new deficit API takes explicitly computed edge service in bits.

The local sandbox intermittently denied SSH transfers; commands were rerun with scoped approval. The node itself remained reachable. An unsupported GitHub CLI user-search command was replaced by the read-only GitHub API. Neither issue required changing project models or experiments.

### Scope, limitations and remaining items

The new controller is an implementation of the stated policy, not a validation of the theorem's conditional slack, moment assumptions or model bias constants. It permits cross cycle capacity compensation, and does not guarantee deadlines, per-cycle demand coverage, chance constraints or raw throughput optimality. There is no Lean verification or independent mathematical review. The old PDF static samples are not reused as evidence for deficit stability.

Full million-slot regeneration and full historical optimizer reruns are outside this supplement's validation run. Exact original datasets and calibration artifacts remain external. Historical timings do not predict collaborators' performance. Public source ownership credit is recorded, but the supplied snapshots' exact public version and redistribution permissions are not inferred from matching filenames.

Final test logs, source parity checks, archive checks, published branch and PR status are to be recorded in the completion entry. No additional research conclusion is claimed by the packaging work.

## 2026-11-24 — Validation completion

The final scoped remote run passed 32 tests. MOSEK LP and conic checks ran successfully; two sparse-index copy warnings did not affect the result. The seeded original-to-port comparison passed 22 checks with identical outputs. Read-only archive verification confirmed 2728 policy records, 998 optimizer attempts, 22 archived file fingerprints and all six PDF page 8 coverage values. Two eight-cycle integration examples completed: the archived deterministic counterexample and consecutive cycles from the existing Pareto dataset, using seed 20261124. Neither is a new theorem validation.

A wheel was built and imported from an isolated directory. Its reference tables and theorem mirrors were verified present, exact matching executed correctly, and the controller and evidence verifier imported without Torch or CVXPY. Evidence lives in `../code/synthetic_traffic_validation/validation/`: `pytest.txt`, `port_parity.json`, `archive_check.json`, `deficit_toy.json`, `deficit_existing_traffic.json`, `wheel_check.json`, and `validation_record.json`. The latter records versions, scope, seeds, commands and file fingerprints.

The source port diff was inspected. No traffic law, forecast estimator formula or historical policy parameter was changed. The new package and theory mirrors are additions. Full experiment regeneration, fresh statistical calibration, workload stability assumptions and CI on GitHub remain unverified at this point. This closes local implementation validation; publication status is recorded in the next entry.

Publication checks identified CRLF line endings in immutable historical CSV evidence and significant unified-diff context spaces in the source comparison patch. Scoped Git attributes preserve those bytes and classify their whitespace correctly; the archive is not reformatted to satisfy a cosmetic check. Other staged code and documents pass the normal whitespace check.

The first GitHub push was rejected specifically for adding an active Actions workflow without the OAuth `workflow` scope. No branch was published by that attempt. Account permissions were not changed. The workflow is retained as an optional template under `code/ci/`, with activation instructions; the PR does not install an active workflow. The locally completed tests and their evidence are unchanged. The unpublished delivery commit was amended to contain this scoped correction.

## 2026-11-24 — Publication and collaborator handoff

The supplement branch `codex/forecast-deficit-strategies` was published with implementation commit `57a00ca6fd99044d34e1af88d361c32559be5a23`. [Pull request 2](https://github.com/Cooperyichen/beam-hopping-matching-experiments/pull/2) targets `main`, is open for review, and is not merged. GitHub reported a clean merge state and no status checks at publication; the absence of checks is not a CI pass. The GitHub connector could not create the PR with its integration permissions, so the existing Mac GitHub CLI login was used successfully without changing permissions.

The published review scope contains 74 files, including the optional workflow template rather than an active workflow. The principal handoff documents are the package `README.md`, `PDF_STRATEGY_MAP.md`, `DEFICIT_GUIDE.md`, and `SOURCE_PROVENANCE.md`. Development and validation occurred on ThinkPad WSL. An isolated Mac delivery clone transferred the commits for GitHub publication; the original Mac working tree remains untouched. This publication record is synchronized back to the ThinkPad development clone.

Completed validation remains the 32 scoped tests, 22 source parity checks, archive verification and isolated wheel smoke check described above. No additional experiments were run during publication. Maintainer review, any future workflow activation, full regeneration, exact upstream revision identification for the supplied collaborator snapshots, and validation of theorem assumptions remain outstanding. The controller can now be reused under its documented input and feedback contracts, not as an unconditional service guarantee.
