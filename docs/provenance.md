# Source Attribution

The project owner identifies **Yekta Demirci** as the original traffic and forecasting script contributor. Related public source is in [YektaDemirci/leoForecast](https://github.com/YektaDemirci/leoForecast/tree/3dc4cc24cbb35a91ea89849a277cb0ea8564b562), inspected at commit `3dc4cc24cbb35a91ea89849a277cb0ea8564b562`.

Relevant public files are [`farima_f.py`](https://github.com/YektaDemirci/leoForecast/blob/3dc4cc24cbb35a91ea89849a277cb0ea8564b562/farima_f.py) and two [`gen_onoff_pareto.py`](https://github.com/YektaDemirci/leoForecast/blob/3dc4cc24cbb35a91ea89849a277cb0ea8564b562/beamHop/gen_onoff_pareto.py) variants. Checked public files differ from supplied local snapshots. Their exact upstream revisions remain unverified; matching filenames do not establish equivalence.

## What came from where

| Layer | Contribution or change |
|---|---|
| Supplied collaborator designs | Continuous Pareto ON/OFF sources, stationary residual initialization, interval integration, and covariance based cumulative forecasting |
| Existing experiment adaptation | Per-cell seeds and calibration, CPU/GPU batching, pure Python profile Whittle fitting and SciPy forecast filters, matching and optimization comparisons |
| PR 2 supplement | Namespaced packaging, configurable paths, lazy optimizer loading, portable telemetry, a separate deficit controller and tests |
| Functional layout revision | Source and result relocation, canonical English documentation, import compatibility shims and a lightweight command dispatcher; no model or scoring change |

The Python Whittle estimator replaces the supplied R likelihood path. Equivalent parameters or errors are not claimed. The earlier discrete and continuous traffic generators are not identical models.

## Audit assets

- [Source inventory and hashes](../src/synthetic_traffic_validation/source_provenance.json): supplied snapshots and checked public versions.
- [Original to package patch](../src/synthetic_traffic_validation/source_port_changes.patch): the exact PR 2 source adaptation, not a patch for the later directory migration.
- [Layout manifest](validation/layout_manifest_20261125.json): old and new paths, with preservation checks.
- [Historical evidence](../src/synthetic_traffic_validation/reference_results/revision_20261026/): unchanged protocol, code fingerprints and tables. Historical hashes refer to the source actually used, not later packaged paths.

Raw original collaborator scripts are not redistributed. No new license or ownership claim is assigned. Confirm applicable permissions before redistributing original source outside the project. New tests establish implementation behavior, not estimator equivalence or theorem assumptions.
