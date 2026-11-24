# Forecasting and Deficit Scheduling

This addition has two distinct uses. The historical pipeline implements the policies shown in *Pareto_Forecasting_Experimental_Results.pdf*. The new controller implements fixed resource deficit weighted matching from the archived conditional theorem. Historical results do not evaluate the new controller.

Start with [the PDF strategy map](PDF_STRATEGY_MAP.md), [the deficit integration guide](DEFICIT_GUIDE.md), and [source attribution and changes](SOURCE_PROVENANCE.md). The original collaborator scripts are credited to **Yekta Demirci**, as identified by the project owner. Public related versions are pinned in the provenance guide; they differ from the supplied local snapshots.

## Install and import

From `beam_hopping_experiment_artifacts/code`, use a Python 3.10 or newer environment:

```bash
python -m pip install -e '.[test]'
```

The reusable deficit controller requires NumPy and SciPy. It does not require PyTorch, CVXPY, MOSEK, R, or a GPU. Exact combinatorial matching uses SciPy. For the full historical pipeline:

```bash
python -m pip install -e '.[experiments,test]'
```

MOSEK additionally requires the recipient's valid license. Complete and repair optimizers call MOSEK explicitly and do not select another solver automatically. PyTorch CUDA is optional for traffic generation; CPU execution uses the same source integration model. Select a PyTorch build appropriate to the recipient's hardware.

Run fast controller and archive checks:

```bash
python -m pytest -q synthetic_traffic_validation/tests/test_deficit_matching.py synthetic_traffic_validation/tests/test_deficit_demo.py synthetic_traffic_validation/tests/test_evidence.py
python -m synthetic_traffic_validation.verify_evidence
python -m synthetic_traffic_validation.deficit_demo --cycles 8 --output /tmp/deficit_example.json
```

Choose a fresh example output filename. The optional MOSEK objective comparison is skipped only if CVXPY or the installed solver is absent; an installed solver with a broken license is reported as an error. Full integration tests require experiment dependencies and a working MOSEK license:

```bash
python -m pytest -q synthetic_traffic_validation/tests synthetic_traffic_validation/revision_code/test_revision.py
```

## Source layout

| File | Role |
|---|---|
| `traffic.py` | Stationary continuous Pareto ON/OFF integration, CPU or GPU |
| `forecast.py` | Pure Python Whittle ARFIMA estimation, finite history forecasts, residual reserve |
| `allocation.py` | Historical exact assignment, normalized urgency, nominal and chance allocation, repair |
| `revision_code/revision.py` | The PDF's load, horizon, hybrid, and information ablation protocol |
| `revision_code/summarize.py` | Historical summary and figure generation |
| `deficit_matching.py` | New independent controller and exact cardinality matching |
| `deficit_demo.py` | Minimal example and adapter for consecutive existing traffic cycles |
| `verify_evidence.py` | Read-only check of archived CSV evidence against the PDF |
| `verify_port.py` | Optional finite-input numerical comparison with the original experiment source |
| `reference_results/revision_20261026/` | Original result tables, protocol, statuses and fingerprints |
| `source_provenance.json` | Original and ported source fingerprints plus collaborator source inventory |
| `source_port_changes.patch` | Exact changes from the existing experiment implementation |

The retained `operational.py` describes an older sequential reuse experiment. It is not the source of the revised PDF's static results and does not implement the new deficit controller. Legacy orchestration that depends on machine process identifiers or CPU affinity is not included.

## Data and units

The generator writes `traffic.npz` with `traffic_bits` of shape `(N, physical_slots)`, in bits per slot. The historical validation fixture fixes `N=8`, `K=3`, a slot duration of 0.002 seconds, and total bandwidth of 500 MHz. It converts traffic to Mbit inside the forecasting and allocation experiment modules. The reusable deficit API always uses bits and a separate score in squared bits. Convert Mbit inputs by multiplying by $10^6$ before using that API.

The PDF uses synthetic spectral efficiency with provided means and standard deviations. Its 1000 W power entry is metadata and is not a physical power optimization model. It has no geometry based interference, measured channel predictor, arrival queue or traffic expiration model.

## Reproduction scope

Archived result tables can be inspected without regenerating data. They contain 2728 policy records and 998 optimizer attempts. They are historical evidence, not a new run, and their original solver times are not portable performance promises.

For a new pipeline smoke run, from the code directory and with a fresh output directory:

```bash
python -m synthetic_traffic_validation.cli all --output /tmp/bh_pipeline_smoke --smoke --steps 12000 --cycles 2 --draws 32 --operational-cycles 4
```

That command exercises the earlier pipeline, including its separately labeled operational prototype. To reproduce the PDF revision protocol, generate the six original scenarios and calibrate the earlier policies first, then use:

```bash
python -m synthetic_traffic_validation.cli generate --output /path/to/new_dataset --steps 1000000
python -m synthetic_traffic_validation.cli fit-forecast --output /path/to/new_dataset
python -m synthetic_traffic_validation.cli run-matching --output /path/to/new_dataset --cycles 24 --draws 512
export BEAM_HOPPING_DATA_ROOT=/path/to/new_dataset
export BEAM_HOPPING_REVISION_ROOT=/path/to/new_revision
python -m synthetic_traffic_validation.revision_code.revision prepare
python -m synthetic_traffic_validation.revision_code.revision pilot
python -m synthetic_traffic_validation.revision_code.revision formal
python -m synthetic_traffic_validation.revision_code.verify_results
python -m synthetic_traffic_validation.revision_code.summarize
```

These full commands are expensive and were not rerun for this PR. Existing data is not included in Git; use an original dataset or generate a new one. New fitting and calibration can change policy parameters and optimizer outcomes. Reproducing historical numbers also requires the original arrays and calibration artifacts. This PR does not select parameters from new test outcomes.

The original historical result directory remains an immutable evidence archive. Do not point a writing experiment command at `reference_results/`. Its read-only verification entry is `verify_evidence`; the original post-run verifier additionally requires full forecast artifacts and hashes for the code that actually produced that run.

To compare the package against an existing copy of the original experiment implementation, run `python -m synthetic_traffic_validation.verify_port --original-root /path/to/original/synthetic_traffic_validation`. This requires experiment dependencies and compares fixed seeded traffic, forecasts and matching outputs. It does not compare the Python estimator against the collaborator's original R estimator or establish equivalence for every possible input.
