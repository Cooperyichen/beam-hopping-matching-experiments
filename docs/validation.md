# Validation

## Run checks

After installing experiment and test extras, with a working MOSEK license:

```bash
python -m pytest -q
beam-hopping verify-evidence
```

For minimal dependencies, use the scoped tests in [getting started](getting_started.md). An installed MOSEK with a broken license is an error, not a reason to silently substitute another solver.

## Evidence scopes

| Record | What it establishes |
|---|---|
| [PR 2 validation](../src/synthetic_traffic_validation/validation/validation_record.json) | 32 tests, 22 fixed-input port comparisons, wheel smoke checks and archived table consistency |
| [Layout validation](validation/layout_validation_20261125.json) | Imports, command help, examples, documentation links and byte preservation for this reorganization |
| [PDF evidence verifier](../src/synthetic_traffic_validation/verify_evidence.py) | Integrity of 2728 historical policy records, 998 optimizer attempts and selected displayed values |

None proves workload stability assumptions, statistical service guarantees, full paper reproduction or unchanged performance on another machine. No full benchmark rerun is part of the layout change.

Historical validation commands refer to the directory structure at that time. Current commands are above. Historical source comparisons remain available through `python -m synthetic_traffic_validation.verify_port --original-root /path/to/original/synthetic_traffic_validation`.

The [optional workflow template](../.github/templates/strategy-tests.example.yml) is not active. A maintainer may review and copy it into `.github/workflows/` with appropriate permissions. No GitHub Actions result is claimed.
