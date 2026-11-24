# Optional Integration Workflow

`strategy-tests.example.yml` is a template, not an active GitHub Actions workflow.
The current publishing credential does not have permission to create workflows.
No credential or repository permission was changed for this supplement.

A repository maintainer with the appropriate permission can copy this template
to `.github/workflows/strategy-tests.yml` at the repository root. Its paths assume
that location and the existing repository layout. Review the template before
enabling it. No GitHub CI result is claimed for this PR.

The same lightweight checks can be run manually from the repository root:

```bash
python -m pip install -e './beam_hopping_experiment_artifacts/code[test]'
cd beam_hopping_experiment_artifacts/code
python -m pytest -q \
  synthetic_traffic_validation/tests/test_deficit_matching.py \
  synthetic_traffic_validation/tests/test_deficit_demo.py \
  synthetic_traffic_validation/tests/test_evidence.py
```

MOSEK validation in the ThinkPad environment is recorded separately in
`../synthetic_traffic_validation/validation/`. A lightweight installation alone
does not validate optimizer or GPU dependencies.
