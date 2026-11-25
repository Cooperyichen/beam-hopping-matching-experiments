# Getting Started

Use Python 3.10 or newer. Run commands from the repository root.

```bash
git clone https://github.com/Cooperyichen/beam-hopping-matching-experiments.git
cd beam-hopping-matching-experiments
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
beam-hopping doctor
beam-hopping strategies
python examples/matching_quickstart.py
python examples/deficit_quickstart.py
```

The examples print results and do not create files. The matching example returns 198 bits, rather than the 100 bits selected by a conflicting greedy choice. The deficit example illustrates feedback accounting, not a stability experiment.

## Install only the features you need

| Feature | Install |
|---|---|
| Exact assignment and deficit controller | Base installation above |
| LP, binary and repair optimization | `python -m pip install -e '.[optimization]'` |
| Reinforcement learning prototype | `python -m pip install -e '.[learning]'` |
| Full historical forecasting pipeline | `python -m pip install -e '.[experiments,test]'` |
| Stored figure generator | `python -m pip install -e '.[plots]'` |

MOSEK requires a valid license. `beam-hopping doctor` reports availability, not license validity. Use `beam-hopping doctor --check-mosek` for an explicit small solver test. No automatic solver fallback is provided. Select a PyTorch build appropriate to your CPU or GPU.

## Discover commands

```bash
beam-hopping --help
beam-hopping single-slot --help
beam-hopping forecast --help
beam-hopping strategies --json
```

`python -m beam_hopping` is equivalent to `beam-hopping`. For use from another project, install this repository into that project's environment and import the modules in the [catalogue](strategies.md). Do not copy generic files such as `forecast.py` into an unrelated source directory.

## Tests

```bash
python -m pytest -q tests/test_repository_layout.py tests/test_cli.py \
  src/synthetic_traffic_validation/tests/test_deficit_matching.py \
  src/synthetic_traffic_validation/tests/test_evidence.py
beam-hopping verify-evidence
```

Optional MOSEK comparisons require a valid license when the solver is installed. Full tests and experiment prerequisites are in [validation](validation.md) and [experiments](experiments.md).
