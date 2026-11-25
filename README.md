# Beam Hopping Matching Experiments

Reusable matching, forecasting and scheduling code for satellite resource allocation.

## Find the right module

| Task | Implementation | Guide |
|---|---|---|
| Exact single slot matching and linear programming (LP) relaxation | [Matching](src/beam_hopping/matching/) | [Strategies](docs/strategies.md) |
| Greedy and genetic algorithm comparisons | [Heuristics](src/beam_hopping/matching/heuristics.py) | [Experiments](docs/experiments.md) |
| Demand forecasting and forecast aware matching | [Forecasting pipeline](src/synthetic_traffic_validation/) | [Strategies](docs/strategies.md) |
| Matching with service checks and repair | [Hybrid planner](src/synthetic_traffic_validation/revision_code/revision.py) | [Strategies](docs/strategies.md) |
| Persistent virtual deficit scheduling | [Deficit controller](src/synthetic_traffic_validation/deficit_matching.py) | [Reuse contract](docs/deficit.md) |
| Earlier online scheduling and reinforcement learning prototypes | [Scheduling](src/beam_hopping/scheduling/) and [learning](src/beam_hopping/learning/) | [Experiments](docs/experiments.md) |
| Recorded tables and figures | [Results](results/README.md) | [Validation](docs/validation.md) |

## Start here

```bash
python -m pip install -e '.[test]'
beam-hopping --help
beam-hopping strategies
python examples/matching_quickstart.py
```

The minimal installation needs no optimizer license. Optimization experiments use MOSEK explicitly; optional dependencies are installed by feature. See [setup and commands](docs/getting_started.md).

For collaboration, read the [input and feedback contract](docs/deficit.md), [source attribution](docs/provenance.md), and [contribution checklist](CONTRIBUTING.md). Detailed guidance has one home in [docs](docs/README.md); old stage reports are available through the [history index](docs/history.md).

## Scope

Total Unimodularity (TU) supports the fixed resource matching relaxation. Added horizon demand constraints or resource allocation require separate analysis. Historical forecasting comparisons, earlier prototypes, and the new deficit controller are distinct experiments. Implementation tests do not establish the assumptions of a stability theorem or guarantee service in every cycle.
