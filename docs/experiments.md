# Experiment Families

All commands run from the repository root after [installation](getting_started.md). Use a new output path; never direct a writing command at an evidence archive.

## Fixed resource matching

```bash
beam-hopping single-slot --solver MOSEK --sizes 4x19x4,10x37x10 \
  --topologies complete --demand-profiles uniform hotspot --repetitions 30 \
  --run-bilp --output runs/single_slot/results.csv
beam-hopping benchmarks --solver MOSEK --sizes 4x19x4,10x37x10 \
  --topologies complete --demand-profiles uniform hotspot --repetitions 30 \
  --run-bilp --output runs/heuristics/results.csv
```

These are explicit suggested new runs, not claims about archived tables. Sparse and geometric topologies remain supported for historical compatibility; they are not the recommended complete graph comparison. The older archived files include earlier designs. Generated gain and noise are synthetic, not a full antenna and propagation model. Single slot demand affects evaluation or heuristic priority, not a hard covering constraint.

## Pareto traffic and forecasting PDF

The PDF's actual protocol is [`revision_code/revision.py`](../src/synthetic_traffic_validation/revision_code/revision.py), not the earlier `experiments.compare()` alone. It uses 8 cells, 3 active opportunities, 2 ms slots, 500 MHz, horizons of 12/24/48 slots, Pareto shape 1.6, and demand multipliers 0.75/1/1.25/1.50/1.75. The power entry is 1000 W metadata, not an optimized physical channel parameter. Channel standard deviation is 20% of its known mean; future realizations are not planning inputs.

Train, select models, calibrate residuals and test on chronological 60/10/10/20 percent partitions. Source data is not committed. Full generation and optimization are expensive:

```bash
beam-hopping forecast generate --output runs/dataset --steps 1000000
beam-hopping forecast fit-forecast --output runs/dataset
beam-hopping forecast run-matching --output runs/dataset --cycles 24 --draws 512
export BEAM_HOPPING_DATA_ROOT="$PWD/runs/dataset"
export BEAM_HOPPING_REVISION_ROOT="$PWD/runs/revision"
beam-hopping forecast-revision prepare
beam-hopping forecast-revision pilot
beam-hopping forecast-revision formal
```

Finish with `python -m synthetic_traffic_validation.revision_code.verify_results` and `python -m synthetic_traffic_validation.revision_code.summarize`. Reproducing historical numbers requires original arrays and calibration artifacts; refitting may change outcomes. Historical optimizer settings are MOSEK, two threads, three seconds of mixed integer search and relative gap $10^{-4}$. Demand and channel risk parameters are each 0.025; dependent residual calibration does not prove unconditional joint protection.

## Separate prototypes

- `beam-hopping online --help`: earlier synthetic horizon heuristics with accurate, noisy and reversed synthetic predictions. These are stress scenarios, not trained forecasting models or optimal adversaries.
- `beam-hopping learning --help`: Deep Q Network and cooperative independent Q learning with Poisson queue arrivals and fixed bandwidth. Short training is a pipeline test, not paper performance reproduction.
- `beam-hopping deficit --help`: persistent deficit feedback on adjacent cycles. Static sampled windows from the PDF are not a deficit trajectory.
- `beam-hopping plots --help`: regenerates historical plots from stored tables, writing to `runs/figures` by default. It does not reproduce newer plots that were never committed.

PDF coverage caps service at realized demand; raw capacity does not. Report bits, runtime and coverage separately. Do not transfer results or guarantees between these models.
