# History and Migration

The current layout is organized by function rather than research stage. Source formulas and archived result bytes were not changed by the reorganization.

| Earlier location | Current location |
|---|---|
| `code/single_slot_matching.py` | `src/beam_hopping/matching/single_slot.py` |
| `code/phase3_benchmarks.py` | `src/beam_hopping/matching/heuristics.py` |
| `code/cross_time_online_matching.py` | `src/beam_hopping/scheduling/online.py` |
| `code/drl_marl_reproduction.py` | `src/beam_hopping/learning/reinforcement.py` |
| `code/presentation_plots.py` | `src/beam_hopping/visualization/presentation.py` |
| `code/synthetic_traffic_validation/` | `src/synthetic_traffic_validation/` |
| Detailed package guides | This `docs/` directory |
| Stage result tables and figures | Functional folders under `results/` |

Earlier `code/` paths were prefixed by `beam_hopping_experiment_artifacts/`. Their five script names retain thin compatibility launchers. Existing `synthetic_traffic_validation` imports remain valid. Prefer installation at the repository root and the unified command. Do not use the earlier nested package installation command.

Verbose Chinese stage reports are no longer current documentation. Their original text remains in [the repository before this layout change](https://github.com/Cooperyichen/beam-hopping-matching-experiments/tree/26e57bb247b661e1c581b90c654f1869fe4ac453/beam_hopping_experiment_artifacts/reports). They record early matching, literature screening, heuristic comparisons and online prototypes, not validated guarantees of the new deficit controller.

[PR 2](https://github.com/Cooperyichen/beam-hopping-matching-experiments/pull/2) merged the forecasting supplement and deficit controller. Optional [theory records](research_notes/README.md) retain their original status. The shared [journal](PROJECT_JOURNAL.md) continues here; no second journal is introduced.
