# Contributing

1. Start from the current main branch and use a separate topic branch.
2. Install the feature and test dependencies you need; see [setup](docs/getting_started.md).
3. Add behavior tests next to the relevant suite. Keep optional imports out of the base CLI.
4. Run the applicable tests and `beam-hopping verify-evidence`. Disclose skipped or unavailable checks.
5. Update the relevant canonical guide rather than adding another overlapping README.
6. Append the purpose, changes, evidence and limits to `docs/PROJECT_JOURNAL.md`.

For experiments, record data source, configuration, seed, dependency versions, units and solver status. Save fresh outputs under `runs/`. Keep training and inference costs separate. No silent solver fallback or relabeling unresolved cases as infeasible.

Do not modify historical result archives or fingerprint inventories to make a check pass. Credit external code and distinguish inherited behavior from new changes. Describe a prototype as a prototype; implementation tests do not confer theoretical guarantees.
