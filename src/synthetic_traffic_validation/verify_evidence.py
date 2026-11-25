"""Read-only verification of the archived PDF result tables and fingerprints."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


DEFAULT = Path(__file__).resolve().parent / "reference_results/revision_20261026"


def verify(root=DEFAULT):
    root = Path(root)
    manifest = json.loads((root / "archive_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"archived evidence changed: {name}")
    with (root / "static_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (root / "static_attempts.csv").open(newline="") as stream:
        attempts = list(csv.DictReader(stream))
    completed = json.loads((root / "formal_complete.json").read_text())
    assert len(rows) == completed["rows"] == 2728
    assert len(attempts) == completed["solver_calls"] == 998
    keys = ["stage", "scenario", "horizon", "scale", "origin", "condition", "policy"]
    assert len({tuple(r[k] for k in keys) for r in rows}) == len(rows)
    assert sum(int(r["repair_attempts"]) + int(r["full_attempts"]) for r in rows) == len(attempts)
    for row in rows:
        assert int(row["origin"]) == int(row["last_available_observation"]) + 1
        assert int(row["target_end_exclusive"]) == int(row["origin"]) + int(row["horizon"])
        assert abs(float(row["useful_mbit"]) + float(row["unmet_mbit"]) - float(row["demand_mbit"])) < 1e-6
        assert float(row["raw_mbit"]) + 1e-8 >= float(row["useful_mbit"])
    for attempt in attempts:
        if attempt["classification"] == "proved_infeasible":
            assert attempt["status"] == "infeasible"
    # Page 8 uses the B_load baseline, 12 slots, load 1.25. Its display rounds
    # the full precision CSV values and retains unresolved-solver fallbacks.
    with (root / "summary.csv").open(newline="") as stream:
        summary = [r for r in csv.DictReader(stream)
                   if r["stage"] == "B_load" and r["scenario"] == "baseline_a1.6"
                   and int(r["horizon"]) == 12 and float(r["scale"]) == 1.25]
    expected = {"throughput_matching": 75.23, "urgency_no_prediction": 98.82,
                "prediction_matching": 99.24, "reserve_matching": 99.54,
                "nominal_horizon": 96.28, "chance_horizon": 99.90}
    assert len(summary) == len(expected)
    for row in summary:
        assert abs(100 * float(row["pooled_coverage"]) - expected[row["policy"]]) <= .006
    return {"policy_records": len(rows), "optimizer_attempts": len(attempts),
            "pdf_page_8_rows": len(summary), "sha256_files": len(manifest["sha256"]),
            "scope": "archive integrity and table agreement; no new experiment or theorem validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
