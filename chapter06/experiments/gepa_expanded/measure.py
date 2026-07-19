"""Emit the ce-optimize hard metrics for the current final or baseline state."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .dataset import (
    DATA_PATH,
    LOCK_PATH,
    ORIGINAL_DATA_PATH,
    RESULTS_DIR,
    SPLIT_PATH,
    assert_lock_unchanged,
    assert_original_rows_preserved,
    read_csv,
    source_family,
    validate_pairs,
)


def _get(value: dict[str, Any], path: str, default: Any = 0) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def metrics(result_path: Path) -> dict[str, Any]:
    lock = assert_lock_unchanged()
    rows = read_csv(DATA_PATH)
    validation = validate_pairs(rows)
    preserved = assert_original_rows_preserved(rows)
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(
        (RESULTS_DIR / "baseline_confirmation" / "summary.json").read_text(encoding="utf-8")
    )
    final = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    optimized_accuracy = float(final.get("optimized_test_accuracy_pct", baseline["baseline_test_accuracy_pct"]))
    baseline_accuracy = float(baseline["baseline_test_accuracy_pct"])
    optimized_correct = int(final.get("optimized_test_correct", baseline["majority_correct"]))
    source_count = len({row["source_id"] for row in rows})
    output = {
        "optimized_test_accuracy_pct": optimized_accuracy,
        "preserved_original_rows": preserved,
        "dataset_rows": len(rows),
        "complete_pair_fraction": validation["complete_pair_fraction"],
        "locked_test_rows": 2 * len(manifest["splits"]["test"]),
        "baseline_test_accuracy_pct": baseline_accuracy,
        "gepa_test_leakage_count": lock["gepa_test_leakage_count"],
        "provenance_complete_fraction": validation["provenance_complete_fraction"],
        "validation_passed": int(
            preserved == 74
            and len(rows) == 300
            and baseline_accuracy <= 50
            and lock["gepa_test_leakage_count"] == 0
        ),
        "pair_count": validation["pair_count"],
        "train_rows": 2 * len(manifest["splits"]["train"]),
        "validation_rows": 2 * len(manifest["splits"]["validation"]),
        "source_count": source_count,
        "source_family_count": len({source_family(row) for row in rows}),
        "baseline_test_correct": baseline["majority_correct"],
        "optimized_test_correct": optimized_correct,
        "absolute_uplift_pct_points": optimized_accuracy - baseline_accuracy,
        "relative_uplift_pct": (
            100 * (optimized_accuracy - baseline_accuracy) / baseline_accuracy
            if baseline_accuracy
            else 0
        ),
        "paired_mcnemar_p_value": float(final.get("paired_mcnemar_p_value", 1.0)),
        "paired_bootstrap_ci_low_pct_points": float(final.get("paired_bootstrap_ci_low_pct_points", 0.0)),
        "paired_bootstrap_ci_high_pct_points": float(final.get("paired_bootstrap_ci_high_pct_points", 0.0)),
        "baseline_repeat_min_accuracy_pct": baseline["min_accuracy_pct"],
        "baseline_repeat_max_accuracy_pct": baseline["max_accuracy_pct"],
        "optimize_cost_usd": float(final.get("optimize_cost_usd", 0.0)),
        "total_experiment_cost_usd": float(
            json.loads((RESULTS_DIR / "budget_ledger.json").read_text(encoding="utf-8"))["total_cost_usd"]
        ),
        "optimize_time_seconds": float(final.get("optimize_time_seconds", 0.0)),
        "baseline_mean_latency_seconds": sum(
            repeat["mean_latency_seconds"] for repeat in baseline["repeats"]
        )
        / len(baseline["repeats"]),
        "optimized_mean_latency_seconds": float(final.get("optimized_mean_latency_seconds", 0.0)),
        "optimized_p95_latency_seconds": float(final.get("optimized_p95_latency_seconds", 0.0)),
    }
    return output


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(metrics(args.result), sort_keys=True))


if __name__ == "__main__":
    main()
