"""Build the Chapter 6 expanded-dataset comparison from durable run artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chapter06.build_optimizer_notebooks import NOTEBOOKS
from chapter06.experiments.gepa_expanded.analysis import (
    mcnemar_exact,
    paired_pair_bootstrap,
)
from chapter06.experiments.gepa_expanded.guardrails import atomic_write_json
from chapter06.optimizer_runtime import DATA_PATH, RESULTS_ROOT, SPLIT_PATH


CHAPTER_DIR = Path(__file__).resolve().parent
GEPA_ROOT = CHAPTER_DIR / "results" / "gepa_expanded"
DISPLAY_NAMES = {
    spec["optimizer"]: spec["title"] for spec in NOTEBOOKS.values()
}
ORDER = [spec["optimizer"] for spec in NOTEBOOKS.values()]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _money(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _cost_status(optimizer: str) -> tuple[str, str]:
    if optimizer in {"copro", "miprov2"}:
        return (
            "partial_lower_bound",
            "unavailable_due_to_detached_dspy_history",
        )
    return "recorded", "recorded"


def _build_run_ledger() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for optimizer in ORDER:
        if optimizer == "gepa":
            continue
        for mode in ("smoke", "full"):
            result_path = RESULTS_ROOT / optimizer / mode / "result.json"
            if not result_path.exists():
                continue
            result = _json(result_path)
            optimization_status, evaluation_status = _cost_status(optimizer)
            evaluation_cost = result.get("evaluation_cost_usd")
            if evaluation_status != "recorded":
                evaluation_cost = None
            runs.append(
                {
                    "optimizer": optimizer,
                    "mode": mode,
                    "status": "completed",
                    "started_at": result.get("started_at"),
                    "finished_at": result.get("finished_at"),
                    "optimization_cost_usd": result.get("optimization_cost_usd"),
                    "optimization_cost_status": optimization_status,
                    "evaluation_cost_usd": evaluation_cost,
                    "evaluation_cost_status": evaluation_status,
                    "optimization_time_seconds": result.get("optimization_seconds"),
                    "result_artifact": str(result_path.relative_to(CHAPTER_DIR.parent)),
                }
            )
    for failure_path in sorted(RESULTS_ROOT.glob("*/*/preflight_failures.json")):
        payload = _json(failure_path)
        for failure in payload.get("failures", []):
            paid = bool(failure.get("paid_requests_started"))
            cost = failure.get("failed_attempt_cost_usd")
            runs.append(
                {
                    "optimizer": payload["optimizer"],
                    "mode": payload["mode"],
                    "status": "failed_preflight",
                    "classification": failure.get("classification"),
                    "started_at": failure.get("observed_at"),
                    "finished_at": failure.get("observed_at"),
                    "paid_requests_started": paid,
                    "paid_teacher_requests": failure.get("paid_teacher_requests"),
                    "optimization_cost_usd": cost,
                    "optimization_cost_status": (
                        failure.get("cost_status", "unavailable_after_exception")
                        if paid
                        else "no_paid_requests"
                    ),
                    "evaluation_cost_usd": 0.0 if not paid else None,
                    "evaluation_cost_status": "not_started",
                    "failure_artifact": str(
                        failure_path.relative_to(CHAPTER_DIR.parent)
                    ),
                }
            )
    recorded_lower_bound = sum(
        _money(run.get("optimization_cost_usd"))
        + _money(run.get("evaluation_cost_usd"))
        for run in runs
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cost_scope": "all newly executed smoke/full results and recorded preflight attempts",
        "recorded_cost_lower_bound_usd": recorded_lower_bound,
        "cost_status": "lower_bound because COPRO/MIPROv2 copied LM histories were detached before the accounting fix and two paid BootstrapFinetune preflight attempts ended before their teacher cost could be recovered",
        "runs": runs,
    }


def _gepa_row() -> dict[str, Any]:
    summary = _json(GEPA_ROOT / "final" / "summary.json")
    validation = _json(GEPA_ROOT / "gepa_candidate_2" / "validation" / "summary.json")
    statistical = _json(GEPA_ROOT / "final" / "statistical_analysis.json")
    return {
        "optimizer": "gepa",
        "display_name": DISPLAY_NAMES["gepa"],
        "status": "completed",
        "evaluation_protocol": "three fresh uncached runs with per-example majority vote",
        "task_model": summary["task_model"],
        "reflection_model": summary["reflection_model"],
        "baseline_accuracy_pct": summary["baseline_test_accuracy_pct"],
        "optimized_validation_accuracy_pct": validation["majority_accuracy_pct"],
        "locked_test_accuracy_pct": summary["optimized_test_accuracy_pct"],
        "locked_test_correct": summary["optimized_test_correct"],
        "locked_test_rows": 80,
        "absolute_uplift_pct_points": summary["absolute_uplift_pct_points"],
        "relative_uplift_pct": summary["relative_uplift_pct"],
        "optimization_cost_usd": summary["optimize_cost_usd"],
        "evaluation_cost_usd": summary["optimized_evaluation_cost_usd"],
        "optimization_time_seconds": summary["optimize_time_seconds"],
        "mean_inference_latency_seconds": summary["optimized_mean_latency_seconds"],
        "p95_inference_latency_seconds": summary["optimized_p95_latency_seconds"],
        "paired_mcnemar_p_value": summary["paired_mcnemar_p_value"],
        "paired_bootstrap_ci_low_pct_points": summary[
            "paired_bootstrap_ci_low_pct_points"
        ],
        "paired_bootstrap_ci_high_pct_points": summary[
            "paired_bootstrap_ci_high_pct_points"
        ],
        "program_artifact": "chapter06/results/gepa_expanded/final/optimized_program.json",
        "prompt_artifact": "chapter06/results/gepa_expanded/final/learned_prompt.json",
        "result_artifact": "chapter06/results/gepa_expanded/final/summary.json",
        "predictions_artifact": "chapter06/results/gepa_expanded/final/test_predictions.jsonl",
        "statistical_analysis": statistical,
        "source_run": "PR #8 frozen reference run",
    }


def _run_row(
    optimizer: str,
    *,
    canonical_baseline: float,
    baseline_predictions: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    result_path = RESULTS_ROOT / optimizer / "full" / "result.json"
    if not result_path.exists():
        return {
            "optimizer": optimizer,
            "display_name": DISPLAY_NAMES[optimizer],
            "status": "pending",
            "reason": "The expanded-dataset full run has not completed.",
        }
    result = _json(result_path)
    final = result["final"]
    validation = result["validation"]
    baseline = result.get("baseline")
    if optimizer == "quickstart":
        baseline_accuracy = final["accuracy"]
    elif baseline:
        baseline_accuracy = baseline["accuracy"]
    else:
        baseline_accuracy = canonical_baseline
    uplift = final["accuracy"] - baseline_accuracy
    optimization_cost_status, evaluation_cost_status = _cost_status(optimizer)
    evaluation_cost = result["evaluation_cost_usd"]
    if evaluation_cost_status != "recorded":
        evaluation_cost = None
    validation_parse_errors = sum(
        item.get("status", "completed") != "completed"
        for item in validation["predictions"]
    )
    test_parse_errors = sum(
        item.get("status", "completed") != "completed"
        for item in final["predictions"]
    )
    row: dict[str, Any] = {
        "optimizer": optimizer,
        "display_name": DISPLAY_NAMES[optimizer],
        "status": "completed",
        "evaluation_protocol": "one fresh uncached validation pass, then one locked-test pass",
        "task_model": result["task_model"],
        "reflection_model": (
            "openai/gpt-5.6-sol"
            if optimizer
            in {"copro", "miprov2", "simba", "bootstrap-finetune", "better-together"}
            else None
        ),
        "baseline_accuracy_pct": baseline_accuracy,
        "optimized_validation_accuracy_pct": validation["accuracy"],
        "locked_test_accuracy_pct": final["accuracy"],
        "locked_test_correct": final["correct"],
        "locked_test_rows": final["count"],
        "absolute_uplift_pct_points": uplift,
        "relative_uplift_pct": 100 * uplift / baseline_accuracy if baseline_accuracy else None,
        "optimization_cost_usd": result["optimization_cost_usd"],
        "optimization_cost_status": optimization_cost_status,
        "evaluation_cost_usd": evaluation_cost,
        "evaluation_cost_status": evaluation_cost_status,
        "optimization_time_seconds": result["optimization_seconds"],
        "mean_inference_latency_seconds": final["mean_latency_seconds"],
        "p95_inference_latency_seconds": final["p95_latency_seconds"],
        "accepted_trace_labels": result.get("accepted_trace_labels"),
        "validation_parse_error_count": validation_parse_errors,
        "locked_test_parse_error_count": test_parse_errors,
        "program_artifact": f"chapter06/results/expanded_notebooks/{optimizer}/full/optimized_program.json",
        "prompt_artifact": f"chapter06/results/expanded_notebooks/{optimizer}/full/learned_prompt.json",
        "result_artifact": f"chapter06/results/expanded_notebooks/{optimizer}/full/result.json",
        "predictions_artifact": f"chapter06/results/expanded_notebooks/{optimizer}/full/test_predictions.jsonl",
        "dspy_version": result["dspy_version"],
        "seed": result["seed"],
        "started_at": result["started_at"],
        "finished_at": result["finished_at"],
        "dataset_sha256": result["dataset_sha256"],
        "split_sha256": result["split_sha256"],
        "usage": result["usage"],
    }
    predictions = final["predictions"]
    if baseline_predictions and optimizer != "quickstart":
        paired = mcnemar_exact(baseline_predictions, predictions)
        bootstrap = paired_pair_bootstrap(
            baseline_predictions, predictions, samples=10_000, seed=42
        )
        row.update(
            {
                "paired_mcnemar_p_value": paired["p_value"],
                "paired_bootstrap_ci_low_pct_points": bootstrap[
                    "ci_low_pct_points"
                ],
                "paired_bootstrap_ci_high_pct_points": bootstrap[
                    "ci_high_pct_points"
                ],
                "statistical_analysis": {
                    "mcnemar": paired,
                    "paired_pair_bootstrap": bootstrap,
                },
            }
        )
    return row


def build() -> dict[str, Any]:
    canonical_baseline = _json(GEPA_ROOT / "final" / "summary.json")[
        "baseline_test_accuracy_pct"
    ]
    quickstart_result = RESULTS_ROOT / "quickstart" / "full" / "result.json"
    quickstart = _json(quickstart_result) if quickstart_result.exists() else None
    baseline_predictions = quickstart["final"]["predictions"] if quickstart else None
    single_pass_baseline = (
        float(quickstart["final"]["accuracy"]) if quickstart else canonical_baseline
    )
    rows = []
    for optimizer in ORDER:
        rows.append(
            _gepa_row()
            if optimizer == "gepa"
            else _run_row(
                optimizer,
                canonical_baseline=single_pass_baseline,
                baseline_predictions=baseline_predictions,
            )
        )
    completed = [row for row in rows if row["status"] == "completed"]
    comparison = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": "data/ai_vs_human_chapter06_expanded.csv",
        "dataset_sha256": _sha256(DATA_PATH),
        "split_path": "data/ai_vs_human_chapter06_expanded_splits.json",
        "split_sha256": _sha256(SPLIT_PATH),
        "split_rows": {"train": 160, "validation": 60, "test": 80},
        "selection_policy": "optimizer choices use train/validation only; locked test is evaluated after the program is frozen",
        "canonical_luna_baseline_accuracy_pct": canonical_baseline,
        "completed_count": len(completed),
        "total_count": len(rows),
        "new_run_cost_usd": sum(
            _money(row.get("optimization_cost_usd"))
            + _money(row.get("evaluation_cost_usd"))
            for row in completed
            if row["optimizer"] != "gepa"
        ),
        "new_run_cost_status": "recorded_lower_bound; COPRO and MIPROv2 detached some copied LM histories before the shared-history fix",
        "rows": rows,
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RESULTS_ROOT / "comparison.json", comparison)
    atomic_write_json(RESULTS_ROOT / "run_ledger.json", _build_run_ledger())
    fields = [
        "optimizer",
        "display_name",
        "status",
        "task_model",
        "baseline_accuracy_pct",
        "optimized_validation_accuracy_pct",
        "locked_test_accuracy_pct",
        "absolute_uplift_pct_points",
        "relative_uplift_pct",
        "optimization_cost_usd",
        "evaluation_cost_usd",
        "optimization_time_seconds",
        "mean_inference_latency_seconds",
        "p95_inference_latency_seconds",
    ]
    with (RESULTS_ROOT / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return comparison


def markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Chapter 6 expanded-dataset optimizer results",
        "",
        "All programs use the canonical 300-row dataset and locked pair-grouped split: "
        "160 train, 60 validation, and 80 test rows. Optimizer selection used validation only; "
        "the locked test was released after each program was frozen.",
        "",
        "| Optimizer | Baseline | Validation | Locked test | Uplift | Opt. cost | Eval. cost | Opt. time | Mean / p95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        if row["status"] != "completed":
            lines.append(
                f"| {row['display_name']} | — | — | {row['status']} | — | — | — | — | — |"
            )
            continue
        cost = row["optimization_cost_usd"]
        cost_text = f"${cost:.4f}"
        if row.get("optimization_cost_status") == "partial_lower_bound":
            cost_text = f">={cost_text}*"
        evaluation_cost = row.get("evaluation_cost_usd")
        evaluation_cost_text = (
            f"${evaluation_cost:.4f}" if evaluation_cost is not None else "unavailable*"
        )
        lines.append(
            "| {display_name} | {baseline_accuracy_pct:.2f}% | "
            "{optimized_validation_accuracy_pct:.2f}% | {locked_test_accuracy_pct:.2f}% "
            "({locked_test_correct}/{locked_test_rows}) | {absolute_uplift_pct_points:+.2f} pp | "
            f"{cost_text} | {evaluation_cost_text} | "
            "{optimization_time_seconds:.1f}s | {mean_inference_latency_seconds:.3f}s / "
            "{p95_inference_latency_seconds:.3f}s |".format(**row)
        )
    lines.extend(
        [
            "",
            "GEPA is the frozen PR #8 result and uses three fresh uncached evaluation passes with "
            "per-example majority vote. Newly executed rows report one uncached pass; that protocol "
            "difference is retained explicitly rather than normalized away.",
            "",
            "`*` COPRO and MIPROv2 optimization cost is a recorded lower bound, and their "
            "evaluation cost is unavailable: those runs preceded the shared-history fix for "
            "deep-copied DSPy language models. Their scores and wall-clock timings remain valid.",
            "",
            "Machine-readable rows, paired statistics, hashes, model/version metadata, prompts, "
            "programs, predictions, cost, timing, and failure manifests are under "
            "`chapter06/results/expanded_notebooks/`.",
            "Local-model responses that could not be parsed are retained in the predictions as "
            "incorrect with `status: parse_error`; they are never dropped from a denominator.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    comparison = build()
    (CHAPTER_DIR / "CHAPTER_RESULTS.md").write_text(
        markdown(comparison), encoding="utf-8"
    )
    print(
        f"Built expanded comparison: {comparison['completed_count']}/{comparison['total_count']} completed"
    )


if __name__ == "__main__":
    main()
