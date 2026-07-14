"""Freeze canonical Chapter 6 programs/prompts and build the benchmark table."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from chapter06.optimizer_experiment import OPTIMIZER_NAMES, RESULTS_ROOT, SPLIT_PATH


REPO_ROOT = Path(__file__).resolve().parents[1]
CHAPTER_DIR = REPO_ROOT / "chapter06"
SUMMARY_JSON = CHAPTER_DIR / "results" / "benchmark_summary.json"
SUMMARY_MARKDOWN = CHAPTER_DIR / "results" / "benchmark_summary.md"
SUMMARY_CSV = CHAPTER_DIR / "results" / "benchmark_summary.csv"
GATE_PATH = CHAPTER_DIR / "results" / "dataset" / "baseline_gate.json"
FROZEN_PROGRAMS_DIR = CHAPTER_DIR / "optimized_programs" / "final"
FROZEN_PROMPTS_DIR = CHAPTER_DIR / "results" / "final_prompts"

DISPLAY_NAMES = {
    "quickstart": "Unoptimized baseline",
    "labeled-few-shot": "LabeledFewShot",
    "bootstrap-few-shot": "BootstrapFewShot",
    "bootstrap-random-search": "BootstrapRS",
    "knn-few-shot": "KNNFewShot",
    "copro": "COPRO",
    "miprov2": "MIPROv2",
    "gepa": "GEPA",
    "simba": "SIMBA",
    "ensemble": "Ensemble",
    "bootstrap-finetune": "BootstrapFinetune",
    "better-together": "BetterTogether",
}


def _latest_matching_run(optimizer: str, dataset_manifest: dict[str, Any]) -> Path | None:
    root = RESULTS_ROOT / "full" / optimizer
    for manifest_path in sorted(root.glob("*/manifest.json"), reverse=True):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_manifest") == dataset_manifest and manifest.get("status") in {
            "completed",
            "hardware_blocked",
        }:
            return manifest_path.parent
    return None


def _cost(metrics: dict[str, Any], key: str) -> float:
    return float(metrics.get(key, {}).get("cost_usd", 0.0))


def _seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 60:
        return f"{value:.1f}s"
    return f"{value / 60:.1f}min"


def build_summary() -> dict[str, Any]:
    dataset_manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    baseline_dir = _latest_matching_run("quickstart", dataset_manifest)
    if baseline_dir is None:
        raise RuntimeError("no completed full baseline matches the frozen split")
    baseline_metrics = json.loads((baseline_dir / "metrics.json").read_text(encoding="utf-8"))
    reference_baseline = float(baseline_metrics["baseline_score"])

    FROZEN_PROGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    FROZEN_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for optimizer in OPTIMIZER_NAMES:
        run_dir = _latest_matching_run(optimizer, dataset_manifest)
        if run_dir is None:
            rows.append(
                {
                    "optimizer": optimizer,
                    "display_name": DISPLAY_NAMES[optimizer],
                    "status": "missing",
                }
            )
            continue
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "optimizer": optimizer,
            "display_name": DISPLAY_NAMES[optimizer],
            "status": manifest["status"],
            "run_id": manifest["run_id"],
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        }
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if manifest["status"] == "completed":
            score = float(metrics["optimized_score"])
            row.update(
                {
                    "run_baseline_accuracy": float(metrics["baseline_score"]),
                    "accuracy": score,
                    "uplift_vs_reference_points": score - reference_baseline,
                    "relative_uplift_vs_reference_percent": (
                        100 * (score - reference_baseline) / reference_baseline
                        if reference_baseline
                        else None
                    ),
                    "run_uplift_points": float(metrics["uplift_points"]),
                    "optimization_cost_usd": _cost(metrics, "optimization_usage"),
                    "evaluation_cost_usd": _cost(metrics, "optimized_evaluation_usage"),
                    "total_measured_cost_usd": sum(
                        _cost(metrics, key)
                        for key in (
                            "baseline_usage",
                            "optimization_usage",
                            "optimized_evaluation_usage",
                        )
                    ),
                    "optimization_seconds": float(metrics["optimization_seconds"]),
                    "mean_inference_latency_seconds": float(
                        metrics["optimized_latency"]["mean_seconds"]
                    ),
                    "p95_inference_latency_seconds": float(
                        metrics["optimized_latency"]["p95_seconds"]
                    ),
                    "reload_prompt_parity": bool(metrics["reload_prompt_parity"]),
                    "artifacts": {
                        name: str((run_dir / filename).relative_to(REPO_ROOT))
                        for name, filename in manifest.get("artifacts", {}).items()
                    },
                }
            )
            shutil.copyfile(run_dir / "program.json", FROZEN_PROGRAMS_DIR / f"{optimizer}.json")
            shutil.copyfile(run_dir / "prompts.json", FROZEN_PROMPTS_DIR / f"{optimizer}.json")
        else:
            row["reason"] = manifest.get("reason", "")
        rows.append(row)

    summary = {
        "schema_version": 1,
        "dataset_manifest": dataset_manifest,
        "baseline_gate": {
            "criterion": gate["criterion"],
            "selection_replicates": gate["repeat_metrics"],
            "selection_mean_accuracy": gate["mean_accuracy"],
            "independent_confirmation_accuracy": reference_baseline,
            "disclosure": gate["disclosure"],
        },
        "reference_baseline_accuracy": reference_baseline,
        "rows": rows,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_fields = [
        "optimizer",
        "display_name",
        "status",
        "accuracy",
        "uplift_vs_reference_points",
        "relative_uplift_vs_reference_percent",
        "run_baseline_accuracy",
        "run_uplift_points",
        "optimization_cost_usd",
        "evaluation_cost_usd",
        "total_measured_cost_usd",
        "optimization_seconds",
        "mean_inference_latency_seconds",
        "p95_inference_latency_seconds",
        "reload_prompt_parity",
        "run_dir",
        "reason",
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=csv_fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    selection_scores = ", ".join(
        f"{item['accuracy']:.1f}%" for item in gate["repeat_metrics"]
    )
    markdown = [
        "# Chapter 6 optimizer benchmark",
        "",
        (
            f"Frozen reference baseline: **{reference_baseline:.1f}%**. "
            f"Adversarial selection replicates: {selection_scores}."
        ),
        "",
        (
            "| Optimizer | Accuracy | "
            f"Uplift vs. {reference_baseline:.1f}% reference | Relative uplift | Optimize cost | "
            "Optimize time | Mean latency | P95 latency |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] == "completed":
            markdown.append(
                "| {display_name} | {accuracy:.1f}% | {uplift_vs_reference_points:+.1f} pts | "
                "{relative_uplift_vs_reference_percent:+.1f}% | ${optimization_cost_usd:.4f} | "
                "{time} | {mean_inference_latency_seconds:.2f}s | "
                "{p95_inference_latency_seconds:.2f}s |".format(
                    **row, time=_seconds(row["optimization_seconds"])
                )
            )
        else:
            markdown.append(
                f"| {row['display_name']} | {row['status']} | — | — | — | — | — | — |"
            )
    markdown.extend(
        [
            "",
            f"> {gate['disclosure']}",
            "",
            "Every completed row links to a run directory in `benchmark_summary.json`. Each directory preserves "
            "the full console output, LM call history, per-example predictions, serialized program, extracted "
            "prompt, metrics, and manifest. Canonical frozen programs live in `chapter06/optimized_programs/final/` "
            "and prompts in `chapter06/results/final_prompts/`.",
        ]
    )
    SUMMARY_MARKDOWN.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = build_summary()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
