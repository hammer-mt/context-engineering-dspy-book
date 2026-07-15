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
CHAPTER_RESULTS = CHAPTER_DIR / "CHAPTER_RESULTS.md"
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
MIN_RELOAD_PARITY_EXAMPLES = 3


def _has_publishable_reload_checks(run_dir: Path) -> bool:
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    parity = metrics.get("reload_prediction_parity") or {}
    checked = parity.get("checked", 0)
    completed = parity.get("completed")
    matching = parity.get("matching")
    return bool(
        metrics.get("reload_prompt_parity")
        and isinstance(checked, int)
        and checked >= MIN_RELOAD_PARITY_EXAMPLES
        and completed == checked
        and isinstance(matching, int)
        and 0 <= matching <= checked
    )


def _latest_matching_run(
    optimizer: str, dataset_manifest: dict[str, Any]
) -> Path | None:
    root = RESULTS_ROOT / "full" / optimizer
    for manifest_path in sorted(root.glob("*/manifest.json"), reverse=True):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_manifest") != dataset_manifest:
            continue
        if manifest.get("status") == "hardware_blocked":
            return manifest_path.parent
        if manifest.get("status") == "completed" and _has_publishable_reload_checks(
            manifest_path.parent
        ):
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


def _result_table(rows: list[dict[str, Any]], reference_baseline: float) -> list[str]:
    lines = [
        (
            "| Optimizer | Accuracy | "
            f"Uplift vs. {reference_baseline:.1f}% reference | Optimize cost | "
            "Optimize time | Mean latency | P95 latency | Reload parity |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] == "completed":
            lines.append(
                "| {display_name} | {accuracy:.1f}% | {uplift_vs_reference_points:+.1f} pts | "
                "${optimization_cost_usd:.4f} | {time} | {mean_inference_latency_seconds:.2f}s | "
                "{p95_inference_latency_seconds:.2f}s | {parity[matching]}/{parity[checked]} |".format(
                    **row,
                    time=_seconds(row["optimization_seconds"]),
                    parity=row["reload_prediction_parity"],
                )
            )
        else:
            lines.append(
                f"| {row['display_name']} | {row['status']} | — | — | — | — | — | — |"
            )
    return lines


def _chapter_results_markdown(summary: dict[str, Any]) -> str:
    """Create a manuscript-ready interpretation directly from the selected reruns."""

    rows = summary["rows"]
    reference = float(summary["reference_baseline_accuracy"])
    dataset = summary["dataset_manifest"]
    completed = [row for row in rows if row["status"] == "completed"]
    optimized = [row for row in completed if row["optimizer"] != "quickstart"]
    best_score = max(row["accuracy"] for row in optimized)
    leaders = [row for row in optimized if row["accuracy"] == best_score]
    leader_names = " and ".join(row["display_name"] for row in leaders)
    fastest_leader = min(leaders, key=lambda row: row["optimization_seconds"])
    free_compile = [
        row
        for row in optimized
        if row["optimization_cost_usd"] == 0 and row["uplift_vs_reference_points"] > 0
    ]
    best_free = max(free_compile, key=lambda row: row["accuracy"], default=None)
    slowest_inference = max(
        completed, key=lambda row: row["mean_inference_latency_seconds"]
    )
    total_measured_cost = sum(row["total_measured_cost_usd"] for row in completed)
    parity_matching = sum(
        row["reload_prediction_parity"]["matching"] for row in completed
    )
    parity_checked = sum(
        row["reload_prediction_parity"]["checked"] for row in completed
    )
    blocked = [row for row in rows if row["status"] == "hardware_blocked"]
    gate = summary["baseline_gate"]
    selection_scores = ", ".join(
        f"{item['accuracy']:.0f}%" for item in gate["selection_replicates"]
    )

    lines = [
        "# Chapter 6 benchmark: rerun-backed optimizer comparison",
        "",
        "## Experimental frame",
        "",
        (
            f"The frozen benchmark contains {dataset['row_count']} passages in "
            f"{dataset['pair_count']} human/AI semantic pairs. Pair IDs—not individual rows—are the "
            "split unit, so a source passage and its rewrite cannot leak across training, validation, "
            "and test. This rerun reused the checked-in split manifest and dataset hash; it did not "
            "regenerate or redesign the adversarial data."
        ),
        "",
        (
            f"The test partition is intentionally adversarial: baseline selection replicates scored "
            f"{selection_scores}, and the frozen reference baseline for the optimizer comparison is "
            f"{reference:.0f}%. Treat this as a stress test of optimization behavior, not as an unbiased "
            "estimate of real-world AI-detection accuracy."
        ),
        "",
        "## Results",
        "",
        *_result_table(rows, reference),
        "",
        "## Interpretation",
        "",
        (
            f"{leader_names} led the locked test set at {best_score:.0f}%, a "
            f"{best_score - reference:+.0f}-point improvement over the frozen baseline. Among the "
            f"leaders, {fastest_leader['display_name']} compiled fastest "
            f"({_seconds(fastest_leader['optimization_seconds'])}) and used "
            f"${fastest_leader['optimization_cost_usd']:.4f} in measured optimization calls."
        ),
    ]
    if best_free is not None:
        lines.extend(
            [
                "",
                (
                    f"The strongest zero-paid-compile option was {best_free['display_name']} at "
                    f"{best_free['accuracy']:.0f}%. That makes it a useful first move when iteration "
                    "speed and cost matter more than extracting the final few points. Zero compile "
                    "cost does not mean zero inference cost: the table separates optimization spend "
                    "from evaluation latency for exactly that reason."
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                f"Inference tradeoffs are visible as well. {slowest_inference['display_name']} had the "
                f"highest mean latency ({slowest_inference['mean_inference_latency_seconds']:.2f}s), "
                "so its accuracy should be weighed against serving cost rather than read in isolation. "
                "The learned instructions and demonstrations are preserved beside each notebook, which "
                "makes qualitative inspection part of the comparison instead of treating accuracy as "
                "the only outcome."
            ),
            "",
            "## Reproducibility and limits",
            "",
            (
                f"The completed rows used ${total_measured_cost:.2f} in measured baseline, optimization, "
                "optimized-evaluation, and bounded reload-verification calls. Every selected run includes "
                "its manifest, console transcript, sanitized LM history, per-example predictions, metrics, "
                "serialized program, and extracted prompt under `chapter06/results/runs/full/`. Canonical "
                "programs and prompts are copied to `chapter06/optimized_programs/final/` and "
                "`chapter06/results/final_prompts/`."
            ),
            "",
            (
                "Serialization was checked two ways where the optimizer ran: prompt/demo state equality "
                f"after reload and bounded prediction-label parity on frozen test examples. Across the "
                f"selected runs, {parity_matching}/{parity_checked} reloaded predictions matched their "
                "pre-serialization labels. The per-run counts are reported in the table because a mismatch "
                "from an uncached stochastic model is evidence to preserve, not a reason to retry until it "
                "disappears; this remains a bounded reproducibility check rather than a claim of global "
                "determinism."
            ),
        ]
    )
    if blocked:
        names = " and ".join(row["display_name"] for row in blocked)
        lines.extend(
            [
                "",
                (
                    f"{names} remain hardware-blocked on the recorded Darwin/arm64 host because their "
                    "weight-optimization paths require an NVIDIA CUDA training stack. Their notebooks "
                    "execute safely, explain the missing result, and show the explicit full-run command; "
                    "no score is imputed for unsupported hardware."
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                "Finally, the test set has only 20 passages, so one changed prediction moves accuracy by "
                "five points. The comparison is most useful for understanding optimizer mechanisms and "
                "tradeoffs under one frozen adversarial workload, not for declaring a universal ranking."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def build_summary() -> dict[str, Any]:
    dataset_manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    baseline_dir = _latest_matching_run("quickstart", dataset_manifest)
    if baseline_dir is None:
        raise RuntimeError("no completed full baseline matches the frozen split")
    baseline_metrics = json.loads(
        (baseline_dir / "metrics.json").read_text(encoding="utf-8")
    )
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
                            "reload_verification_usage",
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
                    "reload_prediction_parity": metrics.get("reload_prediction_parity"),
                    "artifacts": {
                        name: str((run_dir / filename).relative_to(REPO_ROOT))
                        for name, filename in manifest.get("artifacts", {}).items()
                    },
                }
            )
            shutil.copyfile(
                run_dir / "program.json", FROZEN_PROGRAMS_DIR / f"{optimizer}.json"
            )
            shutil.copyfile(
                run_dir / "prompts.json", FROZEN_PROMPTS_DIR / f"{optimizer}.json"
            )
        else:
            row["reason"] = manifest.get("reason", "")
        rows.append(row)

    summary = {
        "schema_version": 2,
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
        "reload_prediction_parity",
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
    CHAPTER_RESULTS.write_text(_chapter_results_markdown(summary), encoding="utf-8")
    return summary


def main() -> int:
    summary = build_summary()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
