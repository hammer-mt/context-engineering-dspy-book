"""Dependency-free helpers for the published Chapter 6 notebooks.

The notebooks inspect compact, checked-in results and make no network calls. This
lets readers run the educational material without credentials or a DSPy install.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CHAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = CHAPTER_DIR.parent
SUMMARY_PATH = CHAPTER_DIR / "results" / "benchmark_summary.json"


def load_summary(path: Path = SUMMARY_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def optimizer_row(
    optimizer: str, summary: dict[str, Any] | None = None
) -> dict[str, Any]:
    summary = summary or load_summary()
    try:
        return next(row for row in summary["rows"] if row["optimizer"] == optimizer)
    except StopIteration as exc:
        raise KeyError(
            f"optimizer {optimizer!r} is not present in {SUMMARY_PATH}"
        ) from exc


def _seconds(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}s" if value < 60 else f"{value / 60:.1f}min"


def benchmark_snapshot(optimizer: str) -> str:
    """Return a compact, stable text summary suitable for saved notebook output."""

    row = optimizer_row(optimizer)
    lines = [
        f"{row['display_name']} — frozen full-profile rerun",
        f"status: {row['status']}",
    ]
    if row["status"] == "completed":
        parity = row.get("reload_prediction_parity") or {}
        parity_text = (
            f"{parity.get('matching', '—')}/{parity.get('checked', '—')} labels"
            if parity
            else "not recorded by this run"
        )
        uplift = (
            f"{row['uplift_vs_reference_points']:+.1f} points vs Luna reference"
            if row.get("reference_comparable", True)
            else f"{row['run_uplift_points']:+.1f} points vs its Qwen run baseline"
        )
        lines.extend(
            [
                f"task model: {row.get('task_model', '—')}",
                f"test accuracy: {row['accuracy']:.1f}%",
                f"uplift: {uplift}",
                f"optimization: ${row['optimization_cost_usd']:.4f} and {_seconds(row['optimization_seconds'])}",
                (
                    "inference latency: "
                    f"mean {row['mean_inference_latency_seconds']:.2f}s; "
                    f"p95 {row['p95_inference_latency_seconds']:.2f}s"
                ),
                (
                    f"reload checks: prompt={row['reload_prompt_parity']}; "
                    f"model={row.get('reload_model_parity', 'n/a')}; predictions={parity_text}"
                ),
            ]
        )
        if not row.get("reference_comparable", True):
            lines.append(
                "comparison boundary: same frozen split, separate Qwen/MPS experiment; not Luna-comparable"
            )
    else:
        lines.append(
            f"reason: {row.get('reason', 'No runnable artifact is available.')}"
        )
    return "\n".join(lines)


def artifact_paths(optimizer: str) -> str:
    row = optimizer_row(optimizer)
    lines = ["Published artifacts:"]
    if row["status"] == "completed":
        lines.extend(
            [
                f"- canonical program snapshot: chapter06/optimized_programs/final/{optimizer}.json",
                f"- canonical prompt: chapter06/results/final_prompts/{optimizer}.json",
                "- chapter comparison: chapter06/CHAPTER_RESULTS.md",
            ]
        )
    return "\n".join(lines)


def learned_program_preview(optimizer: str, *, instruction_chars: int = 1_800) -> str:
    """Show learned instructions and demonstrations without dumping large JSON blobs."""

    prompt_path = CHAPTER_DIR / "results" / "final_prompts" / f"{optimizer}.json"
    if not prompt_path.exists():
        return "No learned prompt artifact exists for this optimizer."
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for predictor_name, state in prompts.items():
        instruction = str(state.get("instructions", "")).strip()
        if len(instruction) > instruction_chars:
            instruction = (
                instruction[:instruction_chars].rstrip()
                + "\n… [preview truncated; open prompts.json for all text]"
            )
        demos = state.get("demos", [])
        lines.extend(
            [
                f"Predictor: {predictor_name}",
                f"Learned instruction ({len(state.get('instructions', ''))} characters):",
                instruction or "[empty]",
                f"\nDemonstrations: {len(demos)}",
            ]
        )
        for index, demo in enumerate(demos[:4], start=1):
            text = " ".join(str(demo.get("text", "")).split())
            if len(text) > 180:
                text = text[:177].rstrip() + "…"
            label = demo.get("is_ai", demo.get("answer", "?"))
            lines.append(f"{index}. is_ai={label}: {text}")
        if len(demos) > 4:
            lines.append(
                f"… {len(demos) - 4} more demos in the complete prompt artifact"
            )
    return "\n".join(lines)


def verify_prompt_artifact(optimizer: str) -> dict[str, Any]:
    """Check that the frozen program state contains the separately extracted prompt."""

    prompt_path = CHAPTER_DIR / "results" / "final_prompts" / f"{optimizer}.json"
    program_path = CHAPTER_DIR / "optimized_programs" / "final" / f"{optimizer}.json"
    if not prompt_path.exists() or not program_path.exists():
        return {
            "checked": False,
            "reason": "serialized program or extracted prompt is missing",
        }
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    program = json.loads(program_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for predictor_name, expected in prompts.items():
        actual = program.get(predictor_name, {})
        actual_prompt = {
            "instructions": actual.get("signature", {}).get("instructions", ""),
            "demos": actual.get("demos", []),
        }
        if actual_prompt != expected:
            mismatches.append(predictor_name)
    return {
        "checked": True,
        "predictors": len(prompts),
        "prompt_state_equal": not mismatches,
        "mismatches": mismatches,
    }
