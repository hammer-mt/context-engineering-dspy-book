"""Helpers for previews of the checked-in Chapter 6 program artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CHAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = CHAPTER_DIR.parent
SUMMARY_PATH = CHAPTER_DIR / "results" / "expanded_notebooks" / "comparison.json"


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
    """Return the one comparable publication number: final test accuracy."""

    row = optimizer_row(optimizer)
    lines = [
        f"{row['display_name']} — frozen full-profile rerun",
        f"status: {row['status']}",
    ]
    if row["status"] == "completed":
        accuracy = row["locked_test_accuracy_pct"]
        lines.extend(
            [
                f"task model: {row.get('task_model', '—')}",
                f"validation accuracy: {row['optimized_validation_accuracy_pct']:.1f}%",
                f"locked-test accuracy: {accuracy:.1f}% ({row['locked_test_correct']}/{row['locked_test_rows']})",
                f"optimization cost: ${row['optimization_cost_usd']:.4f}",
                f"optimization time: {_seconds(row['optimization_time_seconds'])}",
            ]
        )
        counts = row.get("accepted_trace_labels")
        if counts:
            lines.append(
                f"accepted traces: human={counts['human']}, AI={counts['ai']}"
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
                f"- program snapshot: {row['program_artifact']}",
                f"- prompt snapshot: {row['prompt_artifact']}",
                "- chapter comparison: chapter06/CHAPTER_RESULTS.md",
            ]
        )
    return "\n".join(lines)


def learned_program_preview(optimizer: str, *, instruction_chars: int = 1_800) -> str:
    """Show learned instructions and demonstrations without dumping large JSON blobs."""

    row = optimizer_row(optimizer)
    prompt_path = REPO_ROOT / row.get(
        "prompt_artifact",
        f"chapter06/results/expanded_notebooks/{optimizer}/full/learned_prompt.json",
    )
    if not prompt_path.exists():
        return "No learned prompt artifact exists for this optimizer."
    payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    prompts = payload.get("predictors", payload)
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

    row = optimizer_row(optimizer)
    prompt_path = REPO_ROOT / row.get(
        "prompt_artifact",
        f"chapter06/results/expanded_notebooks/{optimizer}/full/learned_prompt.json",
    )
    program_path = REPO_ROOT / row.get(
        "program_artifact",
        f"chapter06/results/expanded_notebooks/{optimizer}/full/optimized_program.json",
    )
    if not prompt_path.exists() or not program_path.exists():
        return {
            "checked": False,
            "reason": "serialized program or extracted prompt is missing",
        }
    payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    prompts = payload.get("predictors", payload)
    program = json.loads(program_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for predictor_name, expected in prompts.items():
        actual = program.get(predictor_name, {})
        actual_prompt = {
            "instructions": actual.get("signature", {}).get("instructions", ""),
            "demos": actual.get("demos", []),
        }
        expected_prompt = {
            "instructions": expected.get("instructions", ""),
            "demos": expected.get("demos", []),
        }
        if actual_prompt != expected_prompt:
            mismatches.append(predictor_name)
    return {
        "checked": True,
        "predictors": len(prompts),
        "prompt_state_equal": not mismatches,
        "mismatches": mismatches,
    }
