"""Helpers that preview the saved Chapter 6 programs and prompts."""

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


def artifact_paths(optimizer: str) -> str:
    row = optimizer_row(optimizer)
    lines = ["Saved artifacts:"]
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
    """Show the saved instruction and demonstrations in a compact form."""

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
                + "\n… [preview truncated; open the prompt snapshot listed above for the full text]"
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
                f"… {len(demos) - 4} more demonstrations in the prompt snapshot"
            )
    return "\n".join(lines)


def verify_prompt_artifact(optimizer: str) -> dict[str, Any]:
    """Check that the saved program contains the separately saved prompt."""

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
