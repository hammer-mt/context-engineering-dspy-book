"""Run one Chapter 6 optimizer over the expanded locked split and save artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import dspy

from chapter06.experiments.gepa_expanded.guardrails import (
    atomic_write_json,
    classify_api_error,
    write_jsonl,
)
from chapter06.experiments.gepa_expanded.run_experiment import _save_program_artifacts
from chapter06.experiments.gepa_expanded.runtime import load_project_env

from chapter06.optimizer_runtime import (
    OPTIMIZERS,
    DATA_PATH,
    RESULTS_ROOT,
    SPLIT_PATH,
    format_result,
    load_frozen_examples,
    run_optimizer,
    split_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _display_path(path: Path) -> str:
    """Return a durable repository-relative path when possible."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("optimizer", choices=sorted(OPTIMIZERS))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifact_manifest(output_dir: str | None) -> dict | None:
    if not output_dir:
        return None
    model_dir = Path(output_dir)
    if not model_dir.exists():
        return None
    files = []
    for name in ("adapter_model.safetensors", "model.safetensors", "config.json"):
        path = model_dir / name
        if path.exists():
            files.append(
                {
                    "path": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {
        "schema_version": 1,
        "model_directory": _display_path(model_dir),
        "storage_policy": "generated model payload is local and Git-ignored; hashes, configuration, program, prompt, predictions, and scores are versioned",
        "files": files,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_project_env()
    artifact_dir = args.artifact_dir or RESULTS_ROOT / args.optimizer / args.mode
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "optimizer": args.optimizer,
        "mode": args.mode,
        "status": "running",
        "started_at": started_at.isoformat(),
        "dataset_path": str(DATA_PATH.relative_to(DATA_PATH.parents[1])),
        "dataset_sha256": _sha256(DATA_PATH),
        "split_path": str(SPLIT_PATH.relative_to(SPLIT_PATH.parents[1])),
        "split_sha256": _sha256(SPLIT_PATH),
        "seed": 42,
        "dspy_version": dspy.__version__,
        "python_version": platform.python_version(),
        "num_retries": int(os.getenv("CHAPTER06_NUM_RETRIES", "1")),
        "test_release_policy": "full mode evaluates locked test once after compilation and validation; smoke mode never reads test examples",
    }
    atomic_write_json(artifact_dir / "run_manifest.json", manifest)
    splits = load_frozen_examples()
    print(f"Frozen split: {split_summary(splits)}", flush=True)
    try:
        output_dir = args.output_dir
        if args.optimizer in {"bootstrap-finetune", "better-together"}:
            output_dir = output_dir or artifact_dir / "model"
        run = run_optimizer(
            args.optimizer,
            splits=splits,
            include_baseline=args.include_baseline,
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            mode=args.mode,
        )
    except Exception as exc:
        manifest.update(
            {
                "status": classify_api_error(exc),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(artifact_dir / "run_manifest.json", manifest)
        raise
    result = run.summary()
    model_manifest = _model_artifact_manifest(run.output_dir)
    if model_manifest:
        result["output_dir"] = model_manifest["model_directory"]
        atomic_write_json(artifact_dir / "model_artifact_manifest.json", model_manifest)
        from chapter06.apple_finetune import local_capabilities

        result["local_capabilities"] = local_capabilities()
    result["baseline"] = run.baseline
    result["final"] = run.final
    result["validation"] = run.validation
    result["usage"] = run.usage
    result.update(
        {
            "dataset_sha256": manifest["dataset_sha256"],
            "split_sha256": manifest["split_sha256"],
            "seed": 42,
            "dspy_version": dspy.__version__,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    print(format_result(result), flush=True)
    _save_program_artifacts(run.program, artifact_dir)
    write_jsonl(artifact_dir / "validation_predictions.jsonl", run.validation["predictions"])
    if run.final is not None:
        write_jsonl(artifact_dir / "test_predictions.jsonl", run.final["predictions"])
    result_path = args.result_json or artifact_dir / "result.json"
    atomic_write_json(result_path, result)
    manifest.update(
        {
            "status": "completed",
            "finished_at": result["finished_at"],
            "optimization_cost_usd": run.optimization_cost_usd,
            "evaluation_cost_usd": run.evaluation_cost_usd,
            "result_path": _display_path(result_path),
        }
    )
    atomic_write_json(artifact_dir / "run_manifest.json", manifest)
    print(f"Saved run artifacts: {artifact_dir}", flush=True)


if __name__ == "__main__":
    main()
