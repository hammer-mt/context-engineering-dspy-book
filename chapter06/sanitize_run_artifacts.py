"""Sanitize local paths and refresh manifests for Chapter 6 run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from chapter06.experiments.gepa_expanded.guardrails import atomic_write_json
from chapter06.optimizer_runtime import RESULTS_ROOT
from chapter06.run_live_optimizer import REPO_ROOT, _model_artifact_manifest


def _replace_repo_path(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    sanitized = text.replace(f"{REPO_ROOT}/", "")
    if sanitized == text:
        return False
    path.write_text(sanitized, encoding="utf-8")
    return True


def sanitize() -> dict[str, int]:
    rewritten = 0
    manifests = 0
    for path in RESULTS_ROOT.rglob("*"):
        if not path.is_file() or "model" in path.parts:
            continue
        if path.suffix in {".json", ".jsonl", ".txt", ".md", ".csv"}:
            rewritten += int(_replace_repo_path(path))

    for mode_dir in sorted(RESULTS_ROOT.glob("*/smoke")) + sorted(
        RESULTS_ROOT.glob("*/full")
    ):
        model_dir = mode_dir / "model"
        manifest = _model_artifact_manifest(str(model_dir))
        if manifest:
            atomic_write_json(mode_dir / "model_artifact_manifest.json", manifest)
            manifests += 1
            result_path = mode_dir / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["output_dir"] = manifest["model_directory"]
                atomic_write_json(result_path, result)

    return {"rewritten_files": rewritten, "model_manifests": manifests}


def main() -> None:
    print(json.dumps(sanitize(), sort_keys=True))


if __name__ == "__main__":
    main()
