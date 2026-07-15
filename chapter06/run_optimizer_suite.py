"""Run the Chapter 6 optimizers sequentially and stop on the first failure."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from chapter06.optimizer_experiment import (
    OPTIMIZER_NAMES,
    RESULTS_ROOT,
    SPLIT_PATH,
    run_experiment,
)


def _has_completed_run(profile: str, optimizer: str) -> bool:
    root = RESULTS_ROOT / profile / optimizer
    current_dataset_manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    for manifest_path in sorted(root.glob("*/manifest.json"), reverse=True):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "completed"
            and manifest.get("dataset_manifest") == current_dataset_manifest
        ):
            return True
    return False


def run_suite(
    *, profile: str, optimizers: Sequence[str], skip_existing: bool
) -> list[dict]:
    results: list[dict] = []
    for optimizer in optimizers:
        if skip_existing and _has_completed_run(profile, optimizer):
            print(f"SKIP {optimizer}: completed {profile} artifact already exists")
            continue
        print(f"RUN {optimizer} ({profile})")
        result = run_experiment(optimizer, profile_name=profile)
        results.append(result)
        print(f"DONE {optimizer}: {result.get('status')}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--optimizer", action="append", choices=OPTIMIZER_NAMES)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_suite(
        profile=args.profile,
        optimizers=args.optimizer or OPTIMIZER_NAMES,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
