"""Run one Chapter 6 optimizer over the frozen split and save compact results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chapter06.optimizer_runtime import (
    OPTIMIZERS,
    format_result,
    load_frozen_examples,
    run_optimizer,
    split_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("optimizer", choices=sorted(OPTIMIZERS))
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    splits = load_frozen_examples()
    print(f"Frozen split: {split_summary(splits)}", flush=True)
    run = run_optimizer(
        args.optimizer,
        splits=splits,
        include_baseline=args.include_baseline,
        output_dir=args.output_dir,
    )
    result = run.summary()
    result["baseline"] = run.baseline
    result["final"] = run.final
    print(format_result(result), flush=True)
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"Saved compact result: {args.result_json}", flush=True)


if __name__ == "__main__":
    main()
