"""Generate the twelve output-free Chapter 6 optimizer notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


CHAPTER_DIR = Path(__file__).resolve().parent


NOTEBOOKS = {
    "quickstart-ai-detector.ipynb": (
        "Unoptimized baseline",
        "Runs the unchanged ChainOfThought detector and freezes its predictions, cost, and latency.",
        "quickstart",
    ),
    "labeled-few-shot.ipynb": (
        "LabeledFewShot",
        "Adds sampled labeled demonstrations without teacher-generated traces.",
        "labeled-few-shot",
    ),
    "bootstrap-few-shot.ipynb": (
        "BootstrapFewShot",
        "Bootstraps successful reasoning traces and attaches them as demonstrations.",
        "bootstrap-few-shot",
    ),
    "bootstrap-random-search.ipynb": (
        "BootstrapFewShotWithRandomSearch",
        "Searches several bootstrapped demonstration sets against the frozen validation split.",
        "bootstrap-random-search",
    ),
    "knn-few-shot.ipynb": (
        "KNNFewShot",
        "Retrieves nearby labeled examples with a deterministic local hashed n-gram vectorizer.",
        "knn-few-shot",
    ),
    "copro.ipynb": (
        "COPRO",
        "Uses Sol to propose instruction variants, then selects them with the fixed metric.",
        "copro",
    ),
    "miprov2.ipynb": (
        "MIPROv2",
        "Jointly searches instructions and demonstrations with Luna as task model and Sol as proposer.",
        "miprov2",
    ),
    "gepa.ipynb": (
        "GEPA",
        "Evolves instructions from textual failure feedback and preserves its reflection logs.",
        "gepa",
    ),
    "simba.ipynb": (
        "SIMBA",
        "Samples trajectories and adds reflective rules or demonstrations for difficult examples.",
        "simba",
    ),
    "ensemble.ipynb": (
        "Ensemble",
        "Combines labeled, bootstrapped, and searched programs with deterministic majority voting.",
        "ensemble",
    ),
    "bootstrap-finetune.ipynb": (
        "BootstrapFinetune (CUDA)",
        "Weight optimization requires an NVIDIA CUDA host; unsupported hosts emit a hardware-blocked artifact.",
        "bootstrap-finetune",
    ),
    "better-together.ipynb": (
        "BetterTogether (CUDA)",
        "Alternating prompt and weight optimization requires the same NVIDIA CUDA training stack.",
        "better-together",
    ),
}


def _source(value: str) -> list[str]:
    return dedent(value).strip("\n").splitlines(keepends=True)


def _markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


def _code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(value),
    }


def make_notebook(title: str, description: str, optimizer: str) -> dict:
    return {
        "cells": [
            _markdown(
                f"""
                # {title}

                {description}

                This notebook uses the pair-grouped, provenance-bearing Chapter 6 dataset.
                Set `CHAPTER06_PROFILE=smoke` for the bounded code-path check or
                `CHAPTER06_PROFILE=full` for the frozen benchmark. Every run writes the
                serialized program, final instruction and demos, complete console output,
                sanitized LM history, per-example predictions, cost, optimization time,
                and latency to `chapter06/results/runs/`.
                """
            ),
            _code(
                """
                %pip install -r ../requirements.txt -q
                """
            ),
            _code(
                f"""
                import os
                import sys
                from pathlib import Path

                cwd = Path.cwd().resolve()
                REPO_ROOT = cwd if (cwd / "chapter06").exists() else cwd.parent
                if str(REPO_ROOT) not in sys.path:
                    sys.path.insert(0, str(REPO_ROOT))

                from chapter06.optimizer_experiment import run_experiment

                OPTIMIZER = {optimizer!r}
                PROFILE = os.getenv("CHAPTER06_PROFILE", "smoke")
                print({{"optimizer": OPTIMIZER, "profile": PROFILE}})
                """
            ),
            _code(
                """
                result = run_experiment(OPTIMIZER, profile_name=PROFILE)
                result
                """
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for filename, (title, description, optimizer) in NOTEBOOKS.items():
        path = CHAPTER_DIR / filename
        path.write_text(
            json.dumps(make_notebook(title, description, optimizer), indent=1) + "\n",
            encoding="utf-8",
        )
    print(f"Generated {len(NOTEBOOKS)} Chapter 6 notebooks.")


if __name__ == "__main__":
    main()
