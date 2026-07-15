"""Generate the concise, publication-ready Chapter 6 optimizer notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any


CHAPTER_DIR = Path(__file__).resolve().parent


NOTEBOOKS: dict[str, dict[str, Any]] = {
    "quickstart-ai-detector.ipynb": {
        "title": "Unoptimized baseline",
        "optimizer": "quickstart",
        "idea": "The control condition: run the original ChainOfThought program without changing its prompt, demonstrations, or weights.",
        "use_when": "Always measure this first. It tells you whether optimization beats the student you would otherwise deploy.",
        "changes": "Nothing. `compile` is intentionally skipped, so optimization time and cost are zero.",
        "config": [
            "same task model and frozen test split as every optimized run",
            "uncached calls so latency and spend remain visible",
        ],
        "compile": "optimized_detector = detector  # no optimizer and no compile step",
        "reading": "The baseline score is the denominator for uplift. Its contemporaneous predictions are also stored in every optimizer run because uncached model behavior can vary.",
    },
    "labeled-few-shot.ipynb": {
        "title": "LabeledFewShot",
        "optimizer": "labeled-few-shot",
        "idea": "Sample labeled training examples and attach them directly as demonstrations; no teacher calls are needed during compilation.",
        "use_when": "You have trustworthy labels, want the cheapest few-shot baseline, and do not need generated reasoning traces.",
        "changes": "Demonstrations only; the original instruction remains unchanged.",
        "config": [
            "`k=4` caps prompt growth",
            "`sample=True` samples from the frozen training split",
        ],
        "compile": "optimizer = dspy.LabeledFewShot(k=4)\noptimized_detector = optimizer.compile(detector, trainset=trainset, sample=True)",
        "reading": "Inspect the four saved demos. This method can improve in-context calibration, but sampled examples are not selected against the validation set.",
    },
    "bootstrap-few-shot.ipynb": {
        "title": "BootstrapFewShot",
        "optimizer": "bootstrap-few-shot",
        "idea": "Run a teacher over training examples, keep successful traces under the metric, and use them as demonstrations.",
        "use_when": "Labels exist but worked reasoning traces may teach the student more than labels alone.",
        "changes": "Adds up to two bootstrapped and two labeled demonstrations; it does not search instruction text.",
        "config": [
            "`max_bootstrapped_demos=2`",
            "`max_labeled_demos=2`",
            "one bootstrap round and one tolerated error",
        ],
        "compile": "optimizer = dspy.BootstrapFewShot(\n    metric=exact_match, max_bootstrapped_demos=2,\n    max_labeled_demos=2, max_rounds=1, max_errors=1,\n)\noptimized_detector = optimizer.compile(detector, trainset=trainset)",
        "reading": "Compare its demos with LabeledFewShot: bootstrapped examples include model-produced reasoning that passed the exact-match metric.",
    },
    "bootstrap-random-search.ipynb": {
        "title": "BootstrapFewShotWithRandomSearch (BootstrapRS)",
        "optimizer": "bootstrap-random-search",
        "idea": "Build several bootstrapped demo sets and choose among them using the frozen validation split.",
        "use_when": "BootstrapFewShot is promising and you can afford multiple candidate programs to reduce dependence on one demo sample.",
        "changes": "Demonstrations only, but candidate selection adds a validation-driven search loop.",
        "config": [
            "8 candidate programs in the full profile (2 in smoke)",
            "two bootstrapped plus two labeled demos per candidate",
            "single evaluation thread by default",
        ],
        "compile": "optimizer = dspy.BootstrapFewShotWithRandomSearch(\n    metric=exact_match, num_candidate_programs=profile.bootstrap_candidates,\n    max_bootstrapped_demos=2, max_labeled_demos=2, num_threads=1,\n)\noptimized_detector = optimizer.compile(detector, trainset=trainset, valset=valset)",
        "reading": "The extra compile spend buys candidate selection, not a new instruction. Check whether the held-out gain justifies the search relative to plain BootstrapFewShot.",
    },
    "knn-few-shot.ipynb": {
        "title": "KNNFewShot",
        "optimizer": "knn-few-shot",
        "idea": "Retrieve examples near each input and compile a local few-shot program at inference time.",
        "use_when": "Different inputs benefit from different demonstrations and a deterministic local retriever is available.",
        "changes": "The instruction stays fixed; the demonstration context is selected by similarity for each input.",
        "config": [
            "`k=4` nearest training examples",
            "deterministic 512-dimensional hashed unigram/bigram vectors",
            "no paid embedding API or bootstrapped demos",
        ],
        "compile": "vectorizer = dspy.Embedder(hashed_ngram_embeddings)\noptimizer = dspy.KNNFewShot(\n    k=4, trainset=trainset, vectorizer=vectorizer, metric=exact_match,\n    max_bootstrapped_demos=0, max_labeled_demos=4,\n)\noptimized_detector = optimizer.compile(detector)  # trainset belongs in the constructor",
        "reading": "Compile cost is nearly zero, but retrieval work moves into inference. DSPy 3.x takes `trainset` and `vectorizer` in the constructor and only the student in `compile`.",
    },
    "copro.ipynb": {
        "title": "COPRO",
        "optimizer": "copro",
        "idea": "Propose instruction variants, evaluate them, and iteratively refine the best candidates.",
        "use_when": "The instruction is likely the bottleneck and you want a direct, interpretable prompt search without demo search.",
        "changes": "Instruction text only; the selected program has no demonstrations in this benchmark.",
        "config": [
            "full-profile breadth 4 and depth 2",
            "GPT-5.6 Sol proposes; Luna executes",
            "validation uses the exact-match metric",
        ],
        "compile": "optimizer = dspy.COPRO(\n    prompt_model=reflection_lm, metric=exact_match,\n    breadth=profile.copro_breadth, depth=profile.copro_depth, track_stats=True,\n)\noptimized_detector = optimizer.compile(detector, trainset=trainset)",
        "reading": "Read the learned instruction before the score. COPRO's artifact is especially auditable because its gain must come from wording rather than hidden examples.",
    },
    "miprov2.ipynb": {
        "title": "MIPROv2",
        "optimizer": "miprov2",
        "idea": "Jointly propose instructions and demonstrations, then use validation feedback to search their combinations.",
        "use_when": "Both prompt wording and examples may matter and you can spend more compile calls for a joint search.",
        "changes": "Instruction plus up to two bootstrapped and two labeled demonstrations.",
        "config": [
            "`auto='light'` search budget",
            "seed 42 with minibatched validation",
            "Sol proposes prompts; Luna runs the task",
        ],
        "compile": "optimizer = dspy.MIPROv2(\n    metric=exact_match, prompt_model=reflection_lm, task_model=task_lm,\n    auto='light', max_bootstrapped_demos=2, max_labeled_demos=2, seed=42,\n)\noptimized_detector = optimizer.compile(\n    detector, trainset=trainset, valset=valset, minibatch=True,\n    requires_permission_to_run=False,\n)",
        "reading": "Inspect instruction and demos together: MIPROv2 optimizes their combination, so attributing the result to either one alone would be misleading.",
    },
    "gepa.ipynb": {
        "title": "GEPA",
        "optimizer": "gepa",
        "idea": "Use textual failure feedback to evolve instructions, preserving reflection logs and best outputs along the way.",
        "use_when": "Your metric can explain errors, not merely score them, and you want a detailed instruction that encodes those lessons.",
        "changes": "In this rerun GEPA learned a long standalone instruction and no final demonstrations.",
        "config": [
            "six full evaluations in the full profile",
            "reflection minibatches of three",
            "feedback metric returns both score and diagnosis",
            "seed 42; merge disabled for a bounded run",
        ],
        "compile": "optimizer = dspy.GEPA(\n    metric=feedback_metric, max_full_evals=profile.gepa_max_full_evals,\n    reflection_minibatch_size=3, reflection_lm=reflection_lm,\n    use_merge=False, track_best_outputs=True, seed=42,\n)\noptimized_detector = optimizer.compile(detector, trainset=trainset, valset=valset)",
        "reading": "The preview is deliberately truncated; follow the prompt path for the complete learned rule set and `optimizer_logs/` for the evolutionary trace.",
    },
    "simba.ipynb": {
        "title": "SIMBA",
        "optimizer": "simba",
        "idea": "Sample trajectories, identify difficult examples, and add reflective rules or demonstrations in a sequence of improvement steps.",
        "use_when": "You want iterative, example-driven improvement and can tolerate a relatively expensive reflective search.",
        "changes": "The selected program may add rules or demonstrations; this rerun's artifact shows which mechanism won.",
        "config": [
            "six steps and four candidates in the full profile",
            "batch size capped at eight",
            "at most two demonstrations",
            "seed 42",
        ],
        "compile": "optimizer = dspy.SIMBA(\n    metric=exact_match, bsize=8, num_candidates=profile.simba_candidates,\n    max_steps=profile.simba_steps, max_demos=2, prompt_model=reflection_lm,\n)\noptimized_detector = optimizer.compile(detector, trainset=trainset, seed=42)",
        "reading": "Compare the final training trajectory with the locked test score. A large gap is evidence that reflective search can still overfit a small benchmark.",
    },
    "ensemble.ipynb": {
        "title": "Ensemble",
        "optimizer": "ensemble",
        "idea": "Compile three different few-shot programs and reduce their boolean predictions with majority voting.",
        "use_when": "Errors are costly enough to justify several model calls per prediction and component diversity improves robustness.",
        "changes": "Combines LabeledFewShot, BootstrapFewShot, and BootstrapRS; it does not learn a single new prompt.",
        "config": [
            "fixed pool of three component programs",
            "typed-boolean majority reducer",
            "all components execute for every test example",
        ],
        "compile": "programs = [labeled_program, bootstrapped_program, searched_program]\noptimizer = dspy.Ensemble(reduce_fn=boolean_majority)\noptimized_detector = optimizer.compile(programs)",
        "reading": "Accuracy is only half the result: compare mean and p95 inference latency with single-program optimizers because every prediction fans out to three calls.",
    },
    "bootstrap-finetune.ipynb": {
        "title": "BootstrapFinetune (Apple Silicon / MPS)",
        "optimizer": "bootstrap-finetune",
        "idea": "Bootstrap successful traces into training data, then update model weights rather than only prompt state.",
        "use_when": "You control a trainable model and want to distill accepted DSPy traces into a reusable local adapter.",
        "changes": "A PEFT LoRA adapter for Qwen2.5-0.5B-Instruct; the prompt remains separately inspectable.",
        "config": [
            "stock DSPy BootstrapFinetune with a Transformers/TRL provider boundary",
            "MPS selected on Apple Silicon; CPU remains an explicit fallback",
            "18 full-profile training steps, batch size 1, LoRA rank 8, seed 42",
            "local self-teaching by default, so this row makes no OpenAI calls",
        ],
        "compile": "optimizer = dspy.BootstrapFinetune(\n    metric=exact_match, train_kwargs=training_config,\n    exclude_demos=True, num_threads=1,\n)\noptimized_detector = optimizer.compile(\n    detector, trainset=trainset, teacher=local_teacher,\n)",
        "reading": "Read score and adapter evidence together. This Qwen/MPS result uses the same frozen split but is not an absolute Luna-to-Luna comparison; follow the training path for device, loss, and adapter metadata.",
    },
    "better-together.ipynb": {
        "title": "BetterTogether (Apple Silicon / MPS)",
        "optimizer": "better-together",
        "idea": "Alternate prompt optimization and weight optimization, evaluate intermediate programs, and retain the best strategy stage.",
        "use_when": "You have both a useful prompt optimizer and a trainable local model, and want them to improve each other rather than run in isolation.",
        "changes": "Prompt demonstrations first, then a Qwen LoRA adapter in the explicit `p -> w` candidate.",
        "config": [
            "BootstrapFewShotWithRandomSearch (`p`) plus BootstrapFinetune (`w`)",
            "stock DSPy BetterTogether with explicit `p -> w` strategy",
            "MPS-backed Qwen2.5-0.5B-Instruct, 18 weight steps, seed 42",
            "preserve original, `p`, and `p -> w` candidates even when validation ties",
        ],
        "compile": "optimizer = dspy.BetterTogether(\n    metric=exact_match, p=prompt_optimizer, w=weight_optimizer,\n)\noptimized_detector = optimizer.compile(\n    detector, trainset=trainset, teacher=local_teacher, valset=valset,\n    strategy='p -> w', seed=42,\n)",
        "reading": "DSPy retained the original program when all validation candidates tied. That is not a failed run: inspect `candidate_programs/` to see the completed prompt-only and trained `p -> w` alternatives and the adapter that was deliberately preserved.",
    },
}


def _source(value: str) -> list[str]:
    return dedent(value).strip("\n").splitlines(keepends=True)


def _markdown(value: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


def _code(value: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(value),
    }


def _with_cell_ids(notebook: dict[str, Any]) -> dict[str, Any]:
    for index, cell in enumerate(notebook["cells"], start=1):
        cell["id"] = f"chapter06-cell-{index:02d}"
    return notebook


def make_notebook(spec: dict[str, Any]) -> dict[str, Any]:
    config = "\n".join(f"- {item}" for item in spec["config"])
    return _with_cell_ids(
        {
            "cells": [
                _markdown(
                    f"""
                # {spec["title"]}

                {spec["idea"]}

                **Use it when:** {spec["use_when"]}

                **What compilation changes:** {spec["changes"]}

                Important configuration in this benchmark:

                {config}

                The 74-row dataset and pair-grouped train/validation/test membership are frozen.
                The test partition is deliberately baseline-adversarial, so these scores teach
                optimizer tradeoffs; they are not a general-purpose AI-detector leaderboard.
                """
                ),
                _code(
                    f"""
                import sys
                from pathlib import Path

                cwd = Path.cwd().resolve()
                REPO_ROOT = cwd if (cwd / "chapter06").is_dir() else cwd.parent
                if not (REPO_ROOT / "chapter06" / "results" / "benchmark_summary.json").exists():
                    raise RuntimeError("Run this notebook from the repository or chapter06 directory.")
                if str(REPO_ROOT) not in sys.path:
                    sys.path.insert(0, str(REPO_ROOT))

                from chapter06.notebook_support import (
                    artifact_paths,
                    benchmark_snapshot,
                    learned_program_preview,
                    verify_prompt_artifact,
                )

                OPTIMIZER = {spec["optimizer"]!r}
                print(f"optimizer={{OPTIMIZER!r}}")
                print("reading the checked-in chapter result; no API calls")
                """
                ),
                _markdown(
                    f"""
                ## Compile shape

                This is the essential DSPy call used by the shared runner (setup variables omitted):

                ```python
                {spec["compile"]}
                ```

                `compile` returns a program. Calling that program on the untouched test examples is
                a separate phase; the notebook reports optimization cost/time separately from inference latency.
                """
                ),
                _code(
                    """
                print(benchmark_snapshot(OPTIMIZER))
                print()
                print(artifact_paths(OPTIMIZER))
                """
                ),
                _markdown(
                    f"""
                ## Read the result

                {spec["reading"]}

                The next cell shows a bounded readable preview. The complete, lossless prompt and
                saved program snapshot remain at the paths printed above.
                """
                ),
                _code(
                    """
                print(learned_program_preview(OPTIMIZER))
                print()
                print("Frozen program/prompt consistency:", verify_prompt_artifact(OPTIMIZER))
                """
                ),
                _markdown(
                    """
                ## Apply the pattern

                Adapt the compile shape above to your own DSPy program, metric, and frozen
                train/validation split. Evaluate the returned program on a test set that was not
                used during compilation, and compare accuracy, compile cost, and inference latency
                rather than treating a single score as the whole result.

                The complete Chapter 6 rerun is summarized in `CHAPTER_RESULTS.md`. Raw provider
                transcripts and temporary training outputs are intentionally excluded from the
                student download.
                """
                ),
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


def main() -> None:
    for filename, spec in NOTEBOOKS.items():
        (CHAPTER_DIR / filename).write_text(
            json.dumps(make_notebook(spec), indent=1) + "\n", encoding="utf-8"
        )
    print(
        f"Generated {len(NOTEBOOKS)} educational Chapter 6 notebooks; execute before publishing."
    )


if __name__ == "__main__":
    main()
