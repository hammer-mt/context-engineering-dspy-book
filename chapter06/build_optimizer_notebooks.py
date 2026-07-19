"""Generate the concise, publication-ready Chapter 6 optimizer notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent, indent
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
        "changes": "Instruction text only; inspect the saved expanded-dataset prompt to see the selected wording.",
        "config": [
            "full-profile breadth 4 and depth 2",
            "GPT-5.6 Sol proposes; Luna executes",
            "validation uses the exact-match metric",
        ],
        "compile": "optimizer = dspy.COPRO(\n    prompt_model=reflection_lm, metric=exact_match,\n    breadth=profile.copro_breadth, depth=profile.copro_depth,\n    init_temperature=1.0, track_stats=True,\n)\noptimized_detector = optimizer.compile(\n    detector, trainset=valset,\n    eval_kwargs={\"num_threads\": 1, \"display_table\": False},\n)",
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
        "changes": "Instruction text and, depending on the search path, demonstrations; inspect the frozen reference artifact.",
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
        "changes": "The selected program may add rules or demonstrations; the saved artifact shows which mechanism won.",
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
            "DSPy BootstrapFinetune and native LocalProvider with a balanced-trace validation guard",
            "thin MacLocalProvider subclass selects Transformers/MPS serving and adapts the pinned TRL argument name",
            "10 epochs, batch size 1, gradient accumulation 4, PEFT LoRA, learning rate 2e-4",
            "GPT-5.6 Sol teacher; Qwen2.5-0.5B student trained locally",
        ],
        "compile": "optimizer = BalancedBootstrapFinetune(\n    metric=exact_match, train_kwargs=training_config,\n    exclude_demos=True, num_threads=1, min_examples_per_class=2,\n)\noptimized_detector = optimizer.compile(\n    detector, trainset=trainset, teacher=sol_teacher,\n)",
        "platform_note": """
        ## LocalProvider on Apple Silicon and with standard SGLang

        DSPy 3.2.1's `LocalProvider` has two separate jobs. Its **training** path
        already uses Transformers and TRL and selects CUDA, then MPS, then CPU.
        We keep that path: DSPy formats the accepted traces, masks non-assistant
        tokens, constructs the PEFT trainer, trains, and saves the merged model.

        Its standard **serving** path starts `python -m sglang.launch_server`.
        That is the path to use on an SGLang-compatible CUDA system:

        ```python
        from dspy.clients.lm_local import LocalProvider

        student_lm = dspy.LM(
            "openai/local:Qwen/Qwen2.5-0.5B-Instruct",
            provider=LocalProvider(),
            cache=False,
            max_tokens=96,
        )
        ```

        Apple Silicon needs a different serving backend. This chapter's
        `MacLocalProvider(LocalProvider)` overrides only `launch` and `kill` so
        the model is loaded by Transformers on MPS. Its `finetune` method calls
        `LocalProvider.finetune` unchanged except for translating DSPy 3.2.1's
        `max_seq_length` keyword to the `max_length` name used by the pinned TRL
        0.24.0. There is no replacement training loop. The complete small shim is
        in `chapter06/apple_finetune.py`; the optimizer call below is identical on
        both platforms.
        """,
        "reading": "Check the accepted human/AI trace counts before the score. A balanced accepted set is a prerequisite for interpreting this fine-tune as a classification experiment.",
    },
    "better-together.ipynb": {
        "title": "BetterTogether (Apple Silicon / MPS)",
        "optimizer": "better-together",
        "idea": "Alternate prompt optimization and weight optimization, evaluate intermediate programs, and retain the best strategy stage.",
        "use_when": "You have both a useful prompt optimizer and a trainable local model, and want them to improve each other rather than run in isolation.",
        "changes": "Prompt demonstrations first, then a Qwen LoRA adapter in the explicit `p -> w` candidate.",
        "config": [
            "BootstrapFewShotWithRandomSearch (`p`) plus BootstrapFinetune (`w`)",
            "DSPy BetterTogether with explicit `p -> w` strategy",
            "DSPy LocalProvider with MPS-backed Qwen2.5-0.5B-Instruct, 10 weight-training epochs",
            "preserve original, `p`, and `p -> w` candidates even when validation ties",
        ],
        "compile": "optimizer = dspy.BetterTogether(\n    metric=exact_match, p=prompt_optimizer, w=weight_optimizer,\n)\noptimized_detector = optimizer.compile(\n    detector, trainset=trainset, teacher=sol_teacher, valset=valset,\n    strategy='p -> w', seed=42,\n    optimizer_compile_args={'p': {'teacher': sol_teacher}},\n)",
        "reading": "Read the validation-selected stage alongside the same-model baseline. If candidates tie or regress, retaining an earlier stage is an honest optimizer outcome, not a reason to consult the locked test.",
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
    config = indent(
        "\n".join(f"- {item}" for item in spec["config"]),
        " " * 16,
    )
    platform_note = spec.get("platform_note", "")
    if platform_note:
        platform_note = indent(dedent(platform_note).strip(), " " * 16)
    compile_shape = indent(spec["compile"], " " * 16)
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

                Every notebook loads the canonical 300-row expanded dataset and frozen,
                pair-grouped membership: 160 training, 60 validation, and 80 locked-test rows.
                A semantic human/AI pair can never cross partitions. Optimizer choices use
                validation only; the locked test is released once after the program is frozen.
                These scores teach optimizer tradeoffs, not a general AI-detector leaderboard.
                """
                ),
                _code(
                    f"""
                import os
                import sys
                from pathlib import Path

                cwd = Path.cwd().resolve()
                REPO_ROOT = cwd if (cwd / "chapter06").is_dir() else cwd.parent
                if not (REPO_ROOT / "chapter06" / "results" / "expanded_notebooks" / "comparison.json").exists():
                    raise RuntimeError("Run this notebook from the repository or chapter06 directory.")
                if str(REPO_ROOT) not in sys.path:
                    sys.path.insert(0, str(REPO_ROOT))

                from chapter06.notebook_support import artifact_paths, learned_program_preview, verify_prompt_artifact
                from chapter06.optimizer_runtime import (
                    format_result,
                    load_frozen_examples,
                    published_result,
                    run_optimizer,
                    split_summary,
                )

                OPTIMIZER = {spec["optimizer"]!r}
                splits = load_frozen_examples()
                RUN_LIVE = os.getenv("CHAPTER06_RUN_LIVE", "0") == "1"
                print(f"optimizer={{OPTIMIZER!r}}; live={{RUN_LIVE}}")
                print(split_summary(splits))
                """
                ),
                _markdown(
                    f"""
{platform_note}

                ## Compile shape

                This is the essential DSPy call used by the shared executable runner:

                ```python
{compile_shape}
                ```

                `compile` returns a program. The shared runner then evaluates that program on the
                untouched 80-row locked test split. The baseline has its own notebook; all other
                notebooks report validation and locked-test accuracy separately.
                """
                ),
                _code(
                    """
                if RUN_LIVE:
                    live_run = run_optimizer(
                        OPTIMIZER,
                        splits=splits,
                    )
                    result = live_run.summary()
                else:
                    result = published_result(OPTIMIZER)

                print(format_result(result))
                print()
                print(artifact_paths(OPTIMIZER))
                """
                ),
                _markdown(
                    f"""
                ## Read the result

                {spec["reading"]}

                The saved output above uses the checked-in expanded-dataset result, so opening or
                rerunning the notebook is free. The paid run first passed a bounded smoke profile,
                then froze its full program using training and validation only. Set
                `CHAPTER06_RUN_LIVE=1` before launching Jupyter to reproduce that full protocol;
                prompt optimizers require an OpenAI key, while weight optimizers also require the
                local PyTorch/Transformers stack. The next cell previews the durable program artifact.
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
                train/validation split. Keep the test set untouched until the optimizer returns,
                then report final test accuracy as `correct / test examples` so every optimizer is easy
                to compare. Use the separate baseline notebook when you also need uplift.

                The complete Chapter 6 rerun is summarized in `CHAPTER_RESULTS.md`; machine-readable
                scores, prompts, programs, predictions, timing, cost, versions, hashes, failures,
                and retries live under `results/expanded_notebooks/`. Weight-model payloads are
                generated locally and Git-ignored; their checked-in manifests retain file hashes,
                sizes, configuration, prompts, programs, and scores. Credentials and provider
                request bodies are intentionally excluded.
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
