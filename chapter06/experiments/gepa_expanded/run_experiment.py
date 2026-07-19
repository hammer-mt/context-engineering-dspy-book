"""Run locked-baseline confirmation and GEPA on the expanded dataset."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import dspy

from .dataset import DATA_PATH, RESULTS_DIR, SPLIT_PATH, append_jsonl, as_bool, assert_lock_unchanged
from .analysis import majority_vote, mcnemar_exact, paired_pair_bootstrap
from .guardrails import (
    BudgetExceeded,
    BudgetLedger,
    atomic_write_json,
    classify_api_error,
    summarize_history,
    write_jsonl,
)
from .runtime import (
    AIDetector,
    REFLECTION_MODEL,
    TASK_MODEL,
    assert_stage_budget,
    feedback_metric,
    finish_stage,
    load_project_env,
    make_lm,
    new_run_id,
    predict_record,
)


def load_examples() -> dict[str, list[dspy.Example]]:
    assert_lock_unchanged()
    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_pair: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row)
    result: dict[str, list[dspy.Example]] = {}
    seen: set[str] = set()
    for split_name in ("train", "validation", "test"):
        examples: list[dspy.Example] = []
        for pair_id in manifest["splits"][split_name]:
            if pair_id in seen:
                raise ValueError(f"pair {pair_id} appears in more than one split")
            seen.add(pair_id)
            pair = by_pair.get(pair_id, [])
            if len(pair) != 2:
                raise ValueError(f"pair {pair_id} is incomplete")
            for row in sorted(pair, key=lambda item: as_bool(item["is_ai"])):
                examples.append(
                    dspy.Example(
                        pair_id=row["pair_id"],
                        example_id=row["example_id"],
                        text=row["text"],
                        is_ai=as_bool(row["is_ai"]),
                    ).with_inputs("text")
                )
        result[split_name] = examples
    if seen != set(by_pair):
        raise ValueError("split manifest does not cover every pair")
    return result


def evaluate_parallel(
    program: dspy.Module,
    examples: Sequence[dspy.Example],
    *,
    output_path: Path,
    threads: int = 4,
    ledger: BudgetLedger | None = None,
    lm: dspy.LM | None = None,
    stage_cap_usd: float = 5.0,
) -> dict[str, Any]:
    if threads < 1:
        raise ValueError("threads must be positive")
    records: list[dict[str, Any]] = []
    for start in range(0, len(examples), threads):
        if ledger is not None and lm is not None:
            assert_stage_budget(ledger, lm, stage_cap=stage_cap_usd)
        batch = examples[start : start + threads]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(
                    predict_record,
                    program,
                    text=str(example.text),
                    expected=bool(example.is_ai),
                    identity={
                        "pair_id": str(example.pair_id),
                        "example_id": str(example.example_id),
                    },
                )
                for example in batch
            ]
            batch_records = [future.result() for future in as_completed(futures)]
        for record in batch_records:
            append_jsonl(output_path, record)
            if record["status"] != "completed":
                raise RuntimeError(record.get("error", record["status"]))
            records.append(record)
    correct = sum(int(record["correct"]) for record in records)
    latencies = sorted(float(record["latency_seconds"]) for record in records)
    p95_index = max(0, min(len(latencies) - 1, int(0.95 * len(latencies)) - 1))
    return {
        "row_count": len(records),
        "correct": correct,
        "accuracy_pct": 100 * correct / len(records),
        "mean_latency_seconds": sum(latencies) / len(latencies),
        "p95_latency_seconds": latencies[p95_index],
        "predictions": records,
    }


class _Tee(io.TextIOBase):
    """Mirror optimizer output to the terminal and a durable artifact."""

    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _combined_history_summary(*lms: dspy.LM) -> dict[str, Any]:
    histories = [entry for lm in lms for entry in lm.history]
    return summarize_history(histories)


def _assert_compile_budget(
    ledger: BudgetLedger,
    task_lm: dspy.LM,
    reflection_lm: dspy.LM,
    *,
    stage_cap_usd: float,
) -> None:
    spent = float(_combined_history_summary(task_lm, reflection_lm)["cost_usd"])
    if spent >= stage_cap_usd:
        raise BudgetExceeded(
            f"GEPA stage cost ${spent:.4f} reached its ${stage_cap_usd:.2f} cap"
        )
    ledger.assert_can_spend(spent + min(0.10, stage_cap_usd - spent))


def _prompt_artifact(program: dspy.Module) -> dict[str, Any]:
    predictors: dict[str, Any] = {}
    for name, predictor in program.named_predictors():
        signature = predictor.signature
        predictors[name] = {
            "instructions": signature.instructions,
            "signature": str(signature),
            "fields": [
                {
                    "name": field_name,
                    "prefix": field.json_schema_extra.get("prefix", ""),
                    "description": field.json_schema_extra.get("desc", ""),
                }
                for field_name, field in signature.fields.items()
            ],
            "demos": [demo.toDict() for demo in predictor.demos],
        }
    return {
        "task_model": TASK_MODEL,
        "program_class": type(program).__name__,
        "predictors": predictors,
    }


def _save_program_artifacts(program: dspy.Module, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    program.save(output_dir / "optimized_program.json")
    prompt = _prompt_artifact(program)
    atomic_write_json(output_dir / "learned_prompt.json", prompt)
    instructions = [
        f"[{name}]\n{details['instructions']}"
        for name, details in prompt["predictors"].items()
    ]
    (output_dir / "learned_instruction.txt").write_text(
        "\n\n".join(instructions) + "\n", encoding="utf-8"
    )


def _gepa_details_artifact(details: Any) -> dict[str, Any]:
    """Serialize DSPy's detailed result without relying on its module-unsafe to_dict."""

    candidates = []
    for candidate in details.candidates:
        candidates.append(
            {
                name: predictor.signature.instructions
                for name, predictor in candidate.named_predictors()
            }
        )
    return {
        "candidates": candidates,
        "parents": details.parents,
        "val_aggregate_scores": details.val_aggregate_scores,
        "val_subscores": details.val_subscores,
        "per_val_instance_best_candidates": [
            sorted(indices) if isinstance(indices, (set, list, tuple)) else [int(indices)]
            for indices in details.per_val_instance_best_candidates
        ],
        "discovery_eval_counts": details.discovery_eval_counts,
        "best_outputs_valset": details.best_outputs_valset,
        "total_metric_calls": details.total_metric_calls,
        "num_full_val_evals": details.num_full_val_evals,
        "log_dir": details.log_dir,
        "seed": details.seed,
        "best_idx": details.best_idx,
    }


def _load_program(program_path: Path, lm: dspy.LM) -> AIDetector:
    program = AIDetector()
    program.load(program_path)
    program.set_lm(lm)
    return program


def materialize_gepa_candidates(
    *, optimizer_results_path: Path, output_dir: Path, candidate_indexes: Sequence[int]
) -> dict[str, Any]:
    """Turn saved GEPA prompt candidates into reloadable programs without API calls."""

    details = json.loads(optimizer_results_path.read_text(encoding="utf-8"))
    candidates = details["candidates"]
    scores = details["val_aggregate_scores"]
    selected = list(candidate_indexes) if candidate_indexes else list(range(len(candidates)))
    if len(set(selected)) != len(selected):
        raise ValueError("candidate indexes must be unique")
    materialized: list[dict[str, Any]] = []
    for index in selected:
        if index < 0 or index >= len(candidates):
            raise IndexError(f"candidate index {index} is out of range")
        program = AIDetector()
        predictors = dict(program.named_predictors())
        for predictor_name, instructions in candidates[index].items():
            if predictor_name not in predictors:
                raise ValueError(f"unknown predictor {predictor_name!r} in candidate {index}")
            predictor = predictors[predictor_name]
            predictor.signature = predictor.signature.with_instructions(str(instructions))
        candidate_dir = output_dir / f"candidate-{index}"
        _save_program_artifacts(program, candidate_dir)
        candidate_summary = {
            "candidate_index": index,
            "gepa_internal_validation_accuracy_pct": 100 * float(scores[index]),
            "program_path": str(candidate_dir / "optimized_program.json"),
            "learned_prompt_path": str(candidate_dir / "learned_prompt.json"),
            "source_optimizer_results": str(optimizer_results_path),
            "test_rows_seen": 0,
        }
        atomic_write_json(candidate_dir / "summary.json", candidate_summary)
        materialized.append(candidate_summary)
    summary = {
        "source_optimizer_results": str(optimizer_results_path),
        "candidate_count": len(materialized),
        "candidate_indexes": selected,
        "test_rows_seen": 0,
        "candidates": materialized,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    assert_lock_unchanged()
    return summary


def run_gepa(
    *,
    mode: str,
    output_dir: Path,
    max_full_evals: int,
    stage_cap_usd: float,
    threads: int = 4,
    seed_program_path: Path | None = None,
    candidate_selection_strategy: str = "pareto",
) -> dict[str, Any]:
    """Compile a smoke or full GEPA candidate without touching the test split."""

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    if max_full_evals < 1:
        raise ValueError("max_full_evals must be positive")
    if (output_dir / "optimized_program.json").exists():
        raise FileExistsError(f"refusing to overwrite completed candidate at {output_dir}")

    load_project_env()
    splits = load_examples()
    trainset = splits["train"] if mode == "full" else splits["train"][:12]
    valset = splits["validation"] if mode == "full" else splits["validation"][:8]
    # The locked test is deliberately never passed to GEPA.
    test_example_ids = {str(example.example_id) for example in splits["test"]}
    compile_example_ids = {str(example.example_id) for example in trainset + valset}
    if test_example_ids & compile_example_ids:
        raise ValueError("GEPA compile data overlaps the locked test")

    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger()
    ledger.assert_can_spend(stage_cap_usd)
    task_lm = make_lm(TASK_MODEL, cache=False, max_tokens=500)
    reflection_lm = make_lm(REFLECTION_MODEL, cache=False, max_tokens=8_000)
    if candidate_selection_strategy not in {"pareto", "current_best"}:
        raise ValueError("candidate selection strategy must be pareto or current_best")
    detector = AIDetector()
    if seed_program_path is not None:
        detector.load(seed_program_path)
    run_id = new_run_id(f"gepa-{mode}")

    def guarded_feedback_metric(*args: Any, **kwargs: Any):
        _assert_compile_budget(
            ledger,
            task_lm,
            reflection_lm,
            stage_cap_usd=stage_cap_usd,
        )
        return feedback_metric(*args, **kwargs)

    optimizer = dspy.GEPA(
        metric=guarded_feedback_metric,
        max_full_evals=max_full_evals,
        reflection_minibatch_size=3,
        candidate_selection_strategy=candidate_selection_strategy,
        reflection_lm=reflection_lm,
        use_merge=False,
        track_best_outputs=True,
        track_stats=True,
        num_threads=threads,
        seed=42,
        log_dir=str(output_dir / "optimizer_trace"),
        gepa_kwargs={"use_cloudpickle": True},
    )
    started = time.monotonic()
    status = "completed"
    optimized: dspy.Module | None = None
    error_message: str | None = None
    console_path = output_dir / "console.log"
    try:
        with console_path.open("w", encoding="utf-8") as console:
            with contextlib.redirect_stdout(_Tee(sys.stdout, console)), contextlib.redirect_stderr(
                _Tee(sys.stderr, console)
            ):
                # Keep the task LM in DSPy's execution context rather than on the
                # student itself. GEPA deep-copies candidate programs; a predictor-
                # local LM would be copied too, splitting its history and hiding
                # task-call cost from the experiment ledger.
                with dspy.context(lm=task_lm):
                    optimized = optimizer.compile(detector, trainset=trainset, valset=valset)
                successful_task_calls = int(summarize_history(task_lm.history)["request_count"])
                if successful_task_calls < len(valset):
                    raise RuntimeError(
                        "GEPA returned without enough successful task-model calls to evaluate "
                        f"the validation set ({successful_task_calls} < {len(valset)}); refusing "
                        "to treat swallowed model errors as a valid optimizer result"
                    )
    except BaseException as exc:
        status = classify_api_error(exc)
        error_message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration = time.monotonic() - started
        task_summary = summarize_history(task_lm.history)
        reflection_summary = summarize_history(reflection_lm.history)
        combined_summary = _combined_history_summary(task_lm, reflection_lm)
        write_jsonl(output_dir / "task_lm_history.jsonl", task_lm.history)
        write_jsonl(output_dir / "reflection_lm_history.jsonl", reflection_lm.history)
        ledger.record(
            stage=f"gepa_{mode}",
            run_id=run_id,
            cost_usd=float(combined_summary["cost_usd"]),
            metadata={
                "status": status,
                "duration_seconds": duration,
                "max_full_evals": max_full_evals,
                "train_rows": len(trainset),
                "validation_rows": len(valset),
                "test_rows_seen": 0,
                "seed_program_path": str(seed_program_path) if seed_program_path else None,
                "candidate_selection_strategy": candidate_selection_strategy,
                "task_history": task_summary,
                "reflection_history": reflection_summary,
                "error": error_message,
            },
        )
        atomic_write_json(
            output_dir / "run_status.json",
            {
                "run_id": run_id,
                "mode": mode,
                "status": status,
                "duration_seconds": duration,
                "stage_cap_usd": stage_cap_usd,
                "cost": combined_summary,
                "error": error_message,
            },
        )

    if optimized is None:
        raise RuntimeError("GEPA returned no optimized program")
    _save_program_artifacts(optimized, output_dir)
    details = getattr(optimized, "detailed_results", None)
    if details is not None:
        atomic_write_json(output_dir / "optimizer_results.json", _gepa_details_artifact(details))
        best_idx = int(details.best_idx)
        internal_score = float(details.val_aggregate_scores[best_idx])
        candidate_count = len(details.val_aggregate_scores)
        metric_calls = details.total_metric_calls
    else:
        best_idx = 0
        internal_score = 0.0
        candidate_count = 0
        metric_calls = None
    summary = {
        "run_id": run_id,
        "mode": mode,
        "status": "completed",
        "task_model": TASK_MODEL,
        "reflection_model": REFLECTION_MODEL,
        "train_rows": len(trainset),
        "validation_rows": len(valset),
        "test_rows_seen": 0,
        "seed_program_path": str(seed_program_path) if seed_program_path else None,
        "candidate_selection_strategy": candidate_selection_strategy,
        "max_full_evals": max_full_evals,
        "best_candidate_index": best_idx,
        "internal_validation_score": internal_score,
        "candidate_count": candidate_count,
        "metric_calls": metric_calls,
        "optimize_time_seconds": duration,
        "cost": combined_summary,
        "program_path": str(output_dir / "optimized_program.json"),
        "learned_prompt_path": str(output_dir / "learned_prompt.json"),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    assert_lock_unchanged()
    return summary


def evaluate_repeated(
    *,
    program_path: Path,
    split_name: str,
    output_dir: Path,
    repeats: int = 3,
    stage_cap_usd: float = 2.0,
    threads: int = 4,
    validation_summary_path: Path | None = None,
    minimum_validation_accuracy_pct: float = 85.0,
) -> dict[str, Any]:
    """Evaluate a frozen candidate repeatedly; test requires a validation gate."""

    if split_name not in {"validation", "test"}:
        raise ValueError("only validation and test evaluation are supported")
    if repeats < 3 or repeats % 2 == 0:
        raise ValueError("repeats must be an odd number of at least three")
    if split_name == "test":
        if validation_summary_path is None or not validation_summary_path.exists():
            raise ValueError("locked-test evaluation requires a persisted validation summary")
        validation = json.loads(validation_summary_path.read_text(encoding="utf-8"))
        if float(validation["majority_accuracy_pct"]) < minimum_validation_accuracy_pct:
            raise ValueError(
                f"validation majority accuracy {validation['majority_accuracy_pct']:.2f}% "
                f"is below the {minimum_validation_accuracy_pct:.2f}% release gate"
            )
    load_project_env()
    examples = load_examples()[split_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed evaluation at {output_dir}")

    repeat_summaries: list[dict[str, Any]] = []
    repeat_records: list[list[dict[str, Any]]] = []
    for index in range(1, repeats + 1):
        repeat_dir = output_dir / f"repeat-{index}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = repeat_dir / "predictions.jsonl"
        if predictions_path.exists():
            raise FileExistsError(predictions_path)
        ledger = BudgetLedger()
        ledger.assert_can_spend(stage_cap_usd)
        lm = make_lm(TASK_MODEL, cache=False, max_tokens=500)
        program = _load_program(program_path, lm)
        run_id = new_run_id(f"optimized-{split_name}-repeat-{index}")
        status = "completed"
        started = time.monotonic()
        error_message: str | None = None
        try:
            evaluation = evaluate_parallel(
                program,
                examples,
                output_path=predictions_path,
                threads=threads,
                ledger=ledger,
                lm=lm,
                stage_cap_usd=stage_cap_usd,
            )
        except BaseException as exc:
            status = classify_api_error(exc)
            error_message = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration = time.monotonic() - started
            history = summarize_history(lm.history)
            write_jsonl(repeat_dir / "lm_history.jsonl", lm.history)
            ledger.record(
                stage=f"optimized_{split_name}",
                run_id=run_id,
                cost_usd=float(history["cost_usd"]),
                metadata={
                    "status": status,
                    "repeat": index,
                    "duration_seconds": duration,
                    "error": error_message,
                    "summary": history,
                },
            )
        repeat = {
            "repeat": index,
            "run_id": run_id,
            **{key: value for key, value in evaluation.items() if key != "predictions"},
            "duration_seconds": duration,
            "cost": history,
        }
        atomic_write_json(repeat_dir / "summary.json", repeat)
        repeat_summaries.append(repeat)
        repeat_records.append(evaluation["predictions"])

    majority = majority_vote(repeat_records)
    majority_correct = sum(int(record["correct"]) for record in majority)
    write_jsonl(output_dir / "majority_predictions.jsonl", majority)
    summary = {
        "split": split_name,
        "program_path": str(program_path),
        "reference_strategy": "per-example majority vote across fresh uncached runs",
        "repeat_count": repeats,
        "row_count": len(majority),
        "accuracies_pct": [repeat["accuracy_pct"] for repeat in repeat_summaries],
        "majority_correct": majority_correct,
        "majority_accuracy_pct": 100 * majority_correct / len(majority),
        "mean_latency_seconds": sum(
            repeat["mean_latency_seconds"] for repeat in repeat_summaries
        ) / repeats,
        "p95_latency_seconds": max(
            repeat["p95_latency_seconds"] for repeat in repeat_summaries
        ),
        "cost_usd": sum(float(repeat["cost"]["cost_usd"]) for repeat in repeat_summaries),
        "repeats": repeat_summaries,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    assert_lock_unchanged()
    return summary


def finalize_result(
    *, candidate_dir: Path, test_evaluation_dir: Path, final_dir: Path
) -> dict[str, Any]:
    """Create the learner-facing final bundle and paired significance analysis."""

    compile_summary = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    test_summary = json.loads((test_evaluation_dir / "summary.json").read_text(encoding="utf-8"))
    baseline_summary = json.loads(
        (RESULTS_DIR / "baseline_confirmation" / "summary.json").read_text(encoding="utf-8")
    )
    baseline_rows = [
        json.loads(line)
        for line in (RESULTS_DIR / "baseline_confirmation" / "majority_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    optimized_rows = [
        json.loads(line)
        for line in (test_evaluation_dir / "majority_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    mcnemar = mcnemar_exact(baseline_rows, optimized_rows)
    bootstrap = paired_pair_bootstrap(baseline_rows, optimized_rows)
    baseline_accuracy = float(baseline_summary["baseline_test_accuracy_pct"])
    optimized_accuracy = float(test_summary["majority_accuracy_pct"])
    ledger = json.loads((RESULTS_DIR / "budget_ledger.json").read_text(encoding="utf-8"))
    baseline_mean_latency = sum(
        float(repeat["mean_latency_seconds"]) for repeat in baseline_summary["repeats"]
    ) / len(baseline_summary["repeats"])
    baseline_p95_latency = max(
        float(repeat["p95_latency_seconds"]) for repeat in baseline_summary["repeats"]
    )
    baseline_evaluation_cost = sum(
        float(repeat["cost"]["cost_usd"]) for repeat in baseline_summary["repeats"]
    )
    optimize_cost = float(compile_summary["cost"]["cost_usd"])
    summary = {
        "baseline_test_accuracy_pct": baseline_accuracy,
        "optimized_test_accuracy_pct": optimized_accuracy,
        "baseline_test_correct": int(baseline_summary["majority_correct"]),
        "optimized_test_correct": int(test_summary["majority_correct"]),
        "absolute_uplift_pct_points": optimized_accuracy - baseline_accuracy,
        "relative_uplift_pct": 100 * (optimized_accuracy - baseline_accuracy) / baseline_accuracy,
        "paired_mcnemar_p_value": mcnemar["p_value"],
        "paired_mcnemar": mcnemar,
        "paired_bootstrap_ci_low_pct_points": bootstrap["ci_low_pct_points"],
        "paired_bootstrap_ci_high_pct_points": bootstrap["ci_high_pct_points"],
        "paired_bootstrap": bootstrap,
        "optimize_cost_usd": optimize_cost,
        "optimize_prompt_tokens": int(compile_summary["cost"]["prompt_tokens"]),
        "optimize_completion_tokens": int(compile_summary["cost"]["completion_tokens"]),
        "optimize_total_tokens": int(compile_summary["cost"]["total_tokens"]),
        "baseline_evaluation_cost_usd": baseline_evaluation_cost,
        "optimized_evaluation_cost_usd": float(test_summary["cost_usd"]),
        "total_experiment_cost_usd": float(ledger["total_cost_usd"]),
        "optimize_time_seconds": float(compile_summary["optimize_time_seconds"]),
        "baseline_mean_latency_seconds": baseline_mean_latency,
        "baseline_p95_latency_seconds": baseline_p95_latency,
        "optimized_mean_latency_seconds": float(test_summary["mean_latency_seconds"]),
        "optimized_p95_latency_seconds": float(test_summary["p95_latency_seconds"]),
        "task_model": TASK_MODEL,
        "reflection_model": REFLECTION_MODEL,
        "candidate_dir": str(candidate_dir),
        "test_evaluation_dir": str(test_evaluation_dir),
        "reference_strategy": test_summary["reference_strategy"],
    }
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in ("optimized_program.json", "learned_prompt.json", "learned_instruction.txt"):
        shutil.copy2(candidate_dir / name, final_dir / name)
    shutil.copy2(test_evaluation_dir / "majority_predictions.jsonl", final_dir / "test_predictions.jsonl")
    atomic_write_json(final_dir / "summary.json", summary)
    atomic_write_json(final_dir / "statistical_analysis.json", {"mcnemar": mcnemar, "bootstrap": bootstrap})
    table_rows = [
        {
            "program": "Unoptimized Luna",
            "test_accuracy_pct": baseline_accuracy,
            "uplift_pct_points": 0.0,
            "optimization_cost_usd": 0.0,
            "evaluation_cost_usd": baseline_evaluation_cost,
            "optimization_time_seconds": 0.0,
            "mean_inference_latency_seconds": baseline_mean_latency,
            "p95_inference_latency_seconds": baseline_p95_latency,
        },
        {
            "program": "GEPA-optimized Luna",
            "test_accuracy_pct": optimized_accuracy,
            "uplift_pct_points": optimized_accuracy - baseline_accuracy,
            "optimization_cost_usd": optimize_cost,
            "evaluation_cost_usd": float(test_summary["cost_usd"]),
            "optimization_time_seconds": float(compile_summary["optimize_time_seconds"]),
            "mean_inference_latency_seconds": float(test_summary["mean_latency_seconds"]),
            "p95_inference_latency_seconds": float(test_summary["p95_latency_seconds"]),
        },
    ]
    with (final_dir / "chapter_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    atomic_write_json(final_dir / "chapter_table.json", table_rows)
    assert_lock_unchanged()
    return summary


def confirm_baseline(*, repeats: int = 3, stage_cap_usd: float = 3.0) -> dict[str, Any]:
    if repeats < 3:
        raise ValueError("at least three independent repeats are required")
    load_project_env()
    splits = load_examples()
    root = RESULTS_DIR / "baseline_confirmation"
    completed_summaries = sorted(root.glob("repeat-*/summary.json"))
    if len(completed_summaries) >= repeats:
        results = [json.loads(path.read_text(encoding="utf-8")) for path in completed_summaries[:repeats]]
    else:
        results = []
        for repeat_index in range(len(completed_summaries), repeats):
            repeat_dir = root / f"repeat-{repeat_index + 1}"
            repeat_dir.mkdir(parents=True, exist_ok=True)
            ledger = BudgetLedger()
            lm = make_lm(TASK_MODEL, cache=False, max_tokens=500)
            detector = AIDetector()
            detector.set_lm(lm)
            run_id = new_run_id(f"expanded-baseline-confirmation-{repeat_index + 1}")
            status = "completed"
            try:
                evaluation = evaluate_parallel(
                    detector,
                    splits["test"],
                    output_path=repeat_dir / "predictions.jsonl",
                    threads=4,
                    ledger=ledger,
                    lm=lm,
                    stage_cap_usd=stage_cap_usd,
                )
            except BaseException as exc:
                status = classify_api_error(exc)
                raise
            finally:
                cost = finish_stage(
                    ledger=ledger,
                    lm=lm,
                    stage="expanded_baseline_confirmation",
                    run_id=run_id,
                    status=status,
                    artifact_dir=repeat_dir,
                    metadata={"repeat": repeat_index + 1, "test_rows": len(splits["test"])},
                )
            result = {
                "repeat": repeat_index + 1,
                "task_model": TASK_MODEL,
                **{key: value for key, value in evaluation.items() if key != "predictions"},
                "cost": cost,
            }
            atomic_write_json(repeat_dir / "summary.json", result)
            results.append(result)
    accuracies = [float(result["accuracy_pct"]) for result in results]
    prediction_runs = []
    for repeat_index in range(repeats):
        path = root / f"repeat-{repeat_index + 1}" / "predictions.jsonl"
        prediction_runs.append(
            {
                record["example_id"]: record
                for record in (
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
        )
    example_ids = set(prediction_runs[0])
    if any(set(run) != example_ids for run in prediction_runs[1:]):
        raise ValueError("baseline repeats cover different test examples")
    majority_correct = 0
    majority_predictions: list[dict[str, Any]] = []
    for example_id in sorted(example_ids):
        records = [run[example_id] for run in prediction_runs]
        positive_votes = sum(bool(record["predicted_is_ai"]) for record in records)
        predicted = positive_votes > repeats / 2
        expected = bool(records[0]["expected_is_ai"])
        majority_correct += int(predicted == expected)
        majority_predictions.append(
            {
                "pair_id": records[0]["pair_id"],
                "example_id": example_id,
                "expected_is_ai": expected,
                "predicted_is_ai": predicted,
                "positive_votes": positive_votes,
                "correct": int(predicted == expected),
            }
        )
    majority_accuracy = 100 * majority_correct / len(example_ids)
    summary = {
        "task_model": TASK_MODEL,
        "repeat_count": len(results),
        "test_rows": 80,
        "accuracies_pct": accuracies,
        "mean_accuracy_pct": sum(accuracies) / len(accuracies),
        "min_accuracy_pct": min(accuracies),
        "max_accuracy_pct": max(accuracies),
        "all_repeats_at_or_below_50": all(value <= 50 for value in accuracies),
        "reference_strategy": "per-example majority vote across three fresh uncached runs",
        "majority_correct": majority_correct,
        "baseline_test_accuracy_pct": majority_accuracy,
        "baseline_gate_passed": majority_accuracy <= 50,
        "repeats": results,
    }
    from .guardrails import write_jsonl

    write_jsonl(root / "majority_predictions.jsonl", majority_predictions)
    atomic_write_json(root / "summary.json", summary)
    assert_lock_unchanged()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--repeats", type=int, default=3)
    baseline.add_argument("--stage-cap-usd", type=float, default=3.0)
    gepa = subparsers.add_parser("gepa")
    gepa.add_argument("--mode", choices=("smoke", "full"), required=True)
    gepa.add_argument("--output-dir", type=Path, required=True)
    gepa.add_argument("--max-full-evals", type=int, required=True)
    gepa.add_argument("--stage-cap-usd", type=float, default=10.0)
    gepa.add_argument("--threads", type=int, default=4)
    gepa.add_argument("--seed-program", type=Path)
    gepa.add_argument(
        "--candidate-selection-strategy",
        choices=("pareto", "current_best"),
        default="pareto",
    )
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--program", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--repeats", type=int, default=3)
    evaluate.add_argument("--stage-cap-usd", type=float, default=2.0)
    evaluate.add_argument("--threads", type=int, default=4)
    evaluate.add_argument("--validation-summary", type=Path)
    evaluate.add_argument("--minimum-validation-accuracy-pct", type=float, default=85.0)
    materialize = subparsers.add_parser("materialize-candidates")
    materialize.add_argument("--optimizer-results", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--candidate-index", type=int, action="append", default=[])
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--candidate-dir", type=Path, required=True)
    finalize.add_argument("--test-evaluation-dir", type=Path, required=True)
    finalize.add_argument("--final-dir", type=Path, default=RESULTS_DIR / "final")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "baseline":
        result = confirm_baseline(repeats=args.repeats, stage_cap_usd=args.stage_cap_usd)
    elif args.command == "gepa":
        result = run_gepa(
            mode=args.mode,
            output_dir=args.output_dir,
            max_full_evals=args.max_full_evals,
            stage_cap_usd=args.stage_cap_usd,
            threads=args.threads,
            seed_program_path=args.seed_program,
            candidate_selection_strategy=args.candidate_selection_strategy,
        )
    elif args.command == "evaluate":
        result = evaluate_repeated(
            program_path=args.program,
            split_name=args.split,
            output_dir=args.output_dir,
            repeats=args.repeats,
            stage_cap_usd=args.stage_cap_usd,
            threads=args.threads,
            validation_summary_path=args.validation_summary,
            minimum_validation_accuracy_pct=args.minimum_validation_accuracy_pct,
        )
    elif args.command == "materialize-candidates":
        result = materialize_gepa_candidates(
            optimizer_results_path=args.optimizer_results,
            output_dir=args.output_dir,
            candidate_indexes=args.candidate_index,
        )
    elif args.command == "finalize":
        result = finalize_result(
            candidate_dir=args.candidate_dir,
            test_evaluation_dir=args.test_evaluation_dir,
            final_dir=args.final_dir,
        )
    else:
        raise ValueError(args.command)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
