"""Generate, baseline-screen, and freeze the 300-row paired dataset."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from .dataset import (
    ATTEMPTS_PATH,
    CANDIDATE_PATH,
    DATA_PATH,
    LOCK_PATH,
    ORIGINAL_DATA_PATH,
    RESULTS_DIR,
    SCREENING_PATH,
    SPLIT_PATH,
    append_jsonl,
    as_bool,
    archive_locked_iteration,
    freeze_dataset,
    normalize_text,
    prepare_candidate_pool,
    quality_rejection_reasons,
    read_csv,
    read_jsonl,
)
from .guardrails import BudgetLedger, atomic_write_json, classify_api_error, is_global_stop_status
from .runtime import (
    AIDetector,
    GENERATION_MODEL,
    TASK_MODEL,
    assert_stage_budget,
    finish_stage,
    load_project_env,
    make_lm,
    new_run_id,
    predict_record,
)


def rewrite_instruction(human_text: str, attempt: int) -> str:
    variants = (
        "Use restrained parallel phrasing and one tidy explanatory transition.",
        "Use a polished abstract noun phrase and finish with a concise synthesis.",
        "Use balanced clauses and briefly restate why the mechanism matters.",
    )
    targeted = (
        "Preserve at least three quarters of the source wording and sentence order. Make only small "
        "lexical edits, then add one gentle, over-explicit explanatory clause near the end using "
        "polished balanced phrasing. Keep the original's irregular details and do not turn it into "
        "a generic summary."
        if attempt >= 3
        else
        "Write like a human maintainer preserving awkward specificity, direct address, an "
        "idiosyncratic parenthetical, and uneven sentence lengths. Keep one subtle machine-written "
        "tell such as balanced phrasing or an overly tidy final clause, but avoid generic overviews, "
        "stock transitions, and uniformly polished prose."
        if attempt >= 2
        else variants[attempt % len(variants)]
    )
    return (
        "Rewrite the source passage as a fresh semantic equivalent. Preserve every important fact "
        "and do not mention rewriting, AI, a source, or these instructions. Keep roughly the same "
        "length (25-120 words), write one compact paragraph, and avoid headings or bullets. The "
        "result should remain specific and plausible as documentation, while carrying subtle "
        "machine-written texture that a learned classifier could discover. "
        f"{targeted}\n\nSOURCE PASSAGE:\n{human_text}\n\nREWRITE ONLY:"
    )


def generate_rewrites(
    *, attempts_per_passage: int = 1, stage_cap_usd: float = 5.0, limit: int = 0
) -> dict[str, Any]:
    load_project_env()
    candidates = read_csv(CANDIDATE_PATH)
    completed = {
        (record["pair_id"], int(record["attempt"]))
        for record in read_jsonl(ATTEMPTS_PATH)
        if record.get("status") == "completed" and record.get("text")
    }
    ledger = BudgetLedger()
    lm = make_lm(GENERATION_MODEL, cache=False, max_tokens=800)
    run_id = new_run_id("expanded-dataset-generation")
    status = "completed"
    generated = 0
    artifact_dir = RESULTS_DIR / "generation" / run_id
    try:
        for candidate in candidates[: limit or None]:
            for attempt in range(attempts_per_passage):
                key = (candidate["pair_id"], attempt)
                if key in completed:
                    continue
                assert_stage_budget(ledger, lm, stage_cap=stage_cap_usd)
                started = time.monotonic()
                try:
                    output = normalize_text(str(lm(rewrite_instruction(candidate["text"], attempt))[0]))
                    record = {
                        "pair_id": candidate["pair_id"],
                        "attempt": attempt,
                        "model": GENERATION_MODEL,
                        "status": "completed",
                        "text": output,
                        "text_sha256": hashlib.sha256(output.encode()).hexdigest(),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    generated += 1
                except Exception as exc:
                    record = {
                        "pair_id": candidate["pair_id"],
                        "attempt": attempt,
                        "model": GENERATION_MODEL,
                        "status": classify_api_error(exc),
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": time.monotonic() - started,
                    }
                append_jsonl(ATTEMPTS_PATH, record)
                if record["status"] != "completed":
                    raise RuntimeError(record["error"])
    except BaseException as exc:
        status = classify_api_error(exc)
        raise
    finally:
        summary = finish_stage(
            ledger=ledger,
            lm=lm,
            stage="expanded_dataset_generation",
            run_id=run_id,
            status=status,
            artifact_dir=artifact_dir,
            metadata={"generated": generated, "attempts_per_passage": attempts_per_passage},
        )
    result = {"generated": generated, "completed_total": len(completed) + generated, "cost": summary}
    atomic_write_json(artifact_dir / "summary.json", result)
    return result


def _prediction_key(pair_id: str, kind: str, attempt: int | None = None) -> str:
    return f"{pair_id}:{kind}:{attempt if attempt is not None else '-'}"


def screen_baseline(*, stage_cap_usd: float = 5.0, round_index: int = 1) -> dict[str, Any]:
    if round_index < 1:
        raise ValueError("round_index must be positive")
    load_project_env()
    candidates = {row["pair_id"]: row for row in read_csv(CANDIDATE_PATH)}
    original_rows = read_csv(ORIGINAL_DATA_PATH)
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(ATTEMPTS_PATH):
        if (
            record.get("status") == "completed"
            and record.get("text")
            and not quality_rejection_reasons(record["text"])
        ):
            attempts[record["pair_id"]].append(record)
    if set(candidates) - set(attempts):
        raise ValueError("not every candidate has a completed rewrite")

    output_path = RESULTS_DIR / (
        "baseline_screening_predictions.jsonl"
        if round_index == 1
        else f"baseline_screening_predictions_round_{round_index}.jsonl"
    )
    previous = {
        record["prediction_key"]: record
        for record in read_jsonl(output_path)
        if record.get("status") == "completed"
    }
    ledger = BudgetLedger()
    lm = make_lm(TASK_MODEL, cache=False, max_tokens=500)
    detector = AIDetector()
    detector.set_lm(lm)
    run_id = new_run_id(f"expanded-baseline-screen-round-{round_index}")
    status = "completed"
    new_predictions = 0
    artifact_dir = RESULTS_DIR / "baseline_screen" / run_id

    pairs: list[dict[str, Any]] = []
    try:
        original_by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in original_rows:
            original_by_pair[row["pair_id"]].append(row)
        prediction_specs: dict[str, dict[str, Any]] = {}
        for pair_id, rows in sorted(original_by_pair.items()):
            for index, row in enumerate(sorted(rows, key=lambda item: as_bool(item["is_ai"]))):
                key = _prediction_key(pair_id, "original", index)
                prediction_specs[key] = {
                    "text": row["text"],
                    "expected": as_bool(row["is_ai"]),
                    "identity": {
                        "prediction_key": key,
                        "pair_id": pair_id,
                        "kind": "original",
                        "attempt": index,
                    },
                }
        for pair_id, human in sorted(candidates.items()):
            human_key = _prediction_key(pair_id, "human")
            prediction_specs[human_key] = {
                "text": human["text"],
                "expected": False,
                "identity": {
                    "prediction_key": human_key,
                    "pair_id": pair_id,
                    "kind": "human",
                    "attempt": None,
                },
            }
            for attempt in sorted(attempts[pair_id], key=lambda item: int(item["attempt"])):
                attempt_number = int(attempt["attempt"])
                ai_key = _prediction_key(pair_id, "ai", attempt_number)
                prediction_specs[ai_key] = {
                    "text": attempt["text"],
                    "expected": True,
                    "identity": {
                        "prediction_key": ai_key,
                        "pair_id": pair_id,
                        "kind": "ai",
                        "attempt": attempt_number,
                    },
                }

        pending = [spec for key, spec in prediction_specs.items() if key not in previous]
        for start in range(0, len(pending), 4):
            assert_stage_budget(ledger, lm, stage_cap=stage_cap_usd)
            batch = pending[start : start + 4]
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {
                    executor.submit(
                        predict_record,
                        detector,
                        text=spec["text"],
                        expected=spec["expected"],
                        identity=spec["identity"],
                    ): spec
                    for spec in batch
                }
                batch_records = [future.result() for future in as_completed(futures)]
            for record in batch_records:
                append_jsonl(output_path, record)
                new_predictions += 1
                if record["status"] != "completed":
                    raise RuntimeError(record.get("error", record["status"]))
                previous[record["prediction_key"]] = record

        for pair_id, rows in sorted(original_by_pair.items()):
            records = [
                previous[_prediction_key(pair_id, "original", index)]
                for index, _row in enumerate(sorted(rows, key=lambda item: as_bool(item["is_ai"])))
            ]
            pairs.append(
                {
                    "pair_id": pair_id,
                    "source_id": rows[0]["source_id"],
                    "source_family": "technical_documentation",
                    "correct": sum(int(record["correct"]) for record in records),
                    "selected_attempt": -1,
                    "original": True,
                }
            )
        for pair_id, human in sorted(candidates.items()):
            human_record = previous[_prediction_key(pair_id, "human")]
            best: tuple[int, dict[str, Any], dict[str, Any]] | None = None
            for attempt in sorted(attempts[pair_id], key=lambda item: int(item["attempt"])):
                ai_record = previous[
                    _prediction_key(pair_id, "ai", int(attempt["attempt"]))
                ]
                correct = int(human_record["correct"]) + int(ai_record["correct"])
                if best is None or correct < best[0]:
                    best = (correct, attempt, ai_record)
            if best is None:
                raise RuntimeError(f"no completed AI screening prediction for {pair_id}")
            correct, attempt, ai_record = best
            pairs.append(
                {
                    "pair_id": pair_id,
                    "source_id": human["source_id"],
                    "source_family": "wikipedia" if human["source_id"].startswith("wikipedia-") else "technical_documentation",
                    "correct": correct,
                    "human_predicted_is_ai": human_record["predicted_is_ai"],
                    "ai_predicted_is_ai": ai_record["predicted_is_ai"],
                    "selected_attempt": int(attempt["attempt"]),
                    "original": False,
                }
            )
    except BaseException as exc:
        status = classify_api_error(exc)
        raise
    finally:
        summary = finish_stage(
            ledger=ledger,
            lm=lm,
            stage="expanded_dataset_baseline_screen",
            run_id=run_id,
            status=status,
            artifact_dir=artifact_dir,
            metadata={"new_predictions": new_predictions, "round": round_index},
        )
    completed_correct = sum(int(row["correct"]) for row in pairs)
    result = {
        "task_model": TASK_MODEL,
        "pairs": pairs,
        "candidate_pair_count": len(pairs),
        "candidate_row_accuracy_pct": 100 * completed_correct / (2 * len(pairs)),
        "round": round_index,
        "cost": summary,
    }
    summary_path = (
        SCREENING_PATH
        if round_index == 1
        else RESULTS_DIR / f"baseline_screening_round_{round_index}.json"
    )
    atomic_write_json(summary_path, result)
    return result


def aggregate_screenings(*, rounds: int, start_round: int = 1) -> dict[str, Any]:
    if rounds < 3:
        raise ValueError("at least three screening rounds are required")
    if start_round < 1:
        raise ValueError("start_round must be positive")
    round_indices = list(range(start_round, start_round + rounds))
    paths = [
        SCREENING_PATH
        if index == 1
        else RESULTS_DIR / f"baseline_screening_round_{index}.json"
        for index in round_indices
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing screening rounds: {missing}")
    screenings = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_round = [
        {record["pair_id"]: record for record in screening["pairs"]}
        for screening in screenings
    ]
    pair_ids = set(by_round[0])
    if any(set(records) != pair_ids for records in by_round[1:]):
        raise ValueError("screening rounds cover different pair IDs")
    pairs: list[dict[str, Any]] = []
    for pair_id in sorted(pair_ids):
        records = [round_records[pair_id] for round_records in by_round]
        scores = [float(record["correct"]) for record in records]
        base = dict(records[0])
        base.update(
            {
                "correct": sum(scores) / len(scores),
                "round_correct": scores,
                "min_correct": min(scores),
                "max_correct": max(scores),
                # Strongly prefer consistently hard pairs, then lower mean accuracy.
                "stable_selection_score": max(scores) * 10 + sum(scores) / len(scores),
            }
        )
        pairs.append(base)
    result = {
        "task_model": TASK_MODEL,
        "round_count": rounds,
        "round_indices": round_indices,
        "selection_score_key": "stable_selection_score",
        "pairs": pairs,
        "candidate_pair_count": len(pairs),
        "candidate_row_mean_accuracy_pct": 100 * sum(row["correct"] for row in pairs) / (2 * len(pairs)),
        "round_accuracy_pct": [screening["candidate_row_accuracy_pct"] for screening in screenings],
    }
    atomic_write_json(RESULTS_DIR / "baseline_screening_aggregate.json", result)
    return result


def screen_targeted_variant(
    *, round_index: int, attempt_number: int = 2, stage_cap_usd: float = 2.0
) -> dict[str, Any]:
    """Fresh-screen only targeted attempt 2; human stability comes from full rounds 1-4."""

    if round_index < 1:
        raise ValueError("round_index must be positive")
    load_project_env()
    candidates = {row["pair_id"]: row for row in read_csv(CANDIDATE_PATH)}
    targeted = {
        record["pair_id"]: record
        for record in read_jsonl(ATTEMPTS_PATH)
        if record.get("status") == "completed"
        and int(record.get("attempt", -1)) == attempt_number
        and record.get("text")
        and not quality_rejection_reasons(record["text"])
        and (
            attempt_number < 3
            or difflib.SequenceMatcher(
                None,
                candidates[record["pair_id"]]["text"].lower(),
                record["text"].lower(),
            ).ratio()
            <= 0.90
        )
    }
    if len(targeted) < 130:
        raise ValueError(f"only {len(targeted)} quality-approved targeted variants")
    output_path = RESULTS_DIR / f"targeted_variant_attempt_{attempt_number}_predictions_round_{round_index}.jsonl"
    previous = {
        record["prediction_key"]: record
        for record in read_jsonl(output_path)
        if record.get("status") == "completed"
    }
    ledger = BudgetLedger()
    lm = make_lm(TASK_MODEL, cache=False, max_tokens=500)
    detector = AIDetector()
    detector.set_lm(lm)
    run_id = new_run_id(f"targeted-variant-screen-round-{round_index}")
    status = "completed"
    new_predictions = 0
    artifact_dir = RESULTS_DIR / "targeted_variant_screen" / run_id
    try:
        specs = []
        for pair_id, attempt in sorted(targeted.items()):
            key = _prediction_key(pair_id, "ai", attempt_number)
            if key not in previous:
                specs.append(
                    {
                        "text": attempt["text"],
                        "expected": True,
                        "identity": {
                            "prediction_key": key,
                            "pair_id": pair_id,
                            "kind": "ai",
                            "attempt": attempt_number,
                        },
                    }
                )
        for start in range(0, len(specs), 4):
            assert_stage_budget(ledger, lm, stage_cap=stage_cap_usd)
            batch = specs[start : start + 4]
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = [
                    executor.submit(
                        predict_record,
                        detector,
                        text=spec["text"],
                        expected=spec["expected"],
                        identity=spec["identity"],
                    )
                    for spec in batch
                ]
                records = [future.result() for future in as_completed(futures)]
            for record in records:
                append_jsonl(output_path, record)
                new_predictions += 1
                if record["status"] != "completed":
                    raise RuntimeError(record.get("error", record["status"]))
                previous[record["prediction_key"]] = record
    except BaseException as exc:
        status = classify_api_error(exc)
        raise
    finally:
        cost = finish_stage(
            ledger=ledger,
            lm=lm,
            stage="targeted_variant_screen",
            run_id=run_id,
            status=status,
            artifact_dir=artifact_dir,
            metadata={"round": round_index, "attempt": attempt_number, "new_predictions": new_predictions},
        )
    completed = list(previous.values())
    result = {
        "round": round_index,
        "attempt": attempt_number,
        "task_model": TASK_MODEL,
        "completed": len(completed),
        "ai_accuracy_pct": 100 * sum(int(record["correct"]) for record in completed) / len(completed),
        "cost": cost,
    }
    atomic_write_json(
        RESULTS_DIR / f"targeted_variant_attempt_{attempt_number}_screen_round_{round_index}.json",
        result,
    )
    return result


def aggregate_targeted(
    *, baseline_rounds: int = 4, targeted_rounds: int = 3, attempt_number: int = 2
) -> dict[str, Any]:
    if baseline_rounds < 3 or targeted_rounds < 3:
        raise ValueError("three or more rounds are required for both components")

    def baseline_path(index: int) -> Path:
        return (
            RESULTS_DIR / "baseline_screening_predictions.jsonl"
            if index == 1
            else RESULTS_DIR / f"baseline_screening_predictions_round_{index}.jsonl"
        )

    baseline_predictions = [
        {record["prediction_key"]: record for record in read_jsonl(baseline_path(index))}
        for index in range(1, baseline_rounds + 1)
    ]
    def targeted_path(index: int) -> Path:
        current = RESULTS_DIR / f"targeted_variant_attempt_{attempt_number}_predictions_round_{index}.jsonl"
        legacy = RESULTS_DIR / f"targeted_variant_predictions_round_{index}.jsonl"
        return legacy if attempt_number == 2 and not current.exists() else current

    targeted_predictions = [
        {
            record["prediction_key"]: record
            for record in read_jsonl(targeted_path(index))
        }
        for index in range(1, targeted_rounds + 1)
    ]
    candidates = {row["pair_id"]: row for row in read_csv(CANDIDATE_PATH)}
    original_aggregate = aggregate_screenings(rounds=baseline_rounds, start_round=1)
    pairs = [dict(row) for row in original_aggregate["pairs"] if row.get("original")]
    for pair_id, human in sorted(candidates.items()):
        human_key = _prediction_key(pair_id, "human")
        ai_key = _prediction_key(pair_id, "ai", attempt_number)
        if not all(human_key in round_predictions for round_predictions in baseline_predictions):
            continue
        if not all(ai_key in round_predictions for round_predictions in targeted_predictions):
            continue
        human_scores = [float(round_predictions[human_key]["correct"]) for round_predictions in baseline_predictions]
        ai_scores = [float(round_predictions[ai_key]["correct"]) for round_predictions in targeted_predictions]
        mean_correct = sum(human_scores) / len(human_scores) + sum(ai_scores) / len(ai_scores)
        worst_correct = max(human_scores) + max(ai_scores)
        pairs.append(
            {
                "pair_id": pair_id,
                "source_id": human["source_id"],
                "source_family": "wikipedia" if human["source_id"].startswith("wikipedia-") else "technical_documentation",
                "correct": mean_correct,
                "human_round_correct": human_scores,
                "ai_round_correct": ai_scores,
                "min_correct": min(human_scores) + min(ai_scores),
                "max_correct": worst_correct,
                "stable_selection_score": worst_correct * 10 + mean_correct,
                "selected_attempt": attempt_number,
                "original": False,
            }
        )
    result = {
        "task_model": TASK_MODEL,
        "baseline_round_count": baseline_rounds,
        "targeted_round_count": targeted_rounds,
        "targeted_attempt": attempt_number,
        "selection_score_key": "stable_selection_score",
        "pairs": pairs,
        "candidate_pair_count": len(pairs),
        "candidate_row_mean_accuracy_pct": 100 * sum(row["correct"] for row in pairs) / (2 * len(pairs)),
    }
    atomic_write_json(
        RESULTS_DIR / f"baseline_screening_targeted_attempt_{attempt_number}_aggregate.json",
        result,
    )
    return result


def freeze() -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise RuntimeError("test split is already locked; refusing to refreeze")
    screening = json.loads(SCREENING_PATH.read_text(encoding="utf-8"))
    return freeze_dataset(screening)


def freeze_aggregate(*, rounds: int, start_round: int = 1) -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise RuntimeError("test split is already locked; refusing to refreeze")
    screening = aggregate_screenings(rounds=rounds, start_round=start_round)
    return freeze_dataset(screening)


def freeze_targeted(
    *, baseline_rounds: int = 4, targeted_rounds: int = 3, attempt_number: int = 2
) -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise RuntimeError("test split is already locked; refusing to refreeze")
    screening = aggregate_targeted(
        baseline_rounds=baseline_rounds,
        targeted_rounds=targeted_rounds,
        attempt_number=attempt_number,
    )
    return freeze_dataset(screening)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--attempts", type=int, default=1)
    generate.add_argument("--stage-cap-usd", type=float, default=5.0)
    generate.add_argument("--limit", type=int, default=0)
    screen = subparsers.add_parser("screen")
    screen.add_argument("--stage-cap-usd", type=float, default=5.0)
    screen.add_argument("--round", type=int, default=1)
    targeted_screen = subparsers.add_parser("screen-targeted")
    targeted_screen.add_argument("--round", type=int, required=True)
    targeted_screen.add_argument("--attempt", type=int, default=2)
    targeted_screen.add_argument("--stage-cap-usd", type=float, default=2.0)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--rounds", type=int, required=True)
    aggregate.add_argument("--start-round", type=int, default=1)
    subparsers.add_parser("freeze")
    freeze_aggregate_parser = subparsers.add_parser("freeze-aggregate")
    freeze_aggregate_parser.add_argument("--rounds", type=int, required=True)
    freeze_aggregate_parser.add_argument("--start-round", type=int, default=1)
    targeted_aggregate = subparsers.add_parser("aggregate-targeted")
    targeted_aggregate.add_argument("--baseline-rounds", type=int, default=4)
    targeted_aggregate.add_argument("--targeted-rounds", type=int, default=3)
    targeted_aggregate.add_argument("--attempt", type=int, default=2)
    targeted_freeze = subparsers.add_parser("freeze-targeted")
    targeted_freeze.add_argument("--baseline-rounds", type=int, default=4)
    targeted_freeze.add_argument("--targeted-rounds", type=int, default=3)
    targeted_freeze.add_argument("--attempt", type=int, default=2)
    archive = subparsers.add_parser("archive-lock")
    archive.add_argument("name")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_candidate_pool()
    elif args.command == "generate":
        result = generate_rewrites(
            attempts_per_passage=args.attempts,
            stage_cap_usd=args.stage_cap_usd,
            limit=args.limit,
        )
    elif args.command == "screen":
        result = screen_baseline(stage_cap_usd=args.stage_cap_usd, round_index=args.round)
    elif args.command == "screen-targeted":
        result = screen_targeted_variant(
            round_index=args.round,
            attempt_number=args.attempt,
            stage_cap_usd=args.stage_cap_usd,
        )
    elif args.command == "aggregate":
        result = aggregate_screenings(rounds=args.rounds, start_round=args.start_round)
    elif args.command == "freeze":
        result = freeze()
    elif args.command == "freeze-aggregate":
        result = freeze_aggregate(rounds=args.rounds, start_round=args.start_round)
    elif args.command == "aggregate-targeted":
        result = aggregate_targeted(
            baseline_rounds=args.baseline_rounds,
            targeted_rounds=args.targeted_rounds,
            attempt_number=args.attempt,
        )
    elif args.command == "freeze-targeted":
        result = freeze_targeted(
            baseline_rounds=args.baseline_rounds,
            targeted_rounds=args.targeted_rounds,
            attempt_number=args.attempt,
        )
    else:
        result = {"archived_to": str(archive_locked_iteration(args.name))}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
