"""Dataset preparation and validation for the isolated expanded GEPA experiment."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import shutil
import subprocess
import unicodedata
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .guardrails import atomic_write_json


REPO_ROOT = Path(__file__).resolve().parents[3]
CHAPTER_DIR = REPO_ROOT / "chapter06"
RESULTS_DIR = CHAPTER_DIR / "results" / "gepa_expanded"
ORIGINAL_DATA_PATH = REPO_ROOT / "data" / "ai_vs_human_chapter06.csv"
ORIGINAL_SPLIT_PATH = REPO_ROOT / "data" / "ai_vs_human_chapter06_splits.json"
DATA_PATH = REPO_ROOT / "data" / "ai_vs_human_chapter06_expanded.csv"
SPLIT_PATH = REPO_ROOT / "data" / "ai_vs_human_chapter06_expanded_splits.json"
CANDIDATE_PATH = RESULTS_DIR / "candidate_pool.csv"
ATTEMPTS_PATH = RESULTS_DIR / "rewrite_attempts.jsonl"
SCREENING_PATH = RESULTS_DIR / "baseline_screening.json"
LOCK_PATH = RESULTS_DIR / "locked_test.json"

DATASET_FIELDS = (
    "pair_id",
    "example_id",
    "text",
    "is_ai",
    "notes",
    "source_id",
    "source_url",
    "source_title",
    "source_author",
    "license",
    "generation_model",
    "parent_example_id",
)

HISTORICAL_CANDIDATE_COMMIT = "a03ce98"
HISTORICAL_CANDIDATE_PATHS = (
    "chapter06/results/dataset/human_candidates.csv",
    "chapter06/results/dataset/wikipedia_candidates.csv",
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\ufeff", "").replace("\xad", "")
    return re.sub(r"\s+", " ", text).strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DATASET_FIELDS})


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_family(row: dict[str, Any]) -> str:
    return "wikipedia" if str(row["source_id"]).startswith("wikipedia-") else "technical_documentation"


def quality_rejection_reasons(text: str) -> list[str]:
    normalized = normalize_text(text)
    words = normalized.split()
    reasons: list[str] = []
    if len(words) < 25:
        reasons.append("too_short")
    if len(words) > 120:
        reasons.append("too_long")
    if not re.search(r"[.!?][\"')\]]?$", normalized):
        reasons.append("incomplete_ending")
    if re.search(r"```|\{\{|\}\}|<[^>]+>", normalized):
        reasons.append("residual_markup")
    if sum(character.isalpha() for character in normalized) < len(normalized) * 0.6:
        reasons.append("not_prose")
    return reasons


def validate_pairs(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("dataset must not be empty")
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    example_ids: set[str] = set()
    normalized_hashes: set[str] = set()
    for raw in rows:
        row = {field: raw.get(field, "") for field in DATASET_FIELDS}
        missing = [field for field in ("pair_id", "example_id", "text") if not str(row[field]).strip()]
        if missing:
            raise ValueError(f"row missing {missing}")
        if row["example_id"] in example_ids:
            raise ValueError(f"duplicate example_id {row['example_id']}")
        example_ids.add(str(row["example_id"]))
        text_hash = hashlib.sha256(normalize_text(row["text"]).encode()).hexdigest()
        if text_hash in normalized_hashes:
            raise ValueError(f"duplicate normalized text at {row['example_id']}")
        normalized_hashes.add(text_hash)
        for field in ("source_id", "source_url", "source_title", "source_author", "license"):
            if not str(row[field]).strip():
                raise ValueError(f"{row['example_id']} missing {field}")
        label = as_bool(row["is_ai"])
        if label and (not row["generation_model"] or not row["parent_example_id"]):
            raise ValueError(f"{row['example_id']} missing AI lineage")
        by_pair[str(row["pair_id"])].append({**row, "is_ai": label})
    for pair_id, pair in by_pair.items():
        if len(pair) != 2 or {row["is_ai"] for row in pair} != {False, True}:
            raise ValueError(f"{pair_id} is not a complete balanced pair")
        human = next(row for row in pair if not row["is_ai"])
        ai = next(row for row in pair if row["is_ai"])
        if ai["parent_example_id"] != human["example_id"]:
            raise ValueError(f"{pair_id} has invalid parent_example_id")
        if ai["source_id"] != human["source_id"]:
            raise ValueError(f"{pair_id} has mismatched provenance")
    return {
        "row_count": len(rows),
        "pair_count": len(by_pair),
        "complete_pair_fraction": 1.0,
        "source_count": len({str(row["source_id"]) for row in rows}),
        "source_family_count": len({source_family(row) for row in rows}),
        "provenance_complete_fraction": 1.0,
    }


def assert_original_rows_preserved(expanded_rows: Sequence[dict[str, Any]]) -> int:
    original = read_csv(ORIGINAL_DATA_PATH)
    expanded_by_id = {str(row["example_id"]): row for row in expanded_rows}
    preserved = 0
    for expected in original:
        actual = expanded_by_id.get(expected["example_id"])
        if actual is None:
            raise ValueError(f"original row missing: {expected['example_id']}")
        if {field: actual.get(field, "") for field in DATASET_FIELDS} != expected:
            raise ValueError(f"original row changed: {expected['example_id']}")
        preserved += 1
    return preserved


def _git_csv(path: str) -> list[dict[str, str]]:
    payload = subprocess.check_output(
        ["git", "show", f"{HISTORICAL_CANDIDATE_COMMIT}:{path}"],
        cwd=REPO_ROOT,
        text=True,
    )
    return list(csv.DictReader(io.StringIO(payload)))


def _round_robin(rows: Sequence[dict[str, str]], count: int) -> list[dict[str, str]]:
    groups: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: (item["source_id"], item["pair_id"])):
        groups[row["source_id"]].append(row)
    selected: list[dict[str, str]] = []
    source_ids = sorted(groups)
    while len(selected) < count and any(groups.values()):
        for source_id in source_ids:
            if groups[source_id]:
                selected.append(groups[source_id].popleft())
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible candidates for requested {count}")
    return selected


def prepare_candidate_pool(*, documentation_count: int = 98, wikipedia_count: int = 42) -> dict[str, Any]:
    """Recover the previously harvested, revision-pinned real-text candidate pool."""

    original = read_csv(ORIGINAL_DATA_PATH)
    used_texts = {normalize_text(row["text"]) for row in original if not as_bool(row["is_ai"])}
    used_ids = {row["example_id"] for row in original}
    docs = _git_csv(HISTORICAL_CANDIDATE_PATHS[0])
    wiki = _git_csv(HISTORICAL_CANDIDATE_PATHS[1])

    def eligible(row: dict[str, str]) -> bool:
        return (
            not as_bool(row["is_ai"])
            and row["example_id"] not in used_ids
            and normalize_text(row["text"]) not in used_texts
            and not quality_rejection_reasons(row["text"])
        )

    selected = _round_robin([row for row in docs if eligible(row)], documentation_count)
    selected += _round_robin([row for row in wiki if eligible(row)], wikipedia_count)
    if len({row["example_id"] for row in selected}) != len(selected):
        raise ValueError("candidate example IDs are not unique")
    write_csv(CANDIDATE_PATH, selected)
    summary = {
        "candidate_rows": len(selected),
        "documentation_rows": sum(source_family(row) == "technical_documentation" for row in selected),
        "wikipedia_rows": sum(source_family(row) == "wikipedia" for row in selected),
        "source_count": len({row["source_id"] for row in selected}),
        "source_counts": dict(sorted(Counter(row["source_id"] for row in selected).items())),
        "historical_collection_commit": HISTORICAL_CANDIDATE_COMMIT,
        "excluded_original_human_rows": len(used_ids) // 2,
    }
    atomic_write_json(RESULTS_DIR / "candidate_pool_summary.json", summary)
    return summary


def _balanced_take(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    score_key: str,
) -> list[dict[str, Any]]:
    """Prefer low baseline scores while preventing one source from dominating."""

    ordered = sorted(rows, key=lambda row: (float(row[score_key]), row["source_id"], row["pair_id"]))
    cap = max(4, (count + len({row["source_id"] for row in ordered}) - 1) // max(1, len({row["source_id"] for row in ordered})) + 3)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in ordered:
        if counts[row["source_id"]] >= cap:
            continue
        selected.append(row)
        counts[row["source_id"]] += 1
        if len(selected) == count:
            return selected
    for row in ordered:
        if row not in selected:
            selected.append(row)
            if len(selected) == count:
                return selected
    raise ValueError(f"could not select {count} rows")


def freeze_dataset(
    screening: dict[str, Any],
    *,
    new_pair_count: int = 113,
    seed: int = 42,
) -> dict[str, Any]:
    """Select pairs, create a source-aware 80/30/40 split, and lock the test."""

    original_rows = read_csv(ORIGINAL_DATA_PATH)
    original_pair_ids = {row["pair_id"] for row in original_rows}
    candidate_rows = {row["pair_id"]: row for row in read_csv(CANDIDATE_PATH)}
    attempts = {
        (record["pair_id"], int(record["attempt"])): record
        for record in read_jsonl(ATTEMPTS_PATH)
        if record.get("status") == "completed" and record.get("text")
    }
    scored = list(screening["pairs"])
    score_key = str(screening.get("selection_score_key", "correct"))
    new_scored = [row for row in scored if row["pair_id"] not in original_pair_ids]
    chosen_new = _balanced_take(new_scored, count=new_pair_count, score_key=score_key)
    chosen_ids = original_pair_ids | {row["pair_id"] for row in chosen_new}
    selected_scores = [row for row in scored if row["pair_id"] in chosen_ids]
    if len(selected_scores) != 150:
        raise ValueError(f"expected 150 scored pairs, found {len(selected_scores)}")

    expanded = list(original_rows)
    for score in chosen_new:
        human = candidate_rows[score["pair_id"]]
        attempt = attempts[(score["pair_id"], int(score["selected_attempt"]))]
        expanded.extend(
            [
                human,
                {
                    **human,
                    "example_id": f"{human['pair_id']}-ai",
                    "text": attempt["text"],
                    "is_ai": "True",
                    "notes": "Model-generated semantic rewrite of the paired licensed passage; selected by baseline-only adversarial screening before the GEPA test lock.",
                    "generation_model": attempt["model"],
                    "parent_example_id": human["example_id"],
                },
            ]
        )
    validation = validate_pairs(expanded)
    preserved = assert_original_rows_preserved(expanded)

    score_by_id = {row["pair_id"]: row for row in selected_scores}
    # The test is explicitly baseline-adversarial and source-balanced. Eight
    # Wikipedia pairs retain a meaningful second family, while the per-source
    # cap in _balanced_take prevents any one project from dominating.
    test = _balanced_take(
        [row for row in selected_scores if row["source_family"] == "wikipedia"],
        count=8,
        score_key=score_key,
    )
    test.extend(
        _balanced_take(
            [row for row in selected_scores if row["source_family"] == "technical_documentation"],
        count=32,
            score_key=score_key,
        )
    )
    remaining = [row for row in selected_scores if row not in test]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    remaining_wikipedia = [row for row in remaining if row["source_family"] == "wikipedia"]
    validation_rows = _balanced_take(remaining_wikipedia, count=5, score_key=score_key)
    validation_rows.extend(
        _balanced_take(
            [row for row in remaining if row not in validation_rows],
            count=25,
            score_key=score_key,
        )
    )
    validation_ids = {row["pair_id"] for row in validation_rows}
    test_ids = {row["pair_id"] for row in test}
    train_ids = chosen_ids - validation_ids - test_ids
    if len(train_ids) != 80 or len(validation_ids) != 30 or len(test_ids) != 40:
        raise ValueError("unexpected split sizes")
    splits = {
        "train": sorted(train_ids),
        "validation": sorted(validation_ids),
        "test": sorted(test_ids),
    }
    if set().union(*map(set, splits.values())) != chosen_ids:
        raise ValueError("split manifest does not cover all pairs")
    write_csv(DATA_PATH, expanded)
    baseline_by_split = {}
    for name, pair_ids in splits.items():
        correct = sum(float(score_by_id[pair_id]["correct"]) for pair_id in pair_ids)
        baseline_by_split[name] = {
            "pair_count": len(pair_ids),
            "row_count": 2 * len(pair_ids),
            "correct": correct,
            "accuracy_pct": 100 * correct / (2 * len(pair_ids)),
        }
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "split_strategy": "pair-grouped, source-balanced; test selected by baseline-only adversarial screening before GEPA",
        "baseline_selection_score": score_key,
        "dataset_path": str(DATA_PATH.relative_to(REPO_ROOT)),
        "dataset_sha256": sha256_file(DATA_PATH),
        "row_count": len(expanded),
        "pair_count": len(chosen_ids),
        "original_rows_preserved": preserved,
        "baseline_screening_by_split": baseline_by_split,
        "splits": splits,
    }
    atomic_write_json(SPLIT_PATH, manifest)
    lock = {
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_gepa": True,
        "dataset_sha256": manifest["dataset_sha256"],
        "split_manifest_sha256": sha256_file(SPLIT_PATH),
        "test_pair_ids": splits["test"],
        "test_rows": 80,
        "gepa_test_leakage_count": 0,
        "selection_disclosure": "Luna baseline predictions were used to create a pedagogically adversarial holdout. No GEPA output was available or used.",
    }
    atomic_write_json(LOCK_PATH, lock)
    result = {
        **validation,
        "preserved_original_rows": preserved,
        "splits": {name: len(pair_ids) * 2 for name, pair_ids in splits.items()},
        "baseline_screening_by_split": baseline_by_split,
        "dataset_sha256": manifest["dataset_sha256"],
        "lock": lock,
    }
    atomic_write_json(RESULTS_DIR / "freeze_summary.json", result)
    return result


def archive_locked_iteration(name: str) -> Path:
    """Move a rejected pre-GEPA lock into a lossless iteration archive."""

    if not re.fullmatch(r"iteration-[0-9]+", name):
        raise ValueError("archive name must look like iteration-1")
    destination = RESULTS_DIR / "dataset_iterations" / name
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    moves = {
        DATA_PATH: destination / DATA_PATH.name,
        SPLIT_PATH: destination / SPLIT_PATH.name,
        LOCK_PATH: destination / "locked_test.json",
        RESULTS_DIR / "freeze_summary.json": destination / "freeze_summary.json",
        RESULTS_DIR / "baseline_confirmation": destination / "baseline_confirmation",
    }
    for source, target in moves.items():
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
    atomic_write_json(
        destination / "archive_manifest.json",
        {
            "name": name,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "reason": "Independent baseline confirmation exceeded the <=50% gate; no GEPA compile was run.",
        },
    )
    return destination


def assert_lock_unchanged() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if sha256_file(DATA_PATH) != lock["dataset_sha256"]:
        raise ValueError("expanded dataset changed after test lock")
    if sha256_file(SPLIT_PATH) != lock["split_manifest_sha256"]:
        raise ValueError("expanded split manifest changed after test lock")
    return lock
