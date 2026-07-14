"""Build the provenance-bearing adversarial dataset used by Chapter 6.

The paid workflow is intentionally staged:

1. ``collect`` downloads public-domain source texts and extracts deterministic
   human passage candidates.
2. ``generate`` creates AI rewrites and stores every raw attempt.
3. ``screen`` evaluates both sides with the unchanged chapter baseline and
   retains the hardest complete pairs.
4. ``freeze`` validates the selected pairs and writes an immutable grouped
   split manifest.

Each command writes intermediate state so an API failure never discards work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_module
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
import urllib.request
from urllib.error import HTTPError
from urllib.parse import urlencode
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from chapter06.experiment_runtime import (
    BudgetLedger,
    classify_api_error,
    is_global_stop_status,
    summarize_history,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CHAPTER_DIR = REPO_ROOT / "chapter06"
RESULTS_DIR = CHAPTER_DIR / "results" / "dataset"
SOURCES_PATH = CHAPTER_DIR / "data_sources.yaml"
HUMAN_CANDIDATES_PATH = RESULTS_DIR / "human_candidates.csv"
WIKIPEDIA_CANDIDATES_PATH = RESULTS_DIR / "wikipedia_candidates.csv"
REWRITE_ATTEMPTS_PATH = RESULTS_DIR / "rewrite_attempts.jsonl"
SCREENED_PAIRS_PATH = RESULTS_DIR / "screened_pairs.csv"
HUMAN_PREDICTIONS_PATH = RESULTS_DIR / "human_baseline_predictions.jsonl"
PAIR_PREDICTIONS_PATH = RESULTS_DIR / "pair_baseline_predictions.jsonl"
STABILITY_PREDICTIONS_PATH = RESULTS_DIR / "baseline_stability_predictions.jsonl"
ADVERSARIAL_ROUNDS_PATH = RESULTS_DIR / "adversarial_rounds.jsonl"
QUALITY_REJECTIONS_PATH = RESULTS_DIR / "quality_rejections.jsonl"
DATASET_PATH = REPO_ROOT / "data" / "ai_vs_human_chapter06.csv"
SPLITS_PATH = REPO_ROOT / "data" / "ai_vs_human_chapter06_splits.json"
BASELINE_GATE_PATH = RESULTS_DIR / "baseline_gate.json"

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


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "").replace("\xad", "")
    return re.sub(r"\s+", " ", text).strip()


def quality_rejection_reasons(text: str) -> list[str]:
    """Return deterministic reasons a source excerpt is unsuitable for the benchmark."""

    normalized = _normalize_text(text)
    word_count = len(normalized.split())
    reasons: list[str] = []
    if word_count < 30:
        reasons.append("too_short")
    if word_count > 100:
        reasons.append("too_long")
    if not re.search(r"[.!?][\"')\]]?$", normalized):
        reasons.append("incomplete_ending")
    if re.search(r"```|\{\{|\}\}|<[^>]+>|~[A-Za-z_][\w.]*", normalized):
        reasons.append("residual_markup")
    return reasons


def _gutenberg_body(raw: str) -> str:
    start = re.search(r"(?im)^\*{3}\s*START OF[^\n]*\*{3}\s*$", raw)
    end = re.search(r"(?im)^\*{3}\s*END OF[^\n]*\*{3}\s*$", raw)
    start_index = start.end() if start else 0
    end_index = end.start() if end and end.start() > start_index else len(raw)
    return raw[start_index:end_index]


def extract_passages(raw: str, *, min_words: int = 80, max_words: int = 180) -> list[str]:
    """Extract prose paragraphs from a Project Gutenberg-style plain-text file."""

    if min_words < 1 or max_words < min_words:
        raise ValueError("word limits must satisfy 1 <= min_words <= max_words")

    body = _gutenberg_body(raw.replace("\r\n", "\n").replace("\r", "\n"))
    passages: list[str] = []
    seen: set[str] = set()
    for block in re.split(r"\n\s*\n", body):
        passage = _normalize_text(block)
        words = passage.split()
        if not min_words <= len(words) <= max_words:
            continue
        if passage.isupper() or sum(character.isalpha() for character in passage) < len(passage) * 0.5:
            continue
        digest = hashlib.sha256(passage.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        passages.append(passage)
    return passages


def extract_html_passages(html: str, *, min_words: int = 40, max_words: int = 120) -> list[str]:
    """Extract clean prose paragraphs from rendered MediaWiki HTML."""

    if min_words < 1 or max_words < min_words:
        raise ValueError("word limits must satisfy 1 <= min_words <= max_words")
    cleaned_html = html
    for tag in ("table", "sup", "style", "script", "math"):
        cleaned_html = re.sub(
            rf"(?is)<{tag}\b[^>]*>.*?</{tag}>",
            " ",
            cleaned_html,
        )
    passages: list[str] = []
    seen: set[str] = set()
    for paragraph in re.findall(r"(?is)<p\b[^>]*>(.*?)</p>", cleaned_html):
        text = re.sub(r"(?s)<[^>]+>", " ", paragraph)
        text = _normalize_text(html_module.unescape(text))
        words = text.split()
        if not min_words <= len(words) <= max_words:
            continue
        if text.startswith(("This article", "For other uses", "Coordinates:")):
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        passages.append(text)
    return passages


def extract_short_passages(raw: str, *, min_words: int = 25, max_words: int = 65) -> list[str]:
    """Extract concise complete sentence windows from a Gutenberg-style text."""

    if min_words < 1 or max_words < min_words:
        raise ValueError("word limits must satisfy 1 <= min_words <= max_words")
    body = _gutenberg_body(raw.replace("\r\n", "\n").replace("\r", "\n"))
    passages: list[str] = []
    seen: set[str] = set()
    for block in re.split(r"\n\s*\n", body):
        paragraph = _normalize_text(block)
        if not paragraph or paragraph.isupper():
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[\"“‘']?[A-Z])", paragraph)
        index = 0
        while index < len(sentences):
            candidate = _normalize_text(sentences[index])
            cursor = index + 1
            while len(candidate.split()) < min_words and cursor < len(sentences):
                combined = _normalize_text(candidate + " " + sentences[cursor])
                if len(combined.split()) > max_words:
                    break
                candidate = combined
                cursor += 1
            words = candidate.split()
            index = max(index + 1, cursor)
            if not min_words <= len(words) <= max_words:
                continue
            if not re.search(r"[.!?][\"”’']?$", candidate):
                continue
            if re.search(r"https?://|www\.|\[[0-9]+\]", candidate):
                continue
            digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            passages.append(candidate)
    return passages


def extract_documentation_passages(
    document: str, *, min_words: int = 25, max_words: int = 100
) -> list[str]:
    """Extract normalized prose paragraphs from Markdown/reStructuredText docs."""

    if min_words < 1 or max_words < min_words:
        raise ValueError("word limits must satisfy 1 <= min_words <= max_words")
    text = document.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?ms)^```.*?^```\s*$", "\n", text)
    text = re.sub(r"(?ms)^::\s*$\n(?:\s{2,}.*\n?)+", "\n", text)
    passages: list[str] = []
    seen: set[str] = set()
    for block in re.split(r"\n\s*\n", text):
        stripped = block.strip()
        if not stripped:
            continue
        lines = stripped.splitlines()
        first = lines[0].lstrip()
        if first.startswith(("#", ".. ", ":::", "- ", "* ", "+ ", "|", "<", ">")):
            continue
        if len(lines) > 1 and set(lines[1].strip()) <= {"=", "-", "~", "^"}:
            continue
        if any(line.startswith(("    ", "\t")) for line in lines):
            continue
        prose = " ".join(line.strip() for line in lines)
        prose = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", prose)
        prose = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", prose)
        prose = re.sub(r":\w+:`([^`]+)`", r"\1", prose)
        prose = re.sub(r"``([^`]+)``|`([^`]+)`", lambda match: match.group(1) or match.group(2), prose)
        prose = re.sub(r"\*\*([^*]+)\*\*|\*([^*]+)\*", lambda match: match.group(1) or match.group(2), prose)
        prose = re.sub(r"\s+:[a-zA-Z0-9_-]+:`[^`]+`", "", prose)
        prose = _normalize_text(prose)
        words = prose.split()
        if not min_words <= len(words) <= max_words:
            continue
        if sum(character.isalpha() for character in prose) < len(prose) * 0.65:
            continue
        if not re.search(r"[.!?][\"”’')\]]?$", prose):
            continue
        digest = hashlib.sha256(prose.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        passages.append(prose)
    return passages


def _canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in DATASET_FIELDS}


def dataset_digest(rows: Sequence[dict[str, Any]]) -> str:
    """Return a stable digest independent of CSV/dict row order."""

    canonical = sorted(
        (_canonical_row(row) for row in rows),
        key=lambda row: (str(row["example_id"]), str(row["pair_id"])),
    )
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean")


def validate_pairs(rows: Sequence[dict[str, Any]]) -> None:
    """Validate balance, complete semantic pairs, provenance, and uniqueness."""

    if not rows:
        raise ValueError("dataset must contain at least one pair")

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    example_ids: set[str] = set()
    text_hashes: set[str] = set()
    for raw_row in rows:
        row = _canonical_row(raw_row)
        missing = [field for field in ("pair_id", "example_id", "text") if not str(row[field]).strip()]
        if missing:
            raise ValueError(f"row is missing required values: {', '.join(missing)}")
        if row["example_id"] in example_ids:
            raise ValueError(f"duplicate example_id: {row['example_id']}")
        example_ids.add(str(row["example_id"]))

        text_hash = hashlib.sha256(_normalize_text(str(row["text"])).encode("utf-8")).hexdigest()
        if text_hash in text_hashes:
            raise ValueError(f"duplicate text detected for {row['example_id']}")
        text_hashes.add(text_hash)

        is_ai = _as_bool(row["is_ai"])
        for field in ("source_id", "source_url", "source_title", "source_author", "license"):
            if not str(row[field]).strip():
                raise ValueError(f"{row['example_id']} is missing {field}")
        if is_ai:
            if not str(row["generation_model"]).strip():
                raise ValueError(f"{row['example_id']} is missing generation_model")
            if not str(row["parent_example_id"]).strip():
                raise ValueError(f"{row['example_id']} is missing parent_example_id")
        by_pair[str(row["pair_id"])].append({**row, "is_ai": is_ai})

    for pair_id, pair in by_pair.items():
        if len(pair) != 2:
            raise ValueError(f"{pair_id} must contain exactly two rows")
        labels = {row["is_ai"] for row in pair}
        if labels != {False, True}:
            raise ValueError(f"{pair_id} must contain one human and one AI row")
        human = next(row for row in pair if not row["is_ai"])
        ai = next(row for row in pair if row["is_ai"])
        if ai["parent_example_id"] != human["example_id"]:
            raise ValueError(f"{pair_id} AI parent_example_id must reference its human row")
        if ai["source_id"] != human["source_id"]:
            raise ValueError(f"{pair_id} source_id differs across the pair")


def split_pair_ids(pair_ids: Sequence[str], *, seed: int = 42) -> dict[str, list[str]]:
    """Create a deterministic 50/25/25 grouped split from unique pair IDs."""

    normalized = sorted(str(pair_id) for pair_id in pair_ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError("pair IDs must be unique")
    if len(normalized) < 4:
        raise ValueError("at least four pairs are required for a 50/25/25 split")

    random.Random(seed).shuffle(normalized)
    train_end = len(normalized) // 2
    validation_end = train_end + len(normalized) // 4
    return {
        "train": normalized[:train_end],
        "validation": normalized[train_end:validation_end],
        "test": normalized[validation_end:],
    }


def baseline_accuracy_by_split(
    selected: Sequence[dict[str, Any]], splits: dict[str, list[str]]
) -> dict[str, dict[str, Any]]:
    """Summarize baseline pair correctness for frozen pair-grouped partitions."""

    correctness = {str(item["pair_id"]): int(item["pair_correct"]) for item in selected}
    metrics: dict[str, dict[str, Any]] = {}
    for split_name, pair_ids in splits.items():
        missing = [pair_id for pair_id in pair_ids if pair_id not in correctness]
        if missing:
            raise ValueError(f"baseline predictions missing pair IDs: {missing}")
        correct = sum(correctness[pair_id] for pair_id in pair_ids)
        row_count = 2 * len(pair_ids)
        metrics[split_name] = {
            "pair_count": len(pair_ids),
            "row_count": row_count,
            "correct": correct,
            "accuracy": 100 * correct / row_count,
        }
    return metrics


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_sources(path: Path = SOURCES_PATH) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources = data.get("sources", []) if isinstance(data, dict) else []
    required = {"id", "title", "author", "url", "catalog_url", "license"}
    for source in sources:
        missing = required - source.keys()
        if missing:
            raise ValueError(f"source {source.get('id', '<unknown>')} missing {sorted(missing)}")
        if "public domain" not in str(source["license"]).lower():
            raise ValueError(f"source {source['id']} is not explicitly marked public domain")
    if not sources:
        raise ValueError("source manifest is empty")
    return sources


def _load_source_manifest(path: Path = SOURCES_PATH) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("source manifest must be a mapping")
    return data


def _urlopen_json(params: dict[str, Any]) -> dict[str, Any]:
    url = "https://en.wikipedia.org/w/api.php?" + urlencode(params)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "context-engineering-dspy-book/1.0 (research benchmark)"},
    )
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            time.sleep(0.75)
            return payload
        except HTTPError as exc:
            if exc.code != 429 or attempt == 1:
                raise
            retry_after = float(exc.headers.get("Retry-After", "5") or 5)
            time.sleep(min(max(retry_after, 2.0), 15.0))
    raise AssertionError("bounded request loop did not return or raise")


_AI_ASSOCIATED_PHRASES = (
    "in conclusion",
    "moreover",
    "furthermore",
    "not only",
    "plays a role",
    "key role",
    "wide range",
    "complex",
    "dynamic",
    "comprehensive",
    "multifaceted",
    "interconnected",
    "framework",
    "holistic",
    "landscape",
    "foster",
    "enhance",
    "facilitate",
    "significant",
    "various",
)


def _ai_like_score(text: str) -> tuple[int, int, int]:
    lowered = text.lower()
    phrase_hits = sum(lowered.count(phrase) for phrase in _AI_ASSOCIATED_PHRASES)
    connective_hits = len(re.findall(r"(?i)\b(however|therefore|additionally|consequently|thus)\b", text))
    # Prefer passages near 70 words, which are self-contained but leave less stylistic evidence.
    length_score = -abs(len(text.split()) - 70)
    return phrase_hits, connective_hits, length_score


def collect_wikipedia_candidates(*, per_page: int = 10) -> list[dict[str, Any]]:
    """Collect human-edited passages from revisions fixed before the LLM era."""

    manifest = _load_source_manifest()
    cutoff = str(manifest["wikipedia_cutoff"])
    license_name = str(manifest["wikipedia_license"])
    pages = list(manifest["wikipedia_pages"])
    candidates = _read_csv(WIKIPEDIA_CANDIDATES_PATH) if WIKIPEDIA_CANDIDATES_PATH.exists() else []
    sources_checkpoint = RESULTS_DIR / "wikipedia_sources.json"
    if sources_checkpoint.exists():
        source_records = list(json.loads(sources_checkpoint.read_text(encoding="utf-8")).get("sources", []))
    else:
        source_records = []
    completed_pages = {record["page_title"] for record in source_records}
    for page_title in pages:
        if page_title in completed_pages:
            continue
        revision_payload = _urlopen_json(
            {
                "action": "query",
                "prop": "revisions",
                "titles": page_title,
                "rvprop": "ids|timestamp",
                "rvstart": cutoff,
                "rvdir": "older",
                "rvlimit": 1,
                "format": "json",
                "formatversion": 2,
            }
        )
        page = revision_payload["query"]["pages"][0]
        revisions = page.get("revisions") or []
        if not revisions:
            continue
        revision = revisions[0]
        revision_id = int(revision["revid"])
        parsed = _urlopen_json(
            {
                "action": "parse",
                "oldid": revision_id,
                "prop": "text",
                "format": "json",
                "formatversion": 2,
            }
        )
        passages = extract_html_passages(parsed["parse"]["text"])
        passages.sort(key=lambda text: (_ai_like_score(text), text), reverse=True)
        slug = re.sub(r"[^a-z0-9]+", "-", page_title.lower()).strip("-")
        old_url = f"https://en.wikipedia.org/w/index.php?oldid={revision_id}"
        source_records.append(
            {
                "page_title": page_title,
                "revision_id": revision_id,
                "revision_timestamp": revision["timestamp"],
                "url": old_url,
                "license": license_name,
            }
        )
        for index, text in enumerate(passages[:per_page], start=1):
            pair_id = f"wiki-{slug}-{revision_id}-{index:02d}"
            candidates.append(
                {
                    "pair_id": pair_id,
                    "example_id": f"{pair_id}-human",
                    "text": text,
                    "is_ai": False,
                    "notes": (
                        "Exact paragraph from a Wikipedia revision dated before 2020; "
                        "selected for AI-associated prose characteristics."
                    ),
                    "source_id": f"wikipedia-{revision_id}",
                    "source_url": old_url,
                    "source_title": page_title,
                    "source_author": "Wikipedia contributors (see revision history)",
                    "license": license_name,
                    "generation_model": "",
                    "parent_example_id": "",
                }
            )
        # Checkpoint after every page; a source-side 429 must not discard prior work.
        _write_csv(WIKIPEDIA_CANDIDATES_PATH, candidates, DATASET_FIELDS)
        sources_checkpoint.write_text(
            json.dumps(
                {
                    "cutoff": cutoff,
                    "license": license_name,
                    "license_url": "https://en.wikipedia.org/wiki/Wikipedia:Copyrights",
                    "sources": source_records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    _write_csv(HUMAN_CANDIDATES_PATH, candidates, DATASET_FIELDS)
    return candidates


def collect_human_candidates(*, per_source: int = 14, seed: int = 42) -> list[dict[str, Any]]:
    """Download source texts and save deterministic human candidate excerpts."""

    candidates: list[dict[str, Any]] = []
    for source in _load_sources():
        request = urllib.request.Request(
            source["url"],
            headers={"User-Agent": "context-engineering-dspy-book/1.0 (research benchmark)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8-sig", errors="replace")
        header = raw[:2_500]
        if re.search(r"(?i)copyrighted|posted with permission", header):
            raise ValueError(f"source {source['id']} contains an embedded copyright restriction")
        passages = extract_passages(raw)
        source_rng = random.Random(f"{seed}:{source['id']}")
        source_rng.shuffle(passages)
        for index, text in enumerate(passages[:per_source], start=1):
            pair_id = f"{source['id']}-{index:03d}"
            candidates.append(
                {
                    "pair_id": pair_id,
                    "example_id": f"{pair_id}-human",
                    "text": text,
                    "is_ai": False,
                    "notes": "Exact excerpt from a public-domain human-authored source.",
                    "source_id": source["id"],
                    "source_url": source["catalog_url"],
                    "source_title": source["title"],
                    "source_author": source["author"],
                    "license": source["license"],
                    "generation_model": "",
                    "parent_example_id": "",
                }
            )
    _write_csv(HUMAN_CANDIDATES_PATH, candidates, DATASET_FIELDS)
    return candidates


def collect_short_human_candidates(*, per_source: int = 20) -> list[dict[str, Any]]:
    """Collect concise public-domain passages ranked for AI-like phrasing."""

    candidates: list[dict[str, Any]] = []
    for source in _load_sources():
        request = urllib.request.Request(
            source["url"],
            headers={"User-Agent": "context-engineering-dspy-book/1.0 (research benchmark)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8-sig", errors="replace")
        header = raw[:2_500]
        if re.search(r"(?i)copyrighted|posted with permission", header):
            raise ValueError(f"source {source['id']} contains an embedded copyright restriction")
        passages = extract_short_passages(raw)
        passages.sort(key=lambda text: (_ai_like_score(text), text), reverse=True)
        for index, text in enumerate(passages[:per_source], start=1):
            pair_id = f"{source['id']}-short-{index:03d}"
            candidates.append(
                {
                    "pair_id": pair_id,
                    "example_id": f"{pair_id}-human",
                    "text": text,
                    "is_ai": False,
                    "notes": (
                        "Exact short passage from a public-domain human-authored source; "
                        "selected for AI-associated prose characteristics."
                    ),
                    "source_id": source["id"],
                    "source_url": source["catalog_url"],
                    "source_title": source["title"],
                    "source_author": source["author"],
                    "license": source["license"],
                    "generation_model": "",
                    "parent_example_id": "",
                }
            )
    _write_csv(HUMAN_CANDIDATES_PATH, candidates, DATASET_FIELDS)
    return candidates


def collect_documentation_candidates(*, per_source: int = 18) -> list[dict[str, Any]]:
    """Collect modern human technical prose frozen at pre-2022 repository tags."""

    manifest = _load_source_manifest()
    sources = list(manifest.get("documentation_sources", []))
    candidates: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for source in sources:
        request = urllib.request.Request(
            source["raw_url"],
            headers={"User-Agent": "context-engineering-dspy-book/1.0 (research benchmark)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            document = response.read().decode("utf-8-sig", errors="replace")
        passages = extract_documentation_passages(document)
        passages.sort(key=lambda text: (_ai_like_score(text), text), reverse=True)
        source_records.append({key: source[key] for key in source})
        for index, text in enumerate(passages[:per_source], start=1):
            pair_id = f"docs-{source['id']}-{source['frozen_ref']}-{index:03d}"
            candidates.append(
                {
                    "pair_id": pair_id,
                    "example_id": f"{pair_id}-human",
                    "text": text,
                    "is_ai": False,
                    "notes": (
                        "Human-authored technical documentation frozen at a pre-2022 "
                        "open-source release tag; markup normalized to prose."
                    ),
                    "source_id": source["id"],
                    "source_url": source["source_url"],
                    "source_title": source["title"],
                    "source_author": source["author"],
                    "license": source["license"],
                    "generation_model": "",
                    "parent_example_id": "",
                }
            )
    _write_csv(HUMAN_CANDIDATES_PATH, candidates, DATASET_FIELDS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "documentation_sources.json").write_text(
        json.dumps({"sources": source_records}, indent=2) + "\n",
        encoding="utf-8",
    )
    return candidates


def _rewrite_instruction(human_text: str, attempt: int) -> str:
    variants = (
        "Keep a few idiosyncratic or old-fashioned turns of phrase, but add polished transitions and balanced structure.",
        "Preserve concrete details and uneven sentence rhythm while adding subtle summary language and orderly progression.",
        "Make the rewrite feel personally authored and specific, yet include restrained AI-associated tells such as parallel phrasing.",
        "Retain uneven human rhythm and concrete terminology, using only one restrained AI-associated tell: a balanced contrast.",
        "Avoid headings and generic conclusions; keep the source's specificity while adding a single smooth signpost between ideas.",
        "Use uneven sentence rhythm, one natural contraction or incidental aside, and one restrained parallel construction; avoid stock summary phrases.",
        "Keep an idiosyncratic human voice and one minor grammatical rough edge, while adding one subtle transition that makes the logic more orderly.",
    )
    source_words = len(human_text.split())
    minimum_words = max(25, round(source_words * 0.85))
    maximum_words = max(minimum_words + 5, round(source_words * 1.15))
    return (
        "Rewrite the passage from scratch while preserving its meaning and factual claims. "
        f"Keep the result to {minimum_words}-{maximum_words} words so its length does not reveal the label. "
        "Do not mention this instruction, AI, language models, or authorship. "
        f"{variants[attempt % len(variants)]} Return only the rewritten passage.\n\n"
        f"SOURCE PASSAGE:\n{human_text}"
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def generate_rewrites(
    *,
    model: str,
    attempts_per_passage: int = 3,
    limit: int = 0,
    max_stage_cost_usd: float = 8.0,
    split_name: str = "",
) -> None:
    """Generate and checkpoint multiple rewrite attempts for each human candidate."""

    import dspy
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY in the repo's .env file")
    ledger = BudgetLedger()
    ledger.assert_can_spend(min(max_stage_cost_usd, 0.05))
    lm = dspy.LM(
        model,
        num_retries=int(os.getenv("CHAPTER06_NUM_RETRIES", "1")),
        cache=True,
        timeout=float(os.getenv("CHAPTER06_REQUEST_TIMEOUT_SECONDS", "120")),
    )
    candidates = _read_csv(HUMAN_CANDIDATES_PATH)
    if HUMAN_PREDICTIONS_PATH.exists():
        difficult_ids = {
            record["pair_id"]
            for record in (
                json.loads(line)
                for line in HUMAN_PREDICTIONS_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if record.get("status") == "completed" and record.get("predicted_is_ai") is True
        }
        candidates = [candidate for candidate in candidates if candidate["pair_id"] in difficult_ids]
    quality_rejections = [
        {
            "pair_id": candidate["pair_id"],
            "example_id": candidate["example_id"],
            "reasons": quality_rejection_reasons(candidate["text"]),
        }
        for candidate in candidates
        if quality_rejection_reasons(candidate["text"])
    ]
    candidates = [candidate for candidate in candidates if not quality_rejection_reasons(candidate["text"])]
    if split_name:
        candidate_splits = split_pair_ids([candidate["pair_id"] for candidate in candidates], seed=42)
        selected_ids = set(candidate_splits[split_name])
        candidates = [candidate for candidate in candidates if candidate["pair_id"] in selected_ids]
    QUALITY_REJECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUALITY_REJECTIONS_PATH.open("w", encoding="utf-8") as handle:
        for rejection in quality_rejections:
            handle.write(json.dumps(rejection, ensure_ascii=False) + "\n")
    completed: set[tuple[str, int]] = set()
    if REWRITE_ATTEMPTS_PATH.exists():
        for line in REWRITE_ATTEMPTS_PATH.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("status") == "completed":
                completed.add((record["pair_id"], int(record["attempt"])))

    run_id = f"rewrite-generation-{time.time_ns()}"
    run_status = "completed"
    try:
        for candidate in candidates[: limit or None]:
            for attempt in range(attempts_per_passage):
                key = (candidate["pair_id"], attempt)
                if key in completed:
                    continue
                current_cost = summarize_history(lm.history)["cost_usd"]
                if current_cost >= max_stage_cost_usd:
                    raise RuntimeError(
                        f"rewrite generation reached its ${max_stage_cost_usd:.2f} stage cap"
                    )
                ledger.assert_can_spend(current_cost + 0.05)
                before = len(lm.history)
                started = time.monotonic()
                try:
                    output = lm(_rewrite_instruction(candidate["text"], attempt))[0]
                    text = _normalize_text(str(output))
                    status = "completed"
                    error = ""
                except Exception as exc:
                    text = ""
                    status = classify_api_error(exc)
                    error = f"{type(exc).__name__}: {exc}"
                history = lm.history[before:]
                _append_jsonl(
                    REWRITE_ATTEMPTS_PATH,
                    {
                        "pair_id": candidate["pair_id"],
                        "attempt": attempt,
                        "model": model,
                        "status": status,
                        "text": text,
                        "error": error,
                        "elapsed_seconds": time.monotonic() - started,
                        "usage": [entry.get("usage", {}) for entry in history],
                        "cost": sum(float(entry.get("cost") or 0) for entry in history),
                    },
                )
                if is_global_stop_status(status):
                    raise RuntimeError(error)
    except Exception as exc:
        run_status = classify_api_error(exc)
        raise
    finally:
        summary = summarize_history(lm.history)
        ledger.record(
            stage="dataset_generation",
            run_id=run_id,
            cost_usd=summary["cost_usd"],
            metadata={
                "model": model,
                "attempts_per_passage": attempts_per_passage,
                "status": run_status,
                "summary": summary,
            },
        )


def _detector_components(
    model: str,
    *,
    num_threads: int = 4,
    cache: bool = True,
    timeout: float = 120,
    num_retries: int = 1,
):
    import dspy

    class DetectAIText(dspy.Signature):
        """Decide whether the supplied passage was generated by an AI."""

        text: str = dspy.InputField(desc="Passage to classify")
        is_ai: bool = dspy.OutputField(desc="True for AI-generated text; otherwise False")

    class AIDetector(dspy.Module):
        def __init__(self):
            super().__init__()
            self.detect = dspy.ChainOfThought(DetectAIText)

        def forward(self, text: str):
            return self.detect(text=text)

    lm = dspy.LM(
        model,
        num_retries=num_retries,
        cache=cache,
        timeout=timeout,
    )
    detector = AIDetector()
    detector.set_lm(lm)
    return detector, lm, num_threads


def _predict_bool(detector: Any, text: str) -> bool:
    return _as_bool(detector(text=text).is_ai)


def screen_human_candidates(*, task_model: str, limit: int = 0) -> dict[str, Any]:
    """Evaluate real human passages and checkpoint baseline false positives."""

    ledger = BudgetLedger()
    ledger.assert_can_spend(0.05)
    detector, lm, _ = _detector_components(task_model)
    previous: dict[str, dict[str, Any]] = {}
    if HUMAN_PREDICTIONS_PATH.exists():
        previous = {
            record["pair_id"]: record
            for record in (
                json.loads(line)
                for line in HUMAN_PREDICTIONS_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if record.get("status") == "completed"
        }

    records: list[dict[str, Any]] = []
    run_id = f"human-baseline-{time.time_ns()}"
    run_status = "completed"
    try:
        for candidate in _read_csv(HUMAN_CANDIDATES_PATH)[: limit or None]:
            if candidate["pair_id"] in previous:
                records.append(previous[candidate["pair_id"]])
                continue
            ledger.assert_can_spend(summarize_history(lm.history)["cost_usd"] + 0.05)
            started = time.monotonic()
            try:
                prediction = _predict_bool(detector, candidate["text"])
                record = {
                    "pair_id": candidate["pair_id"],
                    "example_id": candidate["example_id"],
                    "predicted_is_ai": prediction,
                    "expected_is_ai": False,
                    "correct": not prediction,
                    "status": "completed",
                    "elapsed_seconds": time.monotonic() - started,
                }
            except Exception as exc:
                status = classify_api_error(exc)
                record = {
                    "pair_id": candidate["pair_id"],
                    "example_id": candidate["example_id"],
                    "status": status,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.monotonic() - started,
                }
                records.append(record)
                if is_global_stop_status(status):
                    raise
                continue
            records.append(record)
    except Exception as exc:
        run_status = classify_api_error(exc)
        raise
    finally:
        HUMAN_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with HUMAN_PREDICTIONS_PATH.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = summarize_history(lm.history)
        ledger.record(
            stage="dataset_baseline_screen",
            run_id=run_id,
            cost_usd=summary["cost_usd"],
            metadata={"model": task_model, "status": run_status, "summary": summary},
        )
    completed = [record for record in records if record.get("status") == "completed"]
    if not completed:
        raise RuntimeError("no human baseline predictions completed")
    false_positives = [record for record in completed if record["predicted_is_ai"] is True]
    result = {
        "task_model": task_model,
        "completed": len(completed),
        "false_positives": len(false_positives),
        "human_accuracy": 100 * (len(completed) - len(false_positives)) / len(completed),
        "cost": summary,
    }
    _append_jsonl(ADVERSARIAL_ROUNDS_PATH, {"stage": "human_screen", **result})
    return result


def screen_pairs(*, task_model: str, target_pairs: int = 100) -> list[dict[str, Any]]:
    """Choose the lowest-accuracy complete pairs under the unchanged baseline."""

    humans = {row["pair_id"]: row for row in _read_csv(HUMAN_CANDIDATES_PATH)}
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in REWRITE_ATTEMPTS_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") == "completed" and record.get("text"):
            attempts[record["pair_id"]].append(record)

    ledger = BudgetLedger()
    ledger.assert_can_spend(0.05)
    detector, lm, _ = _detector_components(task_model)
    saved_human_predictions = {
        record["pair_id"]: record
        for record in (
            json.loads(line)
            for line in HUMAN_PREDICTIONS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if record.get("status") == "completed"
    }
    saved_ai_predictions: dict[tuple[str, int], dict[str, Any]] = {}
    if PAIR_PREDICTIONS_PATH.exists():
        saved_ai_predictions = {
            (record["pair_id"], int(record["attempt"])): record
            for record in (
                json.loads(line)
                for line in PAIR_PREDICTIONS_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if record.get("status") == "completed"
        }
    scored: list[tuple[int, str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    run_id = f"pair-screen-{time.time_ns()}"
    run_status = "completed"
    try:
        for pair_id in sorted(set(humans) & set(attempts)):
            human = humans[pair_id]
            saved_human = saved_human_predictions.get(pair_id)
            if saved_human is None:
                ledger.assert_can_spend(summarize_history(lm.history)["cost_usd"] + 0.05)
                human_prediction = _predict_bool(detector, human["text"])
            else:
                human_prediction = bool(saved_human["predicted_is_ai"])
            best: tuple[int, dict[str, Any], bool] | None = None
            for attempt in sorted(attempts[pair_id], key=lambda item: int(item["attempt"])):
                attempt_number = int(attempt["attempt"])
                saved_ai = saved_ai_predictions.get((pair_id, attempt_number))
                if saved_ai is None:
                    ledger.assert_can_spend(summarize_history(lm.history)["cost_usd"] + 0.05)
                    started = time.monotonic()
                    try:
                        ai_prediction = _predict_bool(detector, attempt["text"])
                        record = {
                            "pair_id": pair_id,
                            "attempt": attempt_number,
                            "predicted_is_ai": ai_prediction,
                            "expected_is_ai": True,
                            "status": "completed",
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    except Exception as exc:
                        status = classify_api_error(exc)
                        record = {
                            "pair_id": pair_id,
                            "attempt": attempt_number,
                            "status": status,
                            "error": f"{type(exc).__name__}: {exc}",
                            "elapsed_seconds": time.monotonic() - started,
                        }
                        _append_jsonl(PAIR_PREDICTIONS_PATH, record)
                        if is_global_stop_status(status):
                            raise
                        continue
                    _append_jsonl(PAIR_PREDICTIONS_PATH, record)
                else:
                    ai_prediction = bool(saved_ai["predicted_is_ai"])
                correct = int(not human_prediction) + int(ai_prediction)
                if best is None or correct < best[0]:
                    best = (correct, attempt, ai_prediction)
                if correct == 0:
                    break
            if best is None:
                continue
            correct, attempt, ai_prediction = best
            ai = {
                **human,
                "example_id": f"{pair_id}-ai",
                "text": attempt["text"],
                "is_ai": True,
                "notes": "Model-generated semantic rewrite of the paired licensed passage.",
                "generation_model": attempt["model"],
                "parent_example_id": human["example_id"],
            }
            metadata = {
                "pair_id": pair_id,
                "human_prediction": human_prediction,
                "ai_prediction": ai_prediction,
                "pair_correct": correct,
                "selected_attempt": attempt["attempt"],
            }
            scored.append((correct, pair_id, human, ai, metadata))
    except Exception as exc:
        run_status = classify_api_error(exc)
        raise
    finally:
        history_summary = summarize_history(lm.history)
        ledger.record(
            stage="dataset_pair_screen",
            run_id=run_id,
            cost_usd=history_summary["cost_usd"],
            metadata={"model": task_model, "status": run_status, "summary": history_summary},
        )

    if not scored:
        raise RuntimeError("no complete quality-screened pairs are available")
    selection_target = min(target_pairs, len(scored))

    # Prevent a single work from dominating while still preferring inverted pairs.
    source_count = max(1, len({item[2]["source_id"] for item in scored}))
    per_source_cap = max(2, math.ceil(selection_target / source_count) + 2)
    selected: list[tuple[int, str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    selected_ids: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    for item in sorted(scored, key=lambda value: (value[0], value[1])):
        source_id = item[2]["source_id"]
        if counts[source_id] >= per_source_cap:
            continue
        selected.append(item)
        selected_ids.add(item[1])
        counts[source_id] += 1
        if len(selected) == selection_target:
            break
    if len(selected) < target_pairs:
        for item in sorted(scored, key=lambda value: (value[0], value[1])):
            if item[1] in selected_ids:
                continue
            selected.append(item)
            if len(selected) == selection_target:
                break

    rows = [row for item in selected for row in (item[2], item[3])]
    validate_pairs(rows)
    _write_csv(SCREENED_PAIRS_PATH, rows, DATASET_FIELDS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "screening.json").write_text(
        json.dumps(
            {
                "task_model": task_model,
                "requested_target_pairs": target_pairs,
                "actual_pairs": len(selected),
                "selected": [item[4] for item in selected],
                "baseline_accuracy": 100 * sum(item[0] for item in selected) / (2 * len(selected)),
                "lm_cost": sum(float(entry.get("cost") or 0) for entry in lm.history),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _append_jsonl(
        ADVERSARIAL_ROUNDS_PATH,
        {
            "stage": "pair_screen",
            "task_model": task_model,
            "selected_pairs": len(selected),
            "baseline_accuracy": 100 * sum(item[0] for item in selected) / (2 * len(selected)),
            "cost": history_summary,
        },
    )
    return rows


def run_baseline_stability_screen(
    *,
    task_model: str,
    repeats: int = 2,
    timeout: float = 120,
) -> dict[str, Any]:
    """Run uncached baseline replicates over every selected row, resuming safely."""

    if repeats < 2:
        raise ValueError("stability screening requires at least two repeats")
    rows = _read_csv(SCREENED_PAIRS_PATH)
    validate_pairs(rows)
    completed: dict[tuple[int, str], dict[str, Any]] = {}
    if STABILITY_PREDICTIONS_PATH.exists():
        for line in STABILITY_PREDICTIONS_PATH.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("status") == "completed":
                completed[(int(record["repeat"]), str(record["example_id"]))] = record

    ledger = BudgetLedger()
    ledger.assert_can_spend(0.05)
    detector, lm, _ = _detector_components(
        task_model,
        cache=False,
        timeout=timeout,
        num_retries=1,
    )
    run_id = f"baseline-stability-{time.time_ns()}"
    run_status = "completed"
    try:
        for repeat in range(repeats):
            for row in rows:
                key = (repeat, str(row["example_id"]))
                if key in completed:
                    continue
                ledger.assert_can_spend(summarize_history(lm.history)["cost_usd"] + 0.05)
                started = time.monotonic()
                try:
                    prediction = _predict_bool(detector, str(row["text"]))
                    record = {
                        "repeat": repeat,
                        "pair_id": row["pair_id"],
                        "example_id": row["example_id"],
                        "expected_is_ai": _as_bool(row["is_ai"]),
                        "predicted_is_ai": prediction,
                        "correct": prediction == _as_bool(row["is_ai"]),
                        "status": "completed",
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    completed[key] = record
                except Exception as exc:
                    status = classify_api_error(exc)
                    record = {
                        "repeat": repeat,
                        "pair_id": row["pair_id"],
                        "example_id": row["example_id"],
                        "status": status,
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    _append_jsonl(STABILITY_PREDICTIONS_PATH, record)
                    raise
                _append_jsonl(STABILITY_PREDICTIONS_PATH, record)
    except BaseException as exc:
        run_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else classify_api_error(exc)
        raise
    finally:
        summary = summarize_history(lm.history)
        ledger.record(
            stage="dataset_baseline_stability",
            run_id=run_id,
            cost_usd=summary["cost_usd"],
            metadata={
                "model": task_model,
                "repeats": repeats,
                "timeout": timeout,
                "status": run_status,
                "summary": summary,
            },
        )

    row_count = len(rows)
    repeat_metrics = []
    for repeat in range(repeats):
        records = [
            record for (record_repeat, _), record in completed.items() if record_repeat == repeat
        ]
        if len(records) != row_count:
            raise RuntimeError(f"repeat {repeat} is incomplete: {len(records)}/{row_count} rows")
        correct = sum(bool(record["correct"]) for record in records)
        repeat_metrics.append(
            {"repeat": repeat, "correct": correct, "row_count": row_count, "accuracy": 100 * correct / row_count}
        )
    result = {
        "task_model": task_model,
        "repeats": repeats,
        "row_count": row_count,
        "repeat_metrics": repeat_metrics,
        "mean_accuracy": sum(item["accuracy"] for item in repeat_metrics) / repeats,
        "cost": summary,
    }
    (RESULTS_DIR / "baseline_stability_screening.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _append_jsonl(ADVERSARIAL_ROUNDS_PATH, {"stage": "baseline_stability", **result})
    return result


def stable_adversarial_splits(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    *,
    repeats: int,
    seed: int = 42,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Select the test partition by worst-case repeated baseline pair accuracy."""

    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    source_by_pair = {str(row["pair_id"]): str(row["source_id"]) for row in rows}
    by_key = {
        (int(record["repeat"]), str(record["example_id"])): record
        for record in predictions
        if record.get("status") == "completed"
    }
    example_ids_by_pair: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        example_ids_by_pair[str(row["pair_id"])].append(str(row["example_id"]))

    scored: list[tuple[int, int, str, list[int]]] = []
    for pair_id in pair_ids:
        per_repeat: list[int] = []
        for repeat in range(repeats):
            records = [by_key.get((repeat, example_id)) for example_id in example_ids_by_pair[pair_id]]
            if any(record is None for record in records):
                raise ValueError(f"stability predictions missing repeat {repeat} for {pair_id}")
            per_repeat.append(sum(bool(record["correct"]) for record in records if record))
        scored.append((max(per_repeat), sum(per_repeat), pair_id, per_repeat))

    conventional = split_pair_ids(pair_ids, seed=seed)
    test_count = len(conventional["test"])
    source_count = max(1, len(set(source_by_pair.values())))
    per_source_cap = max(2, math.ceil(test_count / source_count) + 1)
    selected_test: list[str] = []
    source_counts: dict[str, int] = defaultdict(int)
    for _, _, pair_id, _ in sorted(scored):
        source_id = source_by_pair[pair_id]
        if source_counts[source_id] >= per_source_cap:
            continue
        selected_test.append(pair_id)
        source_counts[source_id] += 1
        if len(selected_test) == test_count:
            break
    if len(selected_test) < test_count:
        for _, _, pair_id, _ in sorted(scored):
            if pair_id not in selected_test:
                selected_test.append(pair_id)
            if len(selected_test) == test_count:
                break

    remaining = sorted(set(pair_ids) - set(selected_test))
    random.Random(seed).shuffle(remaining)
    train_count = len(pair_ids) // 2
    splits = {
        "train": remaining[:train_count],
        "validation": remaining[train_count:],
        "test": selected_test,
    }
    selected_metadata = [
        {
            "pair_id": pair_id,
            "source_id": source_by_pair[pair_id],
            "correct_by_repeat": per_repeat,
            "worst_repeat_correct": max(per_repeat),
        }
        for _, _, pair_id, per_repeat in scored
        if pair_id in set(selected_test)
    ]
    return splits, selected_metadata


def freeze_stable_dataset(*, task_model: str, repeats: int = 2, seed: int = 42) -> dict[str, Any]:
    """Freeze a baseline-adversarial split only when every repeat is below chance."""

    rows = _read_csv(SCREENED_PAIRS_PATH)
    validate_pairs(rows)
    predictions = [
        json.loads(line)
        for line in STABILITY_PREDICTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    splits, selected = stable_adversarial_splits(rows, predictions, repeats=repeats, seed=seed)
    prediction_by_key = {
        (int(record["repeat"]), str(record["example_id"])): record
        for record in predictions
        if record.get("status") == "completed"
    }
    test_rows = [row for row in rows if row["pair_id"] in set(splits["test"])]
    repeat_metrics = []
    for repeat in range(repeats):
        records = [prediction_by_key[(repeat, str(row["example_id"]))] for row in test_rows]
        correct = sum(bool(record["correct"]) for record in records)
        repeat_metrics.append(
            {"repeat": repeat, "correct": correct, "row_count": len(records), "accuracy": 100 * correct / len(records)}
        )
    if max(item["accuracy"] for item in repeat_metrics) >= 50.0:
        raise RuntimeError(
            "stable frozen test baseline gate failed: every repeat must be below 50%; "
            f"got {[item['accuracy'] for item in repeat_metrics]}"
        )

    digest = dataset_digest(rows)
    _write_csv(DATASET_PATH, rows, DATASET_FIELDS)
    manifest = {
        "schema_version": 2,
        "seed": seed,
        "split_strategy": "baseline_adversarial_worst_case_repeated",
        "dataset_path": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "dataset_sha256": digest,
        "row_count": len(rows),
        "pair_count": len({row["pair_id"] for row in rows}),
        "splits": splits,
    }
    SPLITS_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    gate = {
        "schema_version": 2,
        "task_model": task_model,
        "dataset_sha256": digest,
        "criterion": "every uncached test replicate accuracy < 50%",
        "passed": True,
        "repeats": repeats,
        "repeat_metrics": repeat_metrics,
        "mean_accuracy": sum(item["accuracy"] for item in repeat_metrics) / repeats,
        "selected_test_pairs": selected,
        "disclosure": "The test partition is deliberately selected using only baseline errors; it is an adversarial optimizer stress test, not an unbiased estimate of generalization.",
    }
    BASELINE_GATE_PATH.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return manifest


def freeze_dataset(*, seed: int = 42) -> dict[str, Any]:
    rows = _read_csv(SCREENED_PAIRS_PATH)
    validate_pairs(rows)
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    splits = split_pair_ids(pair_ids, seed=seed)
    digest = dataset_digest(rows)
    screening = json.loads((RESULTS_DIR / "screening.json").read_text(encoding="utf-8"))
    split_metrics = baseline_accuracy_by_split(screening["selected"], splits)
    if split_metrics["test"]["accuracy"] >= 50.0:
        raise RuntimeError(
            "frozen test baseline gate failed: expected accuracy below 50%, "
            f"got {split_metrics['test']['accuracy']:.2f}%"
        )
    _write_csv(DATASET_PATH, rows, DATASET_FIELDS)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "dataset_path": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "dataset_sha256": digest,
        "row_count": len(rows),
        "pair_count": len(pair_ids),
        "splits": splits,
    }
    SPLITS_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    BASELINE_GATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_GATE_PATH.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_model": screening["task_model"],
                "dataset_sha256": digest,
                "criterion": "test accuracy < 50%",
                "passed": True,
                "overall_accuracy": screening["baseline_accuracy"],
                "splits": split_metrics,
                "selected": screening["selected"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="download and extract human candidates")
    collect.add_argument("--per-source", type=int, default=14)

    collect_short = subparsers.add_parser(
        "collect-short", help="collect concise public-domain human passages"
    )
    collect_short.add_argument("--per-source", type=int, default=20)

    collect_docs = subparsers.add_parser(
        "collect-docs", help="collect pre-2022 human technical documentation"
    )
    collect_docs.add_argument("--per-source", type=int, default=18)

    collect_wikipedia = subparsers.add_parser(
        "collect-wikipedia", help="collect pre-2020 human-edited Wikipedia passages"
    )
    collect_wikipedia.add_argument("--per-page", type=int, default=10)

    generate = subparsers.add_parser("generate", help="generate AI rewrite attempts")
    generate.add_argument("--model", default=os.getenv("TASK_MODEL", "openai/gpt-5.6-luna"))
    generate.add_argument("--attempts", type=int, default=3)
    generate.add_argument("--limit", type=int, default=0)
    generate.add_argument("--max-stage-cost", type=float, default=8.0)
    generate.add_argument("--split", choices=("train", "validation", "test"), default="")

    human_screen = subparsers.add_parser(
        "screen-humans", help="find genuine human passages the baseline predicts as AI"
    )
    human_screen.add_argument(
        "--task-model", default=os.getenv("TASK_MODEL", "openai/gpt-5.6-luna")
    )
    human_screen.add_argument("--limit", type=int, default=0)

    screen = subparsers.add_parser("screen", help="screen and select adversarial pairs")
    screen.add_argument("--task-model", default=os.getenv("TASK_MODEL", "openai/gpt-5.6-luna"))
    screen.add_argument("--target-pairs", type=int, default=100)

    freeze = subparsers.add_parser("freeze", help="validate, split, and freeze the dataset")
    freeze.add_argument("--seed", type=int, default=42)
    stability = subparsers.add_parser(
        "stability-screen", help="run uncached baseline replicates over all selected pairs"
    )
    stability.add_argument(
        "--task-model", default=os.getenv("TASK_MODEL", "openai/gpt-5.6-luna")
    )
    stability.add_argument("--repeats", type=int, default=2)
    stability.add_argument("--timeout", type=float, default=120)
    stable_freeze = subparsers.add_parser(
        "freeze-stable", help="freeze a split that passes the repeated baseline gate"
    )
    stable_freeze.add_argument(
        "--task-model", default=os.getenv("TASK_MODEL", "openai/gpt-5.6-luna")
    )
    stable_freeze.add_argument("--repeats", type=int, default=2)
    stable_freeze.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        rows = collect_human_candidates(per_source=args.per_source)
        print(f"Collected {len(rows)} human candidates at {HUMAN_CANDIDATES_PATH}")
    elif args.command == "collect-short":
        rows = collect_short_human_candidates(per_source=args.per_source)
        print(f"Collected {len(rows)} short human candidates at {HUMAN_CANDIDATES_PATH}")
    elif args.command == "collect-docs":
        rows = collect_documentation_candidates(per_source=args.per_source)
        print(f"Collected {len(rows)} documentation candidates at {HUMAN_CANDIDATES_PATH}")
    elif args.command == "collect-wikipedia":
        rows = collect_wikipedia_candidates(per_page=args.per_page)
        print(f"Collected {len(rows)} pre-2020 Wikipedia candidates at {HUMAN_CANDIDATES_PATH}")
    elif args.command == "generate":
        generate_rewrites(
            model=args.model,
            attempts_per_passage=args.attempts,
            limit=args.limit,
            max_stage_cost_usd=args.max_stage_cost,
            split_name=args.split,
        )
        print(f"Rewrite attempts saved at {REWRITE_ATTEMPTS_PATH}")
    elif args.command == "screen-humans":
        print(
            json.dumps(
                screen_human_candidates(task_model=args.task_model, limit=args.limit),
                indent=2,
            )
        )
    elif args.command == "screen":
        rows = screen_pairs(task_model=args.task_model, target_pairs=args.target_pairs)
        print(f"Selected {len(rows) // 2} pairs at {SCREENED_PAIRS_PATH}")
    elif args.command == "freeze":
        manifest = freeze_dataset(seed=args.seed)
        print(json.dumps(manifest, indent=2))
    elif args.command == "stability-screen":
        print(
            json.dumps(
                run_baseline_stability_screen(
                    task_model=args.task_model,
                    repeats=args.repeats,
                    timeout=args.timeout,
                ),
                indent=2,
            )
        )
    elif args.command == "freeze-stable":
        print(
            json.dumps(
                freeze_stable_dataset(
                    task_model=args.task_model,
                    repeats=args.repeats,
                    seed=args.seed,
                ),
                indent=2,
            )
        )
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
