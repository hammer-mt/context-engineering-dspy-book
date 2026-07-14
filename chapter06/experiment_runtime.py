"""Reproducible accounting, redaction, and artifact helpers for Chapter 6."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "chapter06" / "results"
DEFAULT_LEDGER_PATH = RESULTS_ROOT / "budget_ledger.json"
DEFAULT_BUDGET_CEILING_USD = 95.0

_LOCK = threading.Lock()
_SECRET_KEY_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|access[_-]?token|secret|password|cookie)"
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/-]{12,}"),
)


class BudgetExceeded(RuntimeError):
    """Raised before a stage would cross the experiment's guarded ceiling."""


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(redact(value), indent=2, ensure_ascii=False, default=str) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


class BudgetLedger:
    def __init__(
        self,
        path: Path = DEFAULT_LEDGER_PATH,
        *,
        ceiling_usd: float = DEFAULT_BUDGET_CEILING_USD,
    ) -> None:
        if ceiling_usd <= 0:
            raise ValueError("ceiling_usd must be positive")
        self.path = Path(path)
        self.ceiling_usd = float(ceiling_usd)
        self.entries: list[dict[str, Any]] = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            saved_ceiling = float(data.get("ceiling_usd", self.ceiling_usd))
            if not math.isclose(saved_ceiling, self.ceiling_usd, abs_tol=1e-9):
                raise ValueError(
                    f"ledger ceiling is {saved_ceiling}, requested {self.ceiling_usd}; refusing an implicit change"
                )
            self.entries = list(data.get("entries", []))

    @property
    def total_cost_usd(self) -> float:
        return sum(float(entry.get("cost_usd", 0)) for entry in self.entries)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.ceiling_usd - self.total_cost_usd)

    def assert_can_spend(self, projected_cost_usd: float) -> None:
        if projected_cost_usd < 0:
            raise ValueError("projected cost must be nonnegative")
        projected_total = self.total_cost_usd + float(projected_cost_usd)
        if projected_total > self.ceiling_usd + 1e-9:
            raise BudgetExceeded(
                f"projected spend ${projected_total:.4f} exceeds the guarded "
                f"${self.ceiling_usd:.2f} ceiling"
            )

    def record(
        self,
        *,
        stage: str,
        run_id: str,
        cost_usd: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if cost_usd < 0:
            raise ValueError("cost_usd must be nonnegative")
        with _LOCK:
            # Reload under the lock so two worker threads cannot overwrite entries.
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.entries = list(data.get("entries", []))
            if any(entry.get("run_id") == run_id for entry in self.entries):
                raise ValueError(f"run_id {run_id!r} already exists in the budget ledger")
            self.entries.append(
                {
                    "stage": stage,
                    "run_id": run_id,
                    "cost_usd": round(float(cost_usd), 10),
                    "metadata": redact(metadata or {}),
                }
            )
            atomic_write_json(
                self.path,
                {
                    "schema_version": 1,
                    "ceiling_usd": self.ceiling_usd,
                    "total_cost_usd": self.total_cost_usd,
                    "remaining_usd": self.remaining_usd,
                    "entries": self.entries,
                },
            )


def _response_cache_hit(response: Any) -> bool:
    if isinstance(response, dict):
        return bool(response.get("cache_hit"))
    return bool(getattr(response, "cache_hit", False))


def _number(value: Any) -> float:
    return float(value or 0)


def summarize_history(history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate DSPy/LiteLLM history while retaining a per-model breakdown."""

    models: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "request_count": 0,
            "cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
    )
    for entry in history:
        model = str(entry.get("model") or entry.get("response_model") or "unknown")
        usage = entry.get("usage") or {}
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        prompt_tokens = _number(usage.get("prompt_tokens", usage.get("input_tokens")))
        completion_tokens = _number(usage.get("completion_tokens", usage.get("output_tokens")))
        details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
        if hasattr(details, "model_dump"):
            details = details.model_dump()
        reasoning_tokens = _number(details.get("reasoning_tokens"))
        total_tokens = _number(usage.get("total_tokens")) or prompt_tokens + completion_tokens

        bucket = models[model]
        bucket["request_count"] += 1
        bucket["cache_hits"] += int(_response_cache_hit(entry.get("response")))
        bucket["prompt_tokens"] += int(prompt_tokens)
        bucket["completion_tokens"] += int(completion_tokens)
        bucket["reasoning_tokens"] += int(reasoning_tokens)
        bucket["total_tokens"] += int(total_tokens)
        bucket["cost_usd"] += _number(entry.get("cost"))

    for bucket in models.values():
        bucket["cost_usd"] = round(bucket["cost_usd"], 10)
    return {
        "request_count": sum(bucket["request_count"] for bucket in models.values()),
        "cache_hits": sum(bucket["cache_hits"] for bucket in models.values()),
        "prompt_tokens": sum(bucket["prompt_tokens"] for bucket in models.values()),
        "completion_tokens": sum(bucket["completion_tokens"] for bucket in models.values()),
        "reasoning_tokens": sum(bucket["reasoning_tokens"] for bucket in models.values()),
        "total_tokens": sum(bucket["total_tokens"] for bucket in models.values()),
        "cost_usd": round(sum(bucket["cost_usd"] for bucket in models.values()), 10),
        "models": dict(models),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_latencies(latencies: Sequence[float]) -> dict[str, Any]:
    if not latencies:
        return {"count": 0}
    if any(value < 0 for value in latencies):
        raise ValueError("latencies must be nonnegative")
    values = [float(value) for value in latencies]
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "p50_seconds": _percentile(values, 0.50),
        "p95_seconds": _percentile(values, 0.95),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def classify_api_error(error: BaseException | str) -> str:
    if isinstance(error, BudgetExceeded):
        return "budget_stopped"
    text = f"{type(error).__name__ if isinstance(error, BaseException) else ''}: {error}".lower()
    if any(
        marker in text
        for marker in (
            "insufficient_quota",
            "credit balance",
            "billing hard limit",
            "billing_hard_limit",
            "run out of credit",
            "exceeded your current quota",
        )
    ):
        return "credit_exhausted"
    if any(marker in text for marker in ("ratelimit", "rate limit", "rate_limit", "429")):
        return "rate_limited"
    return "failed"


def is_global_stop_status(status: str) -> bool:
    return status in {"rate_limited", "credit_exhausted", "budget_stopped"}


def redact(value: Any) -> Any:
    """Recursively remove credential-shaped keys and values from persisted data."""

    if not isinstance(value, type) and callable(getattr(value, "model_dump", None)):
        value = value.model_dump()
    elif not isinstance(value, type) and callable(getattr(value, "dict", None)):
        value = value.dict()
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if _SECRET_KEY_PATTERN.search(str(key)):
                cleaned[str(key)] = "[REDACTED]"
            else:
                cleaned[str(key)] = redact(item)
        return cleaned
    if isinstance(value, (list, tuple, set)):
        return [redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        cleaned_text = value
        for pattern in _SECRET_VALUE_PATTERNS:
            cleaned_text = pattern.sub("[REDACTED]", cleaned_text)
        return cleaned_text
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(redact(record), ensure_ascii=False, default=str) + "\n")
