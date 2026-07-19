"""Paired, deterministic analysis for repeated Chapter 6 evaluations."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Sequence


def majority_vote(
    prediction_runs: Sequence[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Collapse odd-numbered fresh runs to one prediction per example."""

    if len(prediction_runs) < 3 or len(prediction_runs) % 2 == 0:
        raise ValueError("majority voting requires an odd number of at least three runs")
    indexed = [
        {str(record["example_id"]): record for record in run}
        for run in prediction_runs
    ]
    example_ids = set(indexed[0])
    if not example_ids or any(set(run) != example_ids for run in indexed[1:]):
        raise ValueError("prediction runs must cover the same non-empty example set")

    output: list[dict[str, Any]] = []
    for example_id in sorted(example_ids):
        records = [run[example_id] for run in indexed]
        expected_values = {bool(record["expected_is_ai"]) for record in records}
        pair_ids = {str(record["pair_id"]) for record in records}
        if len(expected_values) != 1 or len(pair_ids) != 1:
            raise ValueError(f"inconsistent repeated records for {example_id}")
        positive_votes = sum(bool(record["predicted_is_ai"]) for record in records)
        predicted = positive_votes > len(records) / 2
        expected = expected_values.pop()
        output.append(
            {
                "pair_id": pair_ids.pop(),
                "example_id": example_id,
                "expected_is_ai": expected,
                "predicted_is_ai": predicted,
                "positive_votes": positive_votes,
                "run_count": len(records),
                "correct": int(predicted == expected),
            }
        )
    return output


def mcnemar_exact(baseline: Sequence[dict[str, Any]], optimized: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the exact two-sided McNemar test on paired correctness."""

    baseline_by_id = {str(row["example_id"]): row for row in baseline}
    optimized_by_id = {str(row["example_id"]): row for row in optimized}
    if set(baseline_by_id) != set(optimized_by_id):
        raise ValueError("baseline and optimized rows do not cover the same examples")
    baseline_only = 0
    optimized_only = 0
    for example_id in baseline_by_id:
        base_correct = bool(baseline_by_id[example_id]["correct"])
        opt_correct = bool(optimized_by_id[example_id]["correct"])
        baseline_only += int(base_correct and not opt_correct)
        optimized_only += int(opt_correct and not base_correct)
    disagreements = baseline_only + optimized_only
    if disagreements == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(disagreements, k) for k in range(min(baseline_only, optimized_only) + 1)
        ) / (2**disagreements)
        p_value = min(1.0, 2 * tail)
    return {
        "baseline_only_correct": baseline_only,
        "optimized_only_correct": optimized_only,
        "discordant_count": disagreements,
        "p_value": p_value,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_pair_bootstrap(
    baseline: Sequence[dict[str, Any]],
    optimized: Sequence[dict[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap the accuracy uplift while resampling semantic pairs."""

    if samples < 1:
        raise ValueError("samples must be positive")
    baseline_by_id = {str(row["example_id"]): row for row in baseline}
    optimized_by_id = {str(row["example_id"]): row for row in optimized}
    if set(baseline_by_id) != set(optimized_by_id):
        raise ValueError("baseline and optimized rows do not cover the same examples")

    deltas_by_pair: dict[str, list[float]] = defaultdict(list)
    for example_id, baseline_row in baseline_by_id.items():
        optimized_row = optimized_by_id[example_id]
        if str(baseline_row["pair_id"]) != str(optimized_row["pair_id"]):
            raise ValueError(f"pair mismatch for {example_id}")
        deltas_by_pair[str(baseline_row["pair_id"])].append(
            float(bool(optimized_row["correct"])) - float(bool(baseline_row["correct"]))
        )
    if any(len(deltas) != 2 for deltas in deltas_by_pair.values()):
        raise ValueError("each semantic pair must contain exactly two examples")

    pair_ids = sorted(deltas_by_pair)
    rng = random.Random(seed)
    uplifts: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(pair_ids) for _ in pair_ids]
        deltas = [delta for pair_id in sampled for delta in deltas_by_pair[pair_id]]
        uplifts.append(100 * sum(deltas) / len(deltas))
    observed = 100 * sum(
        delta for deltas in deltas_by_pair.values() for delta in deltas
    ) / (2 * len(pair_ids))
    return {
        "unit": "semantic_pair",
        "pair_count": len(pair_ids),
        "samples": samples,
        "seed": seed,
        "observed_uplift_pct_points": observed,
        "ci_low_pct_points": _quantile(uplifts, 0.025),
        "ci_high_pct_points": _quantile(uplifts, 0.975),
    }
