from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chapter06.experiment_runtime import (
    BudgetExceeded,
    BudgetLedger,
    classify_api_error,
    redact,
    summarize_history,
    summarize_latencies,
)


class BudgetLedgerTest(unittest.TestCase):
    def test_records_stages_and_refuses_projected_ceiling_crossing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = BudgetLedger(path, ceiling_usd=95.0)
            ledger.record(stage="dataset", run_id="generation-1", cost_usd=4.25)

            self.assertAlmostEqual(ledger.total_cost_usd, 4.25)
            ledger.assert_can_spend(90.75)
            with self.assertRaises(BudgetExceeded):
                ledger.assert_can_spend(90.76)

            reloaded = BudgetLedger(path, ceiling_usd=95.0)
            self.assertAlmostEqual(reloaded.total_cost_usd, 4.25)
            self.assertEqual(reloaded.entries[0]["stage"], "dataset")

    def test_rejects_negative_costs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json")
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                ledger.record(stage="test", run_id="bad", cost_usd=-0.01)

    def test_records_actual_spend_even_when_it_crosses_the_guarded_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = BudgetLedger(path, ceiling_usd=1.0)

            ledger.record(stage="in-flight", run_id="actual-1", cost_usd=1.2)

            self.assertAlmostEqual(ledger.total_cost_usd, 1.2)
            self.assertEqual(ledger.remaining_usd, 0.0)
            with self.assertRaises(BudgetExceeded):
                ledger.assert_can_spend(0.01)


class HistorySummaryTest(unittest.TestCase):
    def test_splits_tokens_cost_requests_and_cache_hits_by_model(self) -> None:
        history = [
            {
                "model": "openai/task",
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                "cost": 0.01,
                "response": {"cache_hit": False},
            },
            {
                "model": "openai/task",
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
                "cost": None,
                "response": {"cache_hit": True},
            },
            {
                "model": "openai/reflection",
                "usage": {"input_tokens": 200, "output_tokens": 30, "total_tokens": 230},
                "cost": 0.04,
            },
        ]

        summary = summarize_history(history)

        self.assertEqual(summary["request_count"], 3)
        self.assertAlmostEqual(summary["cost_usd"], 0.05)
        self.assertEqual(summary["models"]["openai/task"]["prompt_tokens"], 150)
        self.assertEqual(summary["models"]["openai/task"]["completion_tokens"], 30)
        self.assertEqual(summary["models"]["openai/task"]["cache_hits"], 1)
        self.assertEqual(summary["models"]["openai/reflection"]["prompt_tokens"], 200)


class FailureAndPrivacyTest(unittest.TestCase):
    def test_classifies_terminal_rate_limit_and_credit_exhaustion(self) -> None:
        self.assertEqual(classify_api_error(RuntimeError("429 rate limit after retries")), "rate_limited")
        self.assertEqual(classify_api_error(RuntimeError("insufficient_quota: billing hard limit")), "credit_exhausted")
        self.assertEqual(classify_api_error(ValueError("bad optimizer argument")), "failed")
        self.assertEqual(classify_api_error(BudgetExceeded("guarded ceiling")), "budget_stopped")

    def test_redacts_keys_and_authorization_headers_recursively(self) -> None:
        value = {
            "api_key": "sk-secret-value",
            "headers": {"Authorization": "Bearer secret-token"},
            "message": "request used sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        }

        cleaned = redact(value)
        serialized = json.dumps(cleaned)

        self.assertNotIn("secret-value", serialized)
        self.assertNotIn("secret-token", serialized)
        self.assertNotIn("sk-proj-", serialized)
        self.assertIn("[REDACTED]", serialized)

    def test_redact_handles_class_objects_with_instance_model_dump_methods(self) -> None:
        class PydanticLike:
            def model_dump(self):
                return {"value": 1}

        cleaned = redact({"metadata_class": PydanticLike, "instance": PydanticLike()})

        self.assertIn("PydanticLike", cleaned["metadata_class"])
        self.assertEqual(cleaned["instance"], {"value": 1})


class LatencySummaryTest(unittest.TestCase):
    def test_reports_mean_median_p50_and_p95(self) -> None:
        summary = summarize_latencies([0.1, 0.2, 0.3, 0.4, 1.0])

        self.assertEqual(summary["count"], 5)
        self.assertAlmostEqual(summary["mean_seconds"], 0.4)
        self.assertAlmostEqual(summary["median_seconds"], 0.3)
        self.assertAlmostEqual(summary["p50_seconds"], 0.3)
        self.assertAlmostEqual(summary["p95_seconds"], 0.88)

    def test_empty_latency_sample_is_explicit(self) -> None:
        self.assertEqual(summarize_latencies([]), {"count": 0})


if __name__ == "__main__":
    unittest.main()
