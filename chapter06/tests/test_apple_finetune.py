from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dspy

from chapter06.apple_finetune import (
    AppleSiliconProvider,
    TransformersLocalLM,
    make_model_spec,
    parse_model_spec,
    rehydrate_local_lms,
    resolve_device,
)
from chapter06.optimizer_experiment import AIDetector


class AppleFinetuneTest(unittest.TestCase):
    def test_model_spec_round_trips_a_repository_relative_adapter(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[2]
        ) as directory:
            spec = make_model_spec("org/model", Path(directory) / "adapter")
            base_model, adapter_path = parse_model_spec(spec)

        self.assertEqual(base_model, "org/model")
        self.assertEqual(adapter_path, (Path(directory) / "adapter").resolve())
        self.assertNotIn(str(Path(__file__).resolve().parents[3]), spec)

    def test_auto_device_prefers_mps_and_can_fall_back_to_cpu(self) -> None:
        with (
            patch("torch.backends.mps.is_built", return_value=True),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(resolve_device("auto"), "mps")

        with (
            patch("torch.backends.mps.is_built", return_value=True),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            self.assertEqual(resolve_device("auto"), "cpu")
            with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
                resolve_device("mps")

    def test_provider_is_a_real_dspy_finetuning_provider(self) -> None:
        lm = TransformersLocalLM(base_model="org/model")

        self.assertIsInstance(lm, dspy.LM)
        self.assertIsInstance(lm.provider, AppleSiliconProvider)
        self.assertTrue(lm.provider.finetunable)

    def test_program_reload_can_rehydrate_the_local_lm_type(self) -> None:
        program = AIDetector()
        program.set_lm(dspy.LM(make_model_spec("org/model"), cache=False))

        rehydrate_local_lms(program)

        self.assertIsInstance(program.get_lm(), TransformersLocalLM)


if __name__ == "__main__":
    unittest.main()
