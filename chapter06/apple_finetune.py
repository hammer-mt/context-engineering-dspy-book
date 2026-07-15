"""Apple Silicon training and inference backend for DSPy weight optimizers.

DSPy 3.2 delegates fine-tuning to ``LM.finetune()`` and its provider API, but
does not ship a local provider.  This module supplies that missing provider for
Transformers/TRL.  BootstrapFinetune and BetterTogether themselves remain the
stock DSPy optimizers.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import dspy

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat, validate_data_format
from litellm import ModelResponse


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PREFIX = "hf-mps:"
ADAPTER_SEPARATOR = "::adapter="
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def resolve_device(preference: str = "auto") -> str:
    """Resolve an explicit Transformers device without pretending MPS exists."""

    if preference not in {"auto", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, mps, cpu")
    mps_available = bool(
        torch.backends.mps.is_built() and torch.backends.mps.is_available()
    )
    if preference == "mps" and not mps_available:
        raise RuntimeError(
            "MPS was requested but torch.backends.mps.is_available() is false"
        )
    if preference == "auto":
        return "mps" if mps_available else "cpu"
    return preference


def local_capabilities() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "selected_device": resolve_device(
            os.getenv("CHAPTER06_FINETUNE_DEVICE", "auto")
        ),
    }


def make_model_spec(base_model: str, adapter_path: str | Path | None = None) -> str:
    spec = f"{MODEL_PREFIX}{base_model}"
    if adapter_path is None:
        return spec
    path = Path(adapter_path).resolve()
    try:
        stored_path = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        stored_path = str(path)
    return f"{spec}{ADAPTER_SEPARATOR}{stored_path}"


def parse_model_spec(model: str) -> tuple[str, Path | None]:
    if not model.startswith(MODEL_PREFIX):
        raise ValueError(f"not a Chapter 6 local model spec: {model!r}")
    payload = model[len(MODEL_PREFIX) :]
    if ADAPTER_SEPARATOR not in payload:
        return payload, None
    base_model, adapter_text = payload.split(ADAPTER_SEPARATOR, 1)
    adapter_path = Path(adapter_text)
    if not adapter_path.is_absolute():
        adapter_path = REPO_ROOT / adapter_path
    return base_model, adapter_path.resolve()


def is_local_model(model: str) -> bool:
    return model.startswith(MODEL_PREFIX)


class AppleTrainingJob(TrainingJob):
    def status(self) -> str:
        if self.cancelled():
            return "cancelled"
        if self.done():
            result = self.result()
            return "failed" if isinstance(result, Exception) else "succeeded"
        return "running" if self.thread and self.thread.is_alive() else "pending"


class AppleSiliconProvider(Provider):
    """DSPy fine-tuning provider using TRL SFT and PEFT LoRA on MPS or CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.finetunable = True
        self.TrainingJob = AppleTrainingJob

    @staticmethod
    def is_provider_model(model: str) -> bool:
        return is_local_model(model)

    @staticmethod
    def launch(
        lm: "TransformersLocalLM", launch_kwargs: dict[str, Any] | None = None
    ) -> None:
        del launch_kwargs
        lm.ensure_loaded()

    @staticmethod
    def kill(
        lm: "TransformersLocalLM", launch_kwargs: dict[str, Any] | None = None
    ) -> None:
        del launch_kwargs
        lm.unload()

    @staticmethod
    def finetune(
        job: TrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Run the training job and return a DSPy-loadable local model spec."""

        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        if train_data_format not in {
            TrainDataFormat.CHAT,
            TrainDataFormat.CHAT.value,
            "chat",
        }:
            raise ValueError(
                f"AppleSiliconProvider only supports DSPy chat training data, got {train_data_format}"
            )
        validate_data_format(train_data, TrainDataFormat.CHAT)
        if not train_data:
            raise ValueError("BootstrapFinetune produced no accepted training traces")

        config = dict(train_kwargs or {})
        if "output_dir" not in config:
            raise ValueError("AppleSiliconProvider requires train_kwargs['output_dir']")
        root_output_dir = Path(config.pop("output_dir")).resolve()
        root_output_dir.mkdir(parents=True, exist_ok=True)
        job_output_dir = root_output_dir / f"job-{time.time_ns()}"
        adapter_dir = job_output_dir / "adapter"
        job_output_dir.mkdir(parents=True, exist_ok=False)

        device = resolve_device(str(config.pop("device", "auto")))
        max_steps = int(config.pop("max_steps", -1))
        epochs = float(config.pop("num_train_epochs", 1.0))
        batch_size = int(config.pop("per_device_train_batch_size", 1))
        accum_steps = int(config.pop("gradient_accumulation_steps", 1))
        learning_rate = float(config.pop("learning_rate", 2e-4))
        max_length = int(config.pop("max_length", 768))
        lora_rank = int(config.pop("lora_rank", 8))
        lora_alpha = int(config.pop("lora_alpha", 16))
        seed = int(config.pop("seed", 42))
        if config:
            raise ValueError(f"unsupported Apple fine-tune arguments: {sorted(config)}")

        base_model, prior_adapter = parse_model_spec(model)
        if prior_adapter is not None:
            raise ValueError(
                "repeated fine-tuning of an existing adapter is not enabled in this bounded experiment"
            )

        (job_output_dir / "train_data.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in train_data),
            encoding="utf-8",
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
        base.config.use_cache = False

        sft_config = SFTConfig(
            output_dir=str(job_output_dir / "trainer"),
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accum_steps,
            learning_rate=learning_rate,
            max_length=max_length,
            use_cpu=device == "cpu",
            bf16=False,
            fp16=False,
            gradient_checkpointing=False,
            optim="adamw_torch",
            dataloader_pin_memory=False,
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=1,
            report_to="none",
            seed=seed,
            data_seed=seed,
            packing=False,
        )
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        trainer = SFTTrainer(
            model=base,
            args=sft_config,
            train_dataset=Dataset.from_list(train_data),
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        selected_device = str(trainer.accelerator.device)
        if device == "mps" and selected_device != "mps":
            raise RuntimeError(
                f"TRL selected {selected_device!r} after MPS was requested"
            )

        result = trainer.train()
        trainer.save_model(str(adapter_dir))
        summary = {
            "backend": "transformers-trl-peft",
            "requested_device": device,
            "trainer_device": selected_device,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "base_model": base_model,
            "train_examples": len(train_data),
            "train_data_format": str(train_data_format),
            "training": {
                "max_steps": max_steps,
                "num_train_epochs": epochs,
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": accum_steps,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "seed": seed,
            },
            "metrics": result.metrics,
            "log_history": trainer.state.log_history,
            "adapter_path": str(adapter_dir.relative_to(REPO_ROOT)),
        }
        (job_output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
        )

        del trainer, base
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return make_model_spec(base_model, adapter_dir)


class TransformersLocalLM(dspy.LM):
    """A serializable DSPy LM that performs local Transformers inference."""

    def __init__(
        self,
        model: str | None = None,
        *,
        base_model: str = DEFAULT_LOCAL_MODEL,
        device: str = "auto",
        max_context_tokens: int = 1024,
        max_tokens: int = 96,
        cache: bool = False,
        callbacks: Sequence[Any] | None = None,
        num_retries: int = 0,
        **kwargs: Any,
    ) -> None:
        model = model or make_model_spec(base_model)
        if not is_local_model(model):
            model = make_model_spec(model)
        self.device_preference = device
        self.max_context_tokens = max_context_tokens
        self._runtime_model = None
        self._runtime_tokenizer = None
        self._runtime_lock = threading.RLock()
        super().__init__(
            model,
            cache=cache,
            max_tokens=max_tokens,
            callbacks=list(callbacks or []),
            num_retries=num_retries,
            provider=AppleSiliconProvider(),
            **kwargs,
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "TransformersLocalLM":
        copied = self.copy()
        memo[id(self)] = copied
        return copied

    def copy(self, **kwargs: Any) -> "TransformersLocalLM":
        model = kwargs.pop("model", self.model)
        lm_kwargs = copy.deepcopy(self.kwargs)
        lm_kwargs.update(kwargs)
        max_tokens = int(lm_kwargs.pop("max_tokens", 96))
        return TransformersLocalLM(
            model=model,
            device=self.device_preference,
            max_context_tokens=self.max_context_tokens,
            max_tokens=max_tokens,
            cache=self.cache,
            callbacks=self.callbacks,
            num_retries=self.num_retries,
            **lm_kwargs,
        )

    def ensure_loaded(self) -> None:
        with self._runtime_lock:
            if self._runtime_model is not None:
                return
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            base_model, adapter_path = parse_model_spec(self.model)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                base_model, dtype=torch.float32
            )
            if adapter_path is not None:
                model = PeftModel.from_pretrained(model, str(adapter_path))
            device = resolve_device(self.device_preference)
            model.to(device)
            model.eval()
            self._runtime_tokenizer = tokenizer
            self._runtime_model = model

    def unload(self) -> None:
        with self._runtime_lock:
            self._runtime_model = None
            self._runtime_tokenizer = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        with self._runtime_lock:
            self.ensure_loaded()
            tokenizer = self._runtime_tokenizer
            model = self._runtime_model
            request_messages = messages or [{"role": "user", "content": prompt or ""}]
            rendered = tokenizer.apply_chat_template(
                request_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_tokens,
            ).to(model.device)
            max_new_tokens = int(
                kwargs.get("max_tokens") or self.kwargs.get("max_tokens") or 96
            )
            temperature = kwargs.get("temperature", self.kwargs.get("temperature"))
            do_sample = temperature is not None and float(temperature) > 0
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = float(temperature)
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            output_ids = generated[0, prompt_tokens:]
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
            completion_tokens = int(output_ids.shape[-1])
            return ModelResponse(
                model=self.model,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )

    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        state["model"] = self.model
        state["cache"] = False
        state["device"] = self.device_preference
        state["max_context_tokens"] = self.max_context_tokens
        return state


def rehydrate_local_lms(program: dspy.Module) -> dspy.Module:
    """Replace generic LMs created by ``Module.load`` with the local backend."""

    for predictor in program.predictors():
        lm = predictor.lm
        if (
            lm is not None
            and is_local_model(lm.model)
            and not isinstance(lm, TransformersLocalLM)
        ):
            state = lm.dump_state()
            predictor.lm = TransformersLocalLM(
                model=state["model"],
                device=str(
                    state.get("device", os.getenv("CHAPTER06_FINETUNE_DEVICE", "auto"))
                ),
                max_context_tokens=int(state.get("max_context_tokens", 1024)),
                max_tokens=int(state.get("max_tokens") or 96),
                cache=False,
            )
    return program
