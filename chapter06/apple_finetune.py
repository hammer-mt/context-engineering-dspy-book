"""Minimal Apple Silicon compatibility for DSPy's native local provider.

DSPy owns trace formatting, assistant-token masking, PEFT configuration, and the
Transformers/TRL training loop. This module only supplies Mac-friendly local
inference and maps one renamed TRL configuration argument for DSPy 3.2.1.
"""

from __future__ import annotations

import copy
import gc
import inspect
import os
import threading
from pathlib import Path
from typing import Any, Sequence

import dspy
from dspy.clients.lm_local import LocalProvider
from litellm import ModelResponse


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PREFIX = "openai/local:"
LEGACY_MODEL_PREFIX = "hf-mps:"
ADAPTER_SEPARATOR = "::adapter="
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_TRL_COMPATIBILITY_LOCK = threading.RLock()


def resolve_device(preference: str = "auto") -> str:
    """Resolve an explicit Transformers device without pretending MPS exists."""

    import torch

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
    import torch

    return {
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "selected_device": resolve_device(
            os.getenv("CHAPTER06_FINETUNE_DEVICE", "auto")
        ),
    }


def make_model_spec(base_model: str, adapter_path: str | Path | None = None) -> str:
    """Return DSPy's native local model identifier.

    adapter_path remains supported only for reloading the previous publication
    artifact; new native LocalProvider runs save a merged model directory.
    """

    if adapter_path is None:
        return f"{LOCAL_MODEL_PREFIX}{base_model}"
    return (
        f"{LEGACY_MODEL_PREFIX}{base_model}{ADAPTER_SEPARATOR}"
        f"{Path(adapter_path).resolve()}"
    )


def parse_model_spec(model: str) -> tuple[str, Path | None]:
    """Resolve a native merged model or the prior adapter-based model format."""

    if model.startswith(LOCAL_MODEL_PREFIX):
        return model[len(LOCAL_MODEL_PREFIX) :], None
    if model.startswith(LEGACY_MODEL_PREFIX):
        payload = model[len(LEGACY_MODEL_PREFIX) :]
        if ADAPTER_SEPARATOR not in payload:
            return payload, None
        base_model, adapter_text = payload.split(ADAPTER_SEPARATOR, 1)
        adapter_path = Path(adapter_text)
        if not adapter_path.is_absolute():
            adapter_path = REPO_ROOT / adapter_path
        return base_model, adapter_path.resolve()
    raise ValueError(f"not a Chapter 6 local model spec: {model!r}")


def is_local_model(model: str) -> bool:
    return model.startswith((LOCAL_MODEL_PREFIX, LEGACY_MODEL_PREFIX))


class MacLocalProvider(LocalProvider):
    """DSPy's LocalProvider with only Mac inference and TRL-name compatibility."""

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
    def finetune(job, model, train_data, train_data_format, train_kwargs=None):
        """Delegate training to DSPy, aliasing TRL's renamed length argument."""

        import trl

        native_sft_config = trl.SFTConfig
        if "max_seq_length" in inspect.signature(native_sft_config).parameters:
            return LocalProvider.finetune(
                job, model, train_data, train_data_format, train_kwargs
            )

        def compatible_sft_config(*args: Any, **kwargs: Any):
            if "max_seq_length" in kwargs and "max_length" not in kwargs:
                kwargs["max_length"] = kwargs.pop("max_seq_length")
            return native_sft_config(*args, **kwargs)

        # DSPy's LocalProvider imports SFTConfig inside its training function.
        # Keep the temporary alias tightly scoped because trl is a module
        # singleton and BootstrapFinetune runs provider work in a thread.
        with _TRL_COMPATIBILITY_LOCK:
            trl.SFTConfig = compatible_sft_config
            try:
                return LocalProvider.finetune(
                    job, model, train_data, train_data_format, train_kwargs
                )
            finally:
                trl.SFTConfig = native_sft_config


class TransformersLocalLM(dspy.LM):
    """DSPy LM with native LocalProvider training and Transformers inference."""

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
            provider=MacLocalProvider(),
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
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path, adapter_path = parse_model_spec(self.model)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, fix_mistral_regex=True
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.float32
            )
            if adapter_path is not None:
                model = PeftModel.from_pretrained(model, str(adapter_path))
            model.to(resolve_device(self.device_preference))
            model.eval()
            self._runtime_tokenizer = tokenizer
            self._runtime_model = model

    def unload(self) -> None:
        with self._runtime_lock:
            self._runtime_model = None
            self._runtime_tokenizer = None
            gc.collect()
            try:
                import torch
            except ImportError:
                return
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        import torch

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
    """Replace generic LMs created by Module.load with the local backend."""

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
