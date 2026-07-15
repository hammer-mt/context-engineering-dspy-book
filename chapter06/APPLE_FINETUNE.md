# DSPy weight optimization on Apple Silicon

## What was actually blocked

DSPy 3.2.1's `BootstrapFinetune` does not contain a CUDA check. It bootstraps
accepted chat traces and calls the student's `LM.finetune()` method. `LM.finetune()`
then delegates to a provider. The stock DSPy provider base class is not finetunable,
and DSPy 3.2.1 does not ship a maintained local Transformers provider; that missing
provider boundary was incorrectly reported as a hardware limitation.

The implementation evidence is in DSPy's official source for
[`BootstrapFinetune`](https://github.com/stanfordnlp/dspy/blob/3.2.1/dspy/teleprompt/bootstrap_finetune.py),
[`LM.finetune()`](https://github.com/stanfordnlp/dspy/blob/3.2.1/dspy/clients/lm.py),
and the base
[`Provider`](https://github.com/stanfordnlp/dspy/blob/3.2.1/dspy/clients/provider.py).
DSPy's optimizer guide describes BootstrapFinetune as weight distillation and
BetterTogether as a composition of prompt and weight optimizers:
[`optimizers.md`](https://github.com/stanfordnlp/dspy/blob/3.2.1/docs/docs/learn/optimization/optimizers.md).

## Faithful local route

`chapter06.apple_finetune` supplies the provider DSPy expects and nothing above
that boundary:

1. DSPy's stock BootstrapFinetune creates and filters chat traces.
2. DSPy's stock `LM.finetune()` invokes `AppleSiliconProvider.finetune()`.
3. TRL `SFTTrainer` trains a PEFT LoRA adapter for
   `Qwen/Qwen2.5-0.5B-Instruct` on MPS or CPU.
4. The returned model identifier points to the base model plus adapter, and the
   DSPy LM uses that identifier for local inference and JSON serialization.
5. BetterTogether remains DSPy's stock optimizer and runs the explicit `p -> w`
   strategy with the same weight optimizer.

This follows the official Transformers Apple Silicon path: Trainer detects and
uses PyTorch's `mps` device when available. See Hugging Face's
[Apple Silicon training guide](https://huggingface.co/docs/transformers/en/perf_train_special),
[TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer), and
[PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora).
The chosen [Qwen2.5-0.5B-Instruct model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
is a small Apache-2.0 instruct model suitable for a bounded local demonstration.

## Reproduction controls

From the repository root:

```bash
python -m chapter06.optimizer_experiment bootstrap-finetune --profile smoke
python -m chapter06.optimizer_experiment better-together --profile smoke
python -m chapter06.optimizer_experiment bootstrap-finetune --profile full
python -m chapter06.optimizer_experiment better-together --profile full
```

The default device is `auto`, which chooses MPS when PyTorch reports it available
and otherwise chooses CPU. Set `CHAPTER06_FINETUNE_DEVICE=mps` to require MPS or
`CHAPTER06_FINETUNE_DEVICE=cpu` to force the CPU route. Local self-teaching is the
default and incurs no API cost; `CHAPTER06_FINETUNE_TEACHER=remote` is an explicit
opt-in to the configured remote task model.

The smoke profile uses 6 training, 4 validation, and 4 test examples with one
optimizer step. The full profile uses the frozen 36/18/20 splits and caps training
at 18 optimizer steps. Both profiles use batch size 1, LoRA rank 8, and fixed seed
42.

## Concrete Mac evidence

The completed artifacts were produced on an Apple M3 Max Mac with 64 GB unified
memory. PyTorch 2.13.0 reported MPS built and available, and TRL reported its
trainer device as `mps`.

| Run | Result | Training evidence | Reload evidence |
|---|---|---|---|
| [BootstrapFinetune smoke](results/runs/smoke/bootstrap-finetune/20260715T011714.494591Z/) | completed, 50% → 50%, 14.18s | 3 accepted traces, 1 MPS step | prompt/model/prediction parity |
| [BetterTogether smoke](results/runs/smoke/better-together/20260715T011812.511232Z/) | completed, 50% → 50%, 52.27s | `p -> w`, 3 accepted traces, 1 MPS step | prompt/model/prediction parity |
| [BootstrapFinetune full](results/runs/full/bootstrap-finetune/20260715T034040.794581Z/) | completed, 50% → 50%, 67.53s | 17 accepted traces, 18 MPS steps, train loss 1.6397 | prompt/model parity; predictions 3/3 |
| [BetterTogether full](results/runs/full/better-together/20260715T034246.168669Z/) | completed, 50% → 50%, 278.30s | original, `p`, and `p -> w`; 17 accepted traces; 18 MPS steps | prompt/model parity; predictions 3/3 |

BetterTogether's three validation strategies tied at 50%, so DSPy selected the
original program. The `p -> w` candidate nevertheless contains the trained adapter
and is preserved alongside its exact prompt. No result is relabeled as a win.

## Limitations

- The local weight-optimizer rows use Qwen, while the prompt-optimizer rows use
  GPT-5.6 Luna. The frozen split is identical, but absolute accuracy is not a
  controlled cross-model comparison.
- MPS training is single-device and depends on the model fitting in unified memory.
  CPU is supported but slower. Larger models may still require remote or CUDA
  hardware; that does not invalidate this bounded local route.
- The serialized JSON contains a repository-relative adapter reference. Reload the
  program with `rehydrate_local_lms()` after DSPy's `Module.load()` so the generic
  DSPy LM is restored to the local Transformers implementation.
