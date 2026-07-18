# Chapter 6 benchmark results

## Experimental frame

Every optimizer uses the checked-in `data/ai_vs_human_chapter06.csv` dataset and the same frozen, pair-grouped split: 36 training examples, 18 validation examples, and 20 test examples. A human passage and its AI rewrite always stay in the same split.

The test partition is intentionally adversarial and small. One changed prediction moves accuracy by five percentage points, so these results are best used to compare optimizer behavior on this teaching workload—not as a general AI-detector leaderboard.

## Final accuracy

This is the student-facing comparison: the accuracy of the final program returned by each optimizer on the same 20 test examples. The unoptimized baseline has its own notebook.

| Optimizer | Task model | Final test accuracy | Correct |
|---|---|---:|---:|
| Unoptimized baseline | GPT-5.6 Luna | 50% | 10/20 |
| LabeledFewShot | GPT-5.6 Luna | 75% | 15/20 |
| BootstrapFewShot | GPT-5.6 Luna | 75% | 15/20 |
| BootstrapRS | GPT-5.6 Luna | 70% | 14/20 |
| KNNFewShot | GPT-5.6 Luna | 70% | 14/20 |
| COPRO | GPT-5.6 Luna | 50% | 10/20 |
| MIPROv2 | GPT-5.6 Luna | 95% | 19/20 |
| GEPA | GPT-5.6 Luna | 90% | 18/20 |
| SIMBA | GPT-5.6 Luna | 70% | 14/20 |
| Ensemble | GPT-5.6 Luna | 80% | 16/20 |
| BootstrapFinetune | Qwen2.5-0.5B | 55% | 11/20 |
| BetterTogether | Qwen2.5-0.5B | 50% | 10/20 |

The prompt optimizers use Luna. The weight optimizers use Qwen because their purpose is to train a local model through DSPy on Apple Silicon. Their final accuracies use the same test examples, but their model capacity and baseline differ.

## Corrected BootstrapFinetune run

The earlier BootstrapFinetune experiment accepted only 17 human traces and no AI traces because the small Qwen student was also used as its own teacher. That run was not a valid demonstration of balanced classification fine-tuning.

The corrected path uses DSPy's own `BootstrapFinetune` and `LocalProvider`. The Mac-specific subclass only replaces LocalProvider's SGLang serving hooks with Transformers inference on MPS and translates DSPy 3.2.1's `max_seq_length` argument to the `max_length` name expected by the pinned TRL version. DSPy still prepares the assistant-only training examples, launches TRL/PEFT, and returns the fine-tuned program.

GPT-5.6 Sol scored 83.3% on the 36-example training split, and DSPy accepted 30 traces—16 human and 14 AI. The training configuration was selected with the training and validation partitions only. The clean end-to-end rerun then produced this frozen-test result; no settings were changed in response to it:

- Same-model Qwen baseline: 50% (10/20)
- Fine-tuned Qwen: 55% (11/20)
- Change: +5 percentage points

This replaces the earlier 35% result, which came from a custom training implementation that did not preserve DSPy's assistant-token masking and therefore was not a valid parity test. The native DSPy run trained for 10 epochs on Apple Silicon MPS. Full compact evidence is in `results/bootstrap_finetune_native_rerun.json`.

The checked-in BetterTogether number comes from the earlier run and has not been rerun with the new balanced Sol-teacher path. Its notebook now uses the corrected executable path, but the published 50% should be treated as legacy evidence until that longer experiment is rerun.

## Reproducibility

Each optimizer notebook loads and validates the same dataset hash and split before doing any work. By default it displays the checked-in result; set `CHAPTER06_RUN_LIVE=1` before starting Jupyter to compile and evaluate that optimizer again. The baseline is intentionally separate, so optimizer notebooks report final accuracy rather than mixing reference uplift and within-run uplift.

The detailed historical measurements—optimization time, cost, latency, and reload checks—remain in `results/benchmark_summary.json`. Canonical program and prompt snapshots remain in `optimized_programs/final/` and `results/final_prompts/`. Large adapters, provider transcripts, caches, and temporary training files are excluded from the student download.
