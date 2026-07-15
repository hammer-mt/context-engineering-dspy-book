# Chapter 6 benchmark: rerun-backed optimizer comparison

## Experimental frame

The frozen benchmark contains 74 passages in 37 human/AI semantic pairs. Pair IDs—not individual rows—are the split unit, so a source passage and its rewrite cannot leak across training, validation, and test. This rerun reused the checked-in split manifest and dataset hash; it did not regenerate or redesign the adversarial data.

The test partition is intentionally adversarial: baseline selection replicates scored 35%, 45%, and the frozen reference baseline for the optimizer comparison is 50%. Treat this as a stress test of optimization behavior, not as an unbiased estimate of real-world AI-detection accuracy.

## Results

| Optimizer | Task model | Accuracy | Uplift vs. 50.0% Luna reference | Run uplift | Optimize cost | Optimize time | Mean latency | P95 latency | Reload parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unoptimized baseline | openai/gpt-5.6-luna | 50.0% | +0.0 pts | +0.0 pts | $0.0000 | 0.0s | 2.00s | 3.50s | 2/3 |
| LabeledFewShot | openai/gpt-5.6-luna | 75.0% | +25.0 pts | +15.0 pts | $0.0000 | 0.0s | 2.29s | 3.50s | 3/3 |
| BootstrapFewShot | openai/gpt-5.6-luna | 75.0% | +25.0 pts | +25.0 pts | $0.0029 | 4.3s | 1.94s | 3.14s | 3/3 |
| BootstrapRS | openai/gpt-5.6-luna | 70.0% | +20.0 pts | +25.0 pts | $0.2860 | 6.6min | 1.79s | 2.66s | 3/3 |
| KNNFewShot | openai/gpt-5.6-luna | 70.0% | +20.0 pts | +20.0 pts | $0.0000 | 0.0s | 2.06s | 2.57s | 3/3 |
| COPRO | openai/gpt-5.6-luna | 50.0% | +0.0 pts | +5.0 pts | $0.5491 | 12.6min | 2.31s | 3.05s | 3/3 |
| MIPROv2 | openai/gpt-5.6-luna | 95.0% | +45.0 pts | +55.0 pts | $0.5072 | 7.6min | 1.67s | 2.87s | 3/3 |
| GEPA | openai/gpt-5.6-luna | 90.0% | +40.0 pts | +55.0 pts | $1.5658 | 15.0min | 1.78s | 2.32s | 3/3 |
| SIMBA | openai/gpt-5.6-luna | 70.0% | +20.0 pts | +30.0 pts | $0.9482 | 18.8min | 1.63s | 2.22s | 3/3 |
| Ensemble | openai/gpt-5.6-luna | 80.0% | +30.0 pts | +25.0 pts | $0.3028 | 6.7min | 5.10s | 6.35s | 3/3 |
| BootstrapFinetune | Qwen/Qwen2.5-0.5B-Instruct | 50.0% | — | +0.0 pts | $0.0000 | 1.1min | 1.16s | 1.88s | 3/3 |
| BetterTogether | Qwen/Qwen2.5-0.5B-Instruct | 50.0% | — | +0.0 pts | $0.0000 | 4.6min | 1.39s | 1.94s | 3/3 |

## Interpretation

MIPROv2 led the locked test set at 95%, a +45-point improvement over the frozen baseline. Among the leaders, MIPROv2 compiled fastest (7.6min) and used $0.5072 in measured optimization calls.

The strongest zero-paid-compile option was LabeledFewShot at 75%. That makes it a useful first move when iteration speed and cost matter more than extracting the final few points. Zero compile cost does not mean zero inference cost: the table separates optimization spend from evaluation latency for exactly that reason.

Inference tradeoffs are visible as well. Ensemble had the highest mean latency (5.10s), so its accuracy should be weighed against serving cost rather than read in isolation. The learned instructions and demonstrations are preserved beside each notebook, which makes qualitative inspection part of the comparison instead of treating accuracy as the only outcome.

## Reproducibility and limits

The completed rows used $4.81 in measured baseline, optimization, optimized-evaluation, and bounded reload-verification calls. Every selected run includes its manifest, console transcript, sanitized LM history, per-example predictions, metrics, serialized program, and extracted prompt under `chapter06/results/runs/full/`. Canonical programs and prompts are copied to `chapter06/optimized_programs/final/` and `chapter06/results/final_prompts/`.

Serialization was checked with prompt/demo state equality after reload and bounded prediction-label parity on frozen test examples; the local weight rows additionally verify the saved base-model and adapter references. Across the selected runs, 35/36 reloaded predictions matched their pre-serialization labels. The per-run counts are reported in the table because a mismatch from an uncached stochastic model is evidence to preserve, not a reason to retry until it disappears; this remains a bounded reproducibility check rather than a claim of global determinism.

The two weight optimizers are reported as a separate local-model experiment: BootstrapFinetune 50% → 50% in 1.1min; BetterTogether 50% → 50% in 4.6min. They use `Qwen/Qwen2.5-0.5B-Instruct` through DSPy's Transformers/TRL/PEFT provider boundary on MPS, while the prompt-optimizer rows use GPT-5.6 Luna. Their within-run uplift and training evidence are valid on the same frozen split, but their absolute accuracy is deliberately not compared with the Luna reference. BetterTogether preserves its trained `p -> w` candidate even when DSPy retains the original program after a validation tie.

Finally, the test set has only 20 passages, so one changed prediction moves accuracy by five points. The comparison is most useful for understanding optimizer mechanisms and tradeoffs under one frozen adversarial workload, not for declaring a universal ranking.
