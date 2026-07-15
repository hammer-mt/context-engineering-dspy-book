# Chapter 6 optimizer benchmark

Frozen reference baseline: **50.0%**. Adversarial selection replicates: 35.0%, 45.0%.

| Optimizer | Task model | Accuracy | Uplift vs. 50.0% Luna reference | Run uplift | Optimize cost | Optimize time | Mean latency | P95 latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Unoptimized baseline | openai/gpt-5.6-luna | 50.0% | +0.0 pts | +0.0 pts | $0.0000 | 0.0s | 2.00s | 3.50s |
| LabeledFewShot | openai/gpt-5.6-luna | 75.0% | +25.0 pts | +15.0 pts | $0.0000 | 0.0s | 2.29s | 3.50s |
| BootstrapFewShot | openai/gpt-5.6-luna | 75.0% | +25.0 pts | +25.0 pts | $0.0029 | 4.3s | 1.94s | 3.14s |
| BootstrapRS | openai/gpt-5.6-luna | 70.0% | +20.0 pts | +25.0 pts | $0.2860 | 6.6min | 1.79s | 2.66s |
| KNNFewShot | openai/gpt-5.6-luna | 70.0% | +20.0 pts | +20.0 pts | $0.0000 | 0.0s | 2.06s | 2.57s |
| COPRO | openai/gpt-5.6-luna | 50.0% | +0.0 pts | +5.0 pts | $0.5491 | 12.6min | 2.31s | 3.05s |
| MIPROv2 | openai/gpt-5.6-luna | 95.0% | +45.0 pts | +55.0 pts | $0.5072 | 7.6min | 1.67s | 2.87s |
| GEPA | openai/gpt-5.6-luna | 90.0% | +40.0 pts | +55.0 pts | $1.5658 | 15.0min | 1.78s | 2.32s |
| SIMBA | openai/gpt-5.6-luna | 70.0% | +20.0 pts | +30.0 pts | $0.9482 | 18.8min | 1.63s | 2.22s |
| Ensemble | openai/gpt-5.6-luna | 80.0% | +30.0 pts | +25.0 pts | $0.3028 | 6.7min | 5.10s | 6.35s |
| BootstrapFinetune | Qwen/Qwen2.5-0.5B-Instruct | 50.0% | — | +0.0 pts | $0.0000 | 1.1min | 1.16s | 1.88s |
| BetterTogether | Qwen/Qwen2.5-0.5B-Instruct | 50.0% | — | +0.0 pts | $0.0000 | 4.6min | 1.39s | 1.94s |

> The test partition is deliberately selected using only baseline errors; it is an adversarial optimizer stress test, not an unbiased estimate of generalization.

BootstrapFinetune and BetterTogether use the trainable local `Qwen/Qwen2.5-0.5B-Instruct` model. Their row-baseline uplift is valid on the same frozen split, but their absolute accuracy is not a Luna-vs-Luna comparison.

Every completed row links to a run directory in `benchmark_summary.json`. Each directory preserves the full console output, LM call history, per-example predictions, serialized program, extracted prompt, metrics, and manifest. Canonical frozen programs live in `chapter06/optimized_programs/final/` and prompts in `chapter06/results/final_prompts/`.
