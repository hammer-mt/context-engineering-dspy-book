# Chapter 6 optimizer benchmark

Frozen reference baseline: **40.0%**. Adversarial selection replicates: 35.0%, 45.0%.

| Optimizer | Task model | Accuracy | Uplift vs. 40.0% Luna reference | Run uplift | Optimize cost | Optimize time | Mean latency | P95 latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Unoptimized baseline | openai/gpt-5.6-luna | 40.0% | +0.0 pts | +0.0 pts | $0.0000 | 0.0s | 2.26s | 4.11s |
| LabeledFewShot | openai/gpt-5.6-luna | 50.0% | +10.0 pts | +0.0 pts | $0.0000 | 0.0s | 2.20s | 2.85s |
| BootstrapFewShot | openai/gpt-5.6-luna | 75.0% | +35.0 pts | +35.0 pts | $0.0026 | 6.4s | 4.05s | 13.62s |
| BootstrapRS | openai/gpt-5.6-luna | 65.0% | +25.0 pts | +15.0 pts | $0.2960 | 10.1min | 2.64s | 3.63s |
| KNNFewShot | openai/gpt-5.6-luna | 75.0% | +35.0 pts | +20.0 pts | $0.0000 | 0.0s | 2.97s | 3.84s |
| COPRO | openai/gpt-5.6-luna | 65.0% | +25.0 pts | +30.0 pts | $0.5296 | 16.3min | 2.77s | 3.71s |
| MIPROv2 | openai/gpt-5.6-luna | 90.0% | +50.0 pts | +45.0 pts | $0.4881 | 12.9min | 2.43s | 3.72s |
| GEPA | openai/gpt-5.6-luna | 90.0% | +50.0 pts | +40.0 pts | $1.3275 | 25.8min | 2.25s | 2.99s |
| SIMBA | openai/gpt-5.6-luna | 65.0% | +25.0 pts | +15.0 pts | $0.8292 | 25.2min | 2.19s | 3.32s |
| Ensemble | openai/gpt-5.6-luna | 85.0% | +45.0 pts | +40.0 pts | $0.2988 | 8.0min | 7.14s | 10.38s |
| BootstrapFinetune | Qwen/Qwen2.5-0.5B-Instruct | 50.0% | — | +0.0 pts | $0.0000 | 1.2min | 1.21s | 1.85s |
| BetterTogether | Qwen/Qwen2.5-0.5B-Instruct | 50.0% | — | +0.0 pts | $0.0000 | 4.8min | 1.58s | 2.22s |

> The test partition is deliberately selected using only baseline errors; it is an adversarial optimizer stress test, not an unbiased estimate of generalization.

BootstrapFinetune and BetterTogether use the trainable local `Qwen/Qwen2.5-0.5B-Instruct` model. Their row-baseline uplift is valid on the same frozen split, but their absolute accuracy is not a Luna-vs-Luna comparison.

Every completed row links to a run directory in `benchmark_summary.json`. Each directory preserves the full console output, LM call history, per-example predictions, serialized program, extracted prompt, metrics, and manifest. Canonical frozen programs live in `chapter06/optimized_programs/final/` and prompts in `chapter06/results/final_prompts/`.
