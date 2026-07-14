# Chapter 6 optimizer benchmark

Frozen reference baseline: **40.0%**. Adversarial selection replicates: 35.0%, 45.0%.

| Optimizer | Accuracy | Uplift vs. 40.0% reference | Relative uplift | Optimize cost | Optimize time | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unoptimized baseline | 40.0% | +0.0 pts | +0.0% | $0.0000 | 0.0s | 2.26s | 4.11s |
| LabeledFewShot | 50.0% | +10.0 pts | +25.0% | $0.0000 | 0.0s | 2.20s | 2.85s |
| BootstrapFewShot | 75.0% | +35.0 pts | +87.5% | $0.0026 | 6.4s | 4.05s | 13.62s |
| BootstrapRS | 65.0% | +25.0 pts | +62.5% | $0.2960 | 10.1min | 2.64s | 3.63s |
| KNNFewShot | 75.0% | +35.0 pts | +87.5% | $0.0000 | 0.0s | 2.97s | 3.84s |
| COPRO | 65.0% | +25.0 pts | +62.5% | $0.5296 | 16.3min | 2.77s | 3.71s |
| MIPROv2 | 90.0% | +50.0 pts | +125.0% | $0.4881 | 12.9min | 2.43s | 3.72s |
| GEPA | 90.0% | +50.0 pts | +125.0% | $1.3275 | 25.8min | 2.25s | 2.99s |
| SIMBA | 65.0% | +25.0 pts | +62.5% | $0.8292 | 25.2min | 2.19s | 3.32s |
| Ensemble | 85.0% | +45.0 pts | +112.5% | $0.2988 | 8.0min | 7.14s | 10.38s |
| BootstrapFinetune | hardware_blocked | — | — | — | — | — | — |
| BetterTogether | hardware_blocked | — | — | — | — | — | — |

> The test partition is deliberately selected using only baseline errors; it is an adversarial optimizer stress test, not an unbiased estimate of generalization.

Every completed row links to a run directory in `benchmark_summary.json`. Each directory preserves the full console output, LM call history, per-example predictions, serialized program, extracted prompt, metrics, and manifest. Canonical frozen programs live in `chapter06/optimized_programs/final/` and prompts in `chapter06/results/final_prompts/`.
