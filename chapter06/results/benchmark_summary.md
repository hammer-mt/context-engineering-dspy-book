# Chapter 6 optimizer benchmark

Frozen reference baseline: **50.0%**. Adversarial selection replicates: 35.0%, 45.0%.

| Optimizer | Accuracy | Uplift vs. 50.0% reference | Relative uplift | Optimize cost | Optimize time | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unoptimized baseline | 50.0% | +0.0 pts | +0.0% | $0.0000 | 0.0s | 2.00s | 3.50s |
| LabeledFewShot | 75.0% | +25.0 pts | +50.0% | $0.0000 | 0.0s | 2.29s | 3.50s |
| BootstrapFewShot | 75.0% | +25.0 pts | +50.0% | $0.0029 | 4.3s | 1.94s | 3.14s |
| BootstrapRS | 70.0% | +20.0 pts | +40.0% | $0.2860 | 6.6min | 1.79s | 2.66s |
| KNNFewShot | 70.0% | +20.0 pts | +40.0% | $0.0000 | 0.0s | 2.06s | 2.57s |
| COPRO | 50.0% | +0.0 pts | +0.0% | $0.5491 | 12.6min | 2.31s | 3.05s |
| MIPROv2 | 95.0% | +45.0 pts | +90.0% | $0.5072 | 7.6min | 1.67s | 2.87s |
| GEPA | 90.0% | +40.0 pts | +80.0% | $1.5658 | 15.0min | 1.78s | 2.32s |
| SIMBA | 70.0% | +20.0 pts | +40.0% | $0.9482 | 18.8min | 1.63s | 2.22s |
| Ensemble | 80.0% | +30.0 pts | +60.0% | $0.3028 | 6.7min | 5.10s | 6.35s |
| BootstrapFinetune | hardware_blocked | — | — | — | — | — | — |
| BetterTogether | hardware_blocked | — | — | — | — | — | — |

> The test partition is deliberately selected using only baseline errors; it is an adversarial optimizer stress test, not an unbiased estimate of generalization.

Every completed row links to a run directory in `benchmark_summary.json`. Each directory preserves the full console output, LM call history, per-example predictions, serialized program, extracted prompt, metrics, and manifest. Canonical frozen programs live in `chapter06/optimized_programs/final/` and prompts in `chapter06/results/final_prompts/`.
