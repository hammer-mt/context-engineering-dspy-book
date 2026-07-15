# Chapter 6 benchmark: rerun-backed optimizer comparison

## Experimental frame

The frozen benchmark contains 74 passages in 37 human/AI semantic pairs. Pair IDs—not individual rows—are the split unit, so a source passage and its rewrite cannot leak across training, validation, and test. This rerun reused the checked-in split manifest and dataset hash; it did not regenerate or redesign the adversarial data.

The test partition is intentionally adversarial: baseline selection replicates scored 35%, 45%, and the frozen reference baseline for the optimizer comparison is 50%. Treat this as a stress test of optimization behavior, not as an unbiased estimate of real-world AI-detection accuracy.

## Results

| Optimizer | Accuracy | Uplift vs. 50.0% reference | Optimize cost | Optimize time | Mean latency | P95 latency | Reload parity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unoptimized baseline | 50.0% | +0.0 pts | $0.0000 | 0.0s | 2.00s | 3.50s | 2/3 |
| LabeledFewShot | 75.0% | +25.0 pts | $0.0000 | 0.0s | 2.29s | 3.50s | 3/3 |
| BootstrapFewShot | 75.0% | +25.0 pts | $0.0029 | 4.3s | 1.94s | 3.14s | 3/3 |
| BootstrapRS | 70.0% | +20.0 pts | $0.2860 | 6.6min | 1.79s | 2.66s | 3/3 |
| KNNFewShot | 70.0% | +20.0 pts | $0.0000 | 0.0s | 2.06s | 2.57s | 3/3 |
| COPRO | 50.0% | +0.0 pts | $0.5491 | 12.6min | 2.31s | 3.05s | 3/3 |
| MIPROv2 | 95.0% | +45.0 pts | $0.5072 | 7.6min | 1.67s | 2.87s | 3/3 |
| GEPA | 90.0% | +40.0 pts | $1.5658 | 15.0min | 1.78s | 2.32s | 3/3 |
| SIMBA | 70.0% | +20.0 pts | $0.9482 | 18.8min | 1.63s | 2.22s | 3/3 |
| Ensemble | 80.0% | +30.0 pts | $0.3028 | 6.7min | 5.10s | 6.35s | 3/3 |
| BootstrapFinetune | hardware_blocked | — | — | — | — | — | — |
| BetterTogether | hardware_blocked | — | — | — | — | — | — |

## Interpretation

MIPROv2 led the locked test set at 95%, a +45-point improvement over the frozen baseline. Among the leaders, MIPROv2 compiled fastest (7.6min) and used $0.5072 in measured optimization calls.

The strongest zero-paid-compile option was LabeledFewShot at 75%. That makes it a useful first move when iteration speed and cost matter more than extracting the final few points. Zero compile cost does not mean zero inference cost: the table separates optimization spend from evaluation latency for exactly that reason.

Inference tradeoffs are visible as well. Ensemble had the highest mean latency (5.10s), so its accuracy should be weighed against serving cost rather than read in isolation. The learned instructions and demonstrations are preserved beside each notebook, which makes qualitative inspection part of the comparison instead of treating accuracy as the only outcome.

## Reproducibility and limits

The completed rows used $4.81 in measured baseline, optimization, optimized-evaluation, and bounded reload-verification calls. Every selected run includes its manifest, console transcript, sanitized LM history, per-example predictions, metrics, serialized program, and extracted prompt under `chapter06/results/runs/full/`. Canonical programs and prompts are copied to `chapter06/optimized_programs/final/` and `chapter06/results/final_prompts/`.

Serialization was checked two ways where the optimizer ran: prompt/demo state equality after reload and bounded prediction-label parity on frozen test examples. Across the selected runs, 29/30 reloaded predictions matched their pre-serialization labels. The per-run counts are reported in the table because a mismatch from an uncached stochastic model is evidence to preserve, not a reason to retry until it disappears; this remains a bounded reproducibility check rather than a claim of global determinism.

BootstrapFinetune and BetterTogether remain hardware-blocked on the recorded Darwin/arm64 host because their weight-optimization paths require an NVIDIA CUDA training stack. Their notebooks execute safely, explain the missing result, and show the explicit full-run command; no score is imputed for unsupported hardware.

Finally, the test set has only 20 passages, so one changed prediction moves accuracy by five points. The comparison is most useful for understanding optimizer mechanisms and tradeoffs under one frozen adversarial workload, not for declaring a universal ranking.
