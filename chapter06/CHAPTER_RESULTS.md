# Chapter 6 expanded-dataset optimizer results

All programs use the canonical 300-row dataset and locked pair-grouped split: 160 train, 60 validation, and 80 test rows. Optimizer selection used validation only; the locked test was released after each program was frozen.

| Optimizer | Baseline | Validation | Locked test | Uplift | Opt. cost | Eval. cost | Opt. time | Mean / p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unoptimized baseline | 53.75% | 55.00% | 53.75% (43/80) | +0.00 pp | $0.0000 | $0.1579 | 0.0s | 1.789s / 3.542s |
| LabeledFewShot | 53.75% | 63.33% | 67.50% (54/80) | +13.75 pp | $0.0000 | $0.2263 | 0.0s | 1.621s / 2.482s |
| BootstrapFewShot | 53.75% | 66.67% | 67.50% (54/80) | +13.75 pp | $0.0030 | $0.1919 | 4.5s | 1.647s / 2.715s |
| BootstrapFewShotWithRandomSearch (BootstrapRS) | 53.75% | 61.67% | 65.00% (52/80) | +11.25 pp | $0.8766 | $0.1902 | 1119.1s | 1.579s / 2.663s |
| KNNFewShot | 53.75% | 71.67% | 72.50% (58/80) | +18.75 pp | $0.0000 | $0.2380 | 0.0s | 1.830s / 2.837s |
| COPRO | 53.75% | 53.33% | 50.00% (40/80) | -3.75 pp | >=$0.0732* | unavailable* | 861.1s | 1.711s / 2.658s |
| MIPROv2 | 53.75% | 76.67% | 66.25% (53/80) | +12.50 pp | >=$0.3052* | unavailable* | 270.8s | 1.628s / 2.874s |
| GEPA | 50.00% | 88.33% | 76.25% (61/80) | +26.25 pp | $10.8412 | $0.6086 | 2023.1s | 2.060s / 3.125s |
| SIMBA | 53.75% | 51.67% | 47.50% (38/80) | -6.25 pp | $1.1413 | $0.1589 | 321.3s | 1.711s / 2.408s |
| Ensemble | 53.75% | 65.00% | 70.00% (56/80) | +16.25 pp | $0.8881 | $0.5875 | 1151.2s | 4.803s / 6.601s |
| BootstrapFinetune (Apple Silicon / MPS) | 51.25% | 70.00% | 70.00% (56/80) | +18.75 pp | $0.8651 | $0.0000 | 1026.3s | 1.526s / 1.992s |
| BetterTogether (Apple Silicon / MPS) | 51.25% | 60.00% | 65.00% (52/80) | +13.75 pp | $0.8445 | $0.0000 | 1740.0s | 1.550s / 1.922s |

GEPA is the frozen PR #8 result and uses three fresh uncached evaluation passes with per-example majority vote. Newly executed rows report one uncached pass; that protocol difference is retained explicitly rather than normalized away.

`*` COPRO and MIPROv2 optimization cost is a recorded lower bound, and their evaluation cost is unavailable: those runs preceded the shared-history fix for deep-copied DSPy language models. Their scores and wall-clock timings remain valid.

Machine-readable rows, paired statistics, hashes, model/version metadata, prompts, programs, predictions, cost, timing, and failure manifests are under `chapter06/results/expanded_notebooks/`.
Local-model responses that could not be parsed are retained in the predictions as incorrect with `status: parse_error`; they are never dropped from a denominator.
