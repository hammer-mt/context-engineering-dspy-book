# Chapter 6 optimizer results (Table 6-1)

All programs use the same 300-row dataset and locked pair-grouped split: 160 train, 60 validation, and 80 test rows. Optimizers select programs with the training and validation rows only; the locked test is evaluated once, after each program is frozen.

The Locked test, Uplift, Opt. cost, and Opt. time columns are the Accuracy, Uplift, Optimization cost, and Optimization time columns of Table 6-1 in the book (page 157). The remaining columns match the result block printed for each optimizer in the chapter.

| Optimizer | Baseline | Validation | Locked test | Uplift | Opt. cost | Eval. cost | Opt. time | Mean / p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unoptimized baseline | 53.75% | 55.00% | 53.75% (43/80) | +0.00 pp | $0.0000 | $0.1579 | 0.0s | 1.789s / 3.542s |
| LabeledFewShot | 53.75% | 63.33% | 67.50% (54/80) | +13.75 pp | $0.0000 | $0.2263 | 0.0s | 1.621s / 2.482s |
| BootstrapFewShot | 53.75% | 66.67% | 67.50% (54/80) | +13.75 pp | $0.0030 | $0.1919 | 4.5s | 1.647s / 2.715s |
| BootstrapRS | 53.75% | 61.67% | 65.00% (52/80) | +11.25 pp | $0.8766 | $0.1902 | 1119.1s | 1.579s / 2.663s |
| KNNFewShot | 53.75% | 71.67% | 72.50% (58/80) | +18.75 pp | $0.0000 | $0.2380 | 0.0s | 1.830s / 2.837s |
| COPRO | 53.75% | 55.00% | 56.25% (45/80) | +2.50 pp | $0.8156 | $0.2113 | 1155.8s | 2.310s / 3.242s |
| MIPROv2 | 53.75% | 80.00% | 66.25% (53/80) | +12.50 pp | $0.8548 | $0.2138 | 285.2s | 2.098s / 2.870s |
| GEPA | 53.75% | 80.00% | 80.00% (64/80) | +26.25 pp | $0.5824 | $0.0396 | 616.7s | 2.423s / 3.452s |
| SIMBA | 53.75% | 51.67% | 47.50% (38/80) | -6.25 pp | $1.1413 | $0.1589 | 321.3s | 1.711s / 2.408s |
| Ensemble | 53.75% | 65.00% | 70.00% (56/80) | +16.25 pp | $0.8881 | $0.5875 | 1151.2s | 4.803s / 6.601s |
| BootstrapFinetune | 51.25% | 70.00% | 70.00% (56/80) | +18.75 pp | $0.8651 | $0.0000 | 1026.3s | 1.526s / 1.992s |
| BetterTogether | 51.25% | 60.00% | 65.00% (52/80) | +13.75 pp | $0.8445 | $0.0000 | 1740.0s | 1.550s / 1.922s |

Baseline is the same-model baseline: 53.75% for the prompt optimizers, which run on `openai/gpt-5.6-luna`, and 51.25% for BootstrapFinetune and BetterTogether, which run on the local `Qwen/Qwen2.5-0.5B-Instruct` model.

GEPA uses DSPy's native `auto='light'` budget with Pareto candidate selection and `use_merge=False`. Every row reports one fresh uncached validation pass followed by one locked-test pass.

Machine-readable rows, paired statistics, hashes, model and version metadata, prompts, programs, predictions, cost, and timing are under `chapter06/results/expanded_notebooks/` (GEPA: `chapter06/results/gepa_light_standard/`). [`results/README.md`](results/README.md) describes the layout.
Local-model responses that could not be parsed are kept in the predictions as incorrect with `status: parse_error`; they are never dropped from a denominator.
