"""Result blocks printed in Chapter 6, "Deep Dive into Prompt Optimizers".

Each entry holds the page number and the lines printed under an optimizer's
listing; a block that continues on the next page has (first page, last page).
``test_notebooks.py`` compares them with the results saved under
``chapter06/results/`` and with the blocks the notebooks quote, so the
notebooks keep showing the numbers a reader sees in the book.
"""

from __future__ import annotations


# Optimizer name -> (page, printed lines).
PRINTED_RESULT_BLOCKS: dict[str, tuple[int | tuple[int, int], str]] = {
    "labeled-few-shot": (
        137,
        """
============================================================
OPTIMIZER: LabeledFewShot
============================================================
Validation Accuracy:   63.33%
Locked Test Accuracy:  67.50% (54/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       +13.75 percentage points
------------------------------------------------------------
Optimization Cost:     $0.0000
Evaluation Cost:       $0.2263
Optimization Time:     0.0s
Mean / p95 Latency:    1.621s / 2.482s
============================================================
""",
    ),
    "bootstrap-few-shot": (
        138,
        """
============================================================
OPTIMIZER: BootstrapFewShot
============================================================
Validation Accuracy:   66.67%
Locked Test Accuracy:  67.50% (54/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       +13.75 percentage points
------------------------------------------------------------
Optimization Cost:     $0.0030
Evaluation Cost:       $0.1919
Optimization Time:     4.5s
Mean / p95 Latency:    1.647s / 2.715s
============================================================
""",
    ),
    "bootstrap-random-search": (
        (139, 140),
        """
============================================================
OPTIMIZER: BootstrapFewShotWithRandomSearch
============================================================
Validation Accuracy:   61.67%
Locked Test Accuracy:  65.00% (52/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       +11.25 percentage points
------------------------------------------------------------
Optimization Cost:     $0.8766
Evaluation Cost:       $0.1902
Optimization Time:     1119.1s
Mean / p95 Latency:    1.579s / 2.663s
============================================================
""",
    ),
    "knn-few-shot": (
        141,
        """
============================================================
OPTIMIZER: KNNFewShot
============================================================
Validation Accuracy:   71.67%
Locked Test Accuracy:  72.50% (58/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       +18.75 percentage points
------------------------------------------------------------
Optimization Cost:     $0.0000
Evaluation Cost:       $0.2380
Optimization Time:     0.0s
Mean / p95 Latency:    1.830s / 2.837s
============================================================
""",
    ),
    "copro": (
        142,
        """
============================================================
OPTIMIZER: COPRO
============================================================
Validation Accuracy: 55.0% (33/60)
Locked Test Accuracy: 56.3% (45/80)
Same-Model Baseline:  53.8% (43/80)
Accuracy Uplift:      +2.5 percentage points
------------------------------------------------------------
Total Tokens:         383,539
  - Prompt:           266,047
  - Completion:       117,492
Optimization Cost:    $0.8156
Evaluation Cost:      $0.2113
Total Recorded Cost:  $1.0269
Optimization Time:    1155.8s (19.3 minutes)
Mean / p95 Latency:   2.310s / 3.242s
============================================================
""",
    ),
    "miprov2": (
        144,
        """
============================================================
OPTIMIZER: MIPROv2
============================================================
Validation Accuracy: 80.0% (48/60)
Locked Test Accuracy: 66.3% (53/80)
Same-Model Baseline:  53.8% (43/80)
Accuracy Uplift:      +12.5 percentage points
------------------------------------------------------------
Total Tokens:         565,949
  - Prompt:           471,085
  - Completion:       94,864
Optimization Cost:    $0.8548
Evaluation Cost:      $0.2138
Total Recorded Cost:  $1.0685
Optimization Time:    285.2s (4.8 minutes)
Mean / p95 Latency:   2.098s / 2.870s
============================================================
""",
    ),
    "gepa": (
        147,
        """
============================================================
OPTIMIZER: GEPA
============================================================
Validation Accuracy:   80.00% (48/60)
Locked Test Accuracy:  80.00% (64/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       +26.25 percentage points
------------------------------------------------------------
Optimization Cost:     $0.5824
Evaluation Cost:       $0.0396
Total Recorded Cost:   $0.6221
Optimization Time:     616.7s (10.3 minutes)
Mean / p95 Latency:    2.423s / 3.452s
============================================================
""",
    ),
    "simba": (
        150,
        """
============================================================
OPTIMIZER: SIMBA
============================================================
Validation Accuracy:   51.67%
Locked Test Accuracy:  47.50% (38/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       -6.25 percentage points
------------------------------------------------------------
Optimization Cost:     $1.1413
Evaluation Cost:       $0.1589
Optimization Time:     321.3s
Mean / p95 Latency:    1.711s / 2.408s
============================================================
""",
    ),
    "bootstrap-finetune": (
        152,
        """
============================================================
OPTIMIZER: BootstrapFinetune
============================================================
Student Model:         Qwen/Qwen2.5-0.5B-Instruct
Teacher Model:         openai/gpt-5.6-sol
Validation Accuracy:   70.00%
Locked Test Accuracy:  70.00% (56/80)
Same-Model Baseline:   51.25% (41/80)
Accuracy Uplift:       +18.75 percentage points
Accepted Traces:       77 human / 63 AI
------------------------------------------------------------
Optimization Cost:     $0.8651
Evaluation Cost:       $0.0000
Optimization Time:     1026.3s
Mean / p95 Latency:    1.526s / 1.992s
============================================================
""",
    ),
    "better-together": (
        (153, 154),
        """
============================================================
OPTIMIZER: BetterTogether
============================================================
Student Model:         Qwen/Qwen2.5-0.5B-Instruct
Teacher Model:         openai/gpt-5.6-sol
Validation Accuracy:   60.00%
Locked Test Accuracy:  65.00% (52/80)
Same-Model Baseline:   51.25% (41/80)
Accuracy Uplift:       +13.75 percentage points
Accepted Traces:       77 human / 55 AI
------------------------------------------------------------
Optimization Cost:     $0.8445
Evaluation Cost:       $0.0000
Optimization Time:     1740.0s
Mean / p95 Latency:    1.550s / 1.922s
============================================================
""",
    ),
    "ensemble": (
        156,
        """
============================================================
TRANSFORMATION: Ensemble
============================================================
Validation Accuracy:   65.00%
Locked Test Accuracy:  70.00% (56/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       +16.25 percentage points
------------------------------------------------------------
Optimization Cost:     $0.8881
Evaluation Cost:       $0.5875
Optimization Time:     1151.2s
Mean / p95 Latency:    4.803s / 6.601s
============================================================
""",
    ),
}
