# Chapter 6 benchmark: chapter-ready replacement

## The experiment

The original detector started at 72.3%, which left too little room to show how the
optimizers differ. I replaced the made-up examples with 37 balanced semantic pairs
(74 passages) drawn from genuine, permissively licensed pre-2022 technical
documentation. Each human passage has a meaning-preserving AI rewrite. Human and
rewrite remain in the same split, preventing topic leakage.

The final test partition is intentionally adversarial. I selected it from repeated
errors made by the unoptimized task model, then froze it before any optimizer ran.
The two selection replicates scored 35% and 45%; a separate confirmation run scored
40%. This is useful as an optimizer stress test, but it is not an unbiased estimate
of AI-detection accuracy in the wild.

Every optimizer used the same frozen 36-row training set, 18-row validation set,
20-row test set, GPT-5.6 Luna task model, and GPT-5.6 Sol reflection model where the
optimizer required one. The table reports percentage-point uplift against the
independently confirmed 40% reference baseline. Because uncached model responses
are stochastic, each run's contemporaneous baseline is also retained in the JSON
results, rather than silently treating those reruns as identical.

## Results

| Optimizer | Accuracy | Uplift | Relative uplift | Optimize cost | Optimize time | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unoptimized baseline | 40% | — | — | $0.0000 | 0.0s | 2.26s | 4.11s |
| LabeledFewShot | 50% | +10 pts | +25.0% | $0.0000 | 0.0s | 2.20s | 2.85s |
| BootstrapFewShot | 75% | +35 pts | +87.5% | $0.0026 | 6.4s | 4.05s | 13.62s |
| BootstrapRS | 65% | +25 pts | +62.5% | $0.2960 | 10.1min | 2.64s | 3.63s |
| KNNFewShot | 75% | +35 pts | +87.5% | $0.0000 | 0.0s | 2.97s | 3.84s |
| COPRO | 65% | +25 pts | +62.5% | $0.5296 | 16.3min | 2.77s | 3.71s |
| **MIPROv2** | **90%** | **+50 pts** | **+125.0%** | **$0.4881** | **12.9min** | **2.43s** | **3.72s** |
| **GEPA** | **90%** | **+50 pts** | **+125.0%** | **$1.3275** | **25.8min** | **2.25s** | **2.99s** |
| SIMBA | 65% | +25 pts | +62.5% | $0.8292 | 25.2min | 2.19s | 3.32s |
| Ensemble | 85% | +45 pts | +112.5% | $0.2988 | 8.0min | 7.14s | 10.38s |
| BootstrapFinetune | Hardware blocked | — | — | — | — | — | — |
| BetterTogether | Hardware blocked | — | — | — | — | — | — |

The sophisticated optimizers finally separate—but not merely because they are
sophisticated. MIPROv2 and GEPA both reached 90%, 25 points above BootstrapRS and
50 points above the reference baseline. MIPROv2 was the better value: it matched
GEPA's accuracy for roughly 37% of GEPA's optimization cost and half its optimization
time. GEPA, however, produced the cleanest standalone instruction: it learned an
explicit distinction between ordinary technical polish and *systematic lexical
substitution and rhythmic smoothing*, then calibrated that distinction with
contrasting passage pairs.

The simpler methods still matter. BootstrapFewShot reached 75% for less than one
cent of optimization spend and in 6.4 seconds. KNNFewShot matched it without paid
compile calls. Either is a compelling first move when cost or iteration speed
matters more than the final 15 points.

Complexity alone did not guarantee a win. COPRO and SIMBA both stopped at 65%, the
same test accuracy as BootstrapRS despite taking longer and costing more to
optimize. SIMBA selected a program that scored 88.9% on its final training pass but
only 65% on the locked test set. That gap is exactly why optimizer comparisons need
a holdout that remains untouched during search.

The ensemble reached 85%, but its production tradeoff is visible immediately: mean
inference latency rose from 2.26 seconds to 7.14 seconds because every prediction
runs three component programs. It is a reasonable choice when errors are expensive,
but MIPROv2 and GEPA were both more accurate and much faster at inference here.

## What the learned prompts reveal

MIPROv2 combined a rewritten instruction with two demonstrations. Its instruction
frames the task comparatively: AI rewrites tend to preserve identifiers and
procedure while making the prose longer, smoother, more formal, and more balanced.

GEPA used no demonstrations in its final program. Instead, it produced a detailed,
auditable instruction with paired calibration examples. Its strongest learned rule
was not to confuse domain terminology with human provenance: precise technical
content can survive an AI paraphrase, so the classifier should focus on elevated
substitutions, formulaic transitions, and systematic sentence-level smoothing.

SIMBA retained the original one-line instruction and added two positive
demonstrations. Those examples captured some useful signals, but they did not give
the model the same explicit decision boundary learned by MIPROv2 and GEPA. On this
small adversarial dataset, instruction optimization generalized better than SIMBA's
selected demonstrations.

The exact final prompts are preserved in `chapter06/results/final_prompts/`, and
the reloadable programs are in `chapter06/optimized_programs/final/`. The complete
console output, sanitized LM history, predictions, metrics, and run manifest for
each optimizer are indexed by `chapter06/results/benchmark_summary.json`.

## Cost and limitations

The entire experiment—including dataset generation, adversarial screening, smoke
tests, full runs, and final evaluations—cost **$7.53**, well below the authorized
$100. No OpenAI rate-limit or exhausted-credit error terminated a run. A single
source-side Wikipedia 429 occurred while gathering candidates; that source was
abandoned rather than retried.

The test set contains only 20 passages and was selected to defeat this baseline, so
small count changes move accuracy by five points and the scores should not be read
as universal optimizer rankings. BootstrapFinetune and BetterTogether also require
an NVIDIA CUDA host; their notebooks passed static and fail-fast smoke validation,
but reporting invented Apple Silicon scores would be misleading.
