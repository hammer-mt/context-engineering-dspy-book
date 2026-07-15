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
and 20-row test set. Prompt optimizers used GPT-5.6 Luna as the task model and
GPT-5.6 Sol where reflection was required. Weight optimizers need a model whose
weights can actually be trained, so BootstrapFinetune and BetterTogether used
Qwen2.5-0.5B-Instruct locally on Apple Silicon MPS with a local self-teacher.

The Luna rows report percentage-point uplift against the independently confirmed
40% reference baseline. The two Qwen rows report uplift against their own 50%
contemporaneous Qwen baseline. Those rows share the exact frozen data partitions,
but their absolute scores are not directly comparable to the Luna rows because
the base model differs.

## Results

| Optimizer | Task model | Accuracy | Uplift vs model reference | Optimize cost | Optimize time | Mean latency | P95 latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Unoptimized baseline | GPT-5.6 Luna | 40% | — | $0.0000 | 0.0s | 2.26s | 4.11s |
| LabeledFewShot | GPT-5.6 Luna | 50% | +10 pts | $0.0000 | 0.0s | 2.20s | 2.85s |
| BootstrapFewShot | GPT-5.6 Luna | 75% | +35 pts | $0.0026 | 6.4s | 4.05s | 13.62s |
| BootstrapRS | GPT-5.6 Luna | 65% | +25 pts | $0.2960 | 10.1min | 2.64s | 3.63s |
| KNNFewShot | GPT-5.6 Luna | 75% | +35 pts | $0.0000 | 0.0s | 2.97s | 3.84s |
| COPRO | GPT-5.6 Luna | 65% | +25 pts | $0.5296 | 16.3min | 2.77s | 3.71s |
| **MIPROv2** | **GPT-5.6 Luna** | **90%** | **+50 pts** | **$0.4881** | **12.9min** | **2.43s** | **3.72s** |
| **GEPA** | **GPT-5.6 Luna** | **90%** | **+50 pts** | **$1.3275** | **25.8min** | **2.25s** | **2.99s** |
| SIMBA | GPT-5.6 Luna | 65% | +25 pts | $0.8292 | 25.2min | 2.19s | 3.32s |
| Ensemble | GPT-5.6 Luna | 85% | +45 pts | $0.2988 | 8.0min | 7.14s | 10.38s |
| BootstrapFinetune | Qwen2.5-0.5B | 50% | +0 pts | $0.0000 | 1.2min | 1.21s | 1.85s |
| BetterTogether | Qwen2.5-0.5B | 50% | +0 pts | $0.0000 | 4.8min | 1.58s | 2.22s |

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

## Local weight optimization on Apple Silicon

The earlier CUDA assumption was incorrect. DSPy's stock BootstrapFinetune is
device-agnostic: it bootstraps accepted chat traces and calls `LM.finetune()`, which
delegates training to the LM provider. DSPy 3.2.1 has no maintained local
Transformers provider, so Chapter 6 adds only that missing boundary. The optimizer
and its public `compile(student, trainset, teacher=...)` workflow remain DSPy's.

The bounded smoke runs each performed one LoRA optimizer step on `mps` before the
full runs. Full BootstrapFinetune retained 17 of 36 bootstrapped traces and ran 18
MPS optimizer steps in 12.78 seconds (70.70 seconds end-to-end optimization). Full
BetterTogether evaluated the original, `p`, and `p -> w` strategies, then performed
the same 17-example, 18-step MPS fine-tune. Its `p -> w` adapter tied the original
at 50% validation accuracy, so DSPy correctly selected the original as the final
program. The trained candidate and its prompt are still preserved under that run's
`candidate_programs/` directory; this is a completed optimization, not a substituted
optimizer or a fabricated winning result.

Both full runs reported `mps_built=true`, `mps_available=true`, and
`trainer_device=mps`. Fresh program loads preserved prompt, model reference, and a
deterministic sentinel prediction. The local runs made no paid API calls.

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

The test set contains only 20 passages and was selected to defeat the Luna baseline,
so small count changes move accuracy by five points and the scores should not be read
as universal optimizer rankings. The local Qwen weight-optimizer rows also answer a
different model-specific question from the Luna prompt-optimizer rows. MPS training
uses full-precision base weights plus a LoRA adapter and requires enough unified
memory for the selected model; CPU is legitimate but slower. BetterTogether's final
program remained the original because all three validation candidates tied, even
though its `p -> w` training stage genuinely completed.
