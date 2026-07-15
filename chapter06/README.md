# Chapter 6 notebooks

The completed experiment, interpretation, and manuscript-ready comparison are in
[`CHAPTER_RESULTS.md`](CHAPTER_RESULTS.md). Machine-generated tables and canonical
artifacts remain under `results/`.

Each optimizer has its own self-contained notebook. Run the install cell and then
execute the notebook top-to-bottom, so one optimizer failure cannot cancel every
other experiment.

The default configuration targets stable DSPy 3.2.1 and uses:

- `openai/gpt-5.6-luna` for task execution
- `openai/gpt-5.6-sol` for reflection, judging, and prompt proposal

BootstrapFinetune and BetterTogether are the exception: they use the trainable
`Qwen/Qwen2.5-0.5B-Instruct` model locally through Transformers, TRL, and PEFT.
On Apple Silicon, PyTorch selects MPS; CPU is also a supported fallback. These
rows use the same frozen train/validation/test membership, but their accuracy is
not directly comparable to the Luna rows because the base model differs.

Override those defaults with `TASK_MODEL` and `REFLECTION_MODEL`. Default dataset
and optimizer budgets are intentionally small. Set `TRAIN_LIMIT=0`, `VAL_LIMIT=0`,
and `EVAL_LIMIT=0` to use the complete frozen split.

## Benchmark dataset

The benchmark uses `data/ai_vs_human_chapter06.csv`: 37 balanced semantic pairs
(74 rows) drawn from genuine permissively licensed, pre-2022 technical
documentation and paired with meaning-preserving AI rewrites. Pair IDs are the
split unit, so a human passage and its rewrite can never leak across train,
validation, and test. The exact membership and dataset hash are frozen in
`data/ai_vs_human_chapter06_splits.json`; source URLs and license bases live in
`data_sources.yaml`.

The test set is deliberately baseline-adversarial. It was selected using repeated
unoptimized-model errors, then independently confirmed before optimizer runs. It
is therefore an optimizer stress test, not an unbiased estimate of real-world AI
detection accuracy. The gate, repeated predictions, rejected low-quality rows,
and full disclosure are preserved under `results/dataset/`.

## Reproducing the experiment

The orchestration entry point runs one optimizer at a time, applies bounded API
timeouts/retries, records spend against a $95 enforced ceiling, and stops on an
escaped rate-limit or exhausted-credit error:

```bash
python -m chapter06.run_optimizer_suite --profile smoke
python -m chapter06.run_optimizer_suite --profile full --skip-existing
python -m chapter06.summarize_optimizer_results
```

For local weight optimization, these optional controls keep the workflow explicit:

```bash
CHAPTER06_FINETUNE_DEVICE=mps          # auto (default), mps, or cpu
CHAPTER06_FINETUNE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
CHAPTER06_FINETUNE_TEACHER=local       # self-teacher; remote is opt-in
```

No OpenAI key is needed when `CHAPTER06_FINETUNE_TEACHER=local`. The smoke
profile performs one bounded LoRA step before the full profile's 18-step run.

Every completed run preserves the console transcript, sanitized LM history,
per-example predictions, metrics, manifest, serialized program, and extracted
prompt under `results/runs/`. The summarizer copies canonical programs to
`optimized_programs/final/`, canonical prompts to `results/final_prompts/`, and
generates the chapter-facing benchmark table in `results/benchmark_summary.md`.
The same rows are also exported as `results/benchmark_summary.csv` and structured
`results/benchmark_summary.json`.

## Notebook map

| Notebook | Purpose |
|---|---|
| `quickstart-ai-detector.ipynb` | shared unoptimized baseline |
| `labeled-few-shot.ipynb` | LabeledFewShot |
| `bootstrap-few-shot.ipynb` | BootstrapFewShot |
| `bootstrap-random-search.ipynb` | BootstrapRS (alias for BootstrapFewShotWithRandomSearch) |
| `knn-few-shot.ipynb` | KNNFewShot |
| `copro.ipynb` | COPRO |
| `miprov2.ipynb` | MIPROv2 |
| `gepa.ipynb` | GEPA and the chapter's custom WordLimitProposer |
| `simba.ipynb` | SIMBA |
| `ensemble.ipynb` | Ensemble |
| `bootstrap-finetune.ipynb` | BootstrapFinetune (local Apple Silicon MPS or CPU) |
| `better-together.ipynb` | BetterTogether `p -> w` (local Apple Silicon MPS or CPU) |

## KNNFewShot on DSPy 3.x

`KNNFewShot` is different from the other few-shot optimizers. Supply `k`,
`trainset`, and `vectorizer` to the constructor, then compile with only the student:

```python
optimizer = dspy.KNNFewShot(
    k=4,
    trainset=trainset,
    vectorizer=vectorizer,
    metric=exact_match,
    max_bootstrapped_demos=2,
    max_labeled_demos=0,
)
optimized_detector = optimizer.compile(detector)
```

Do not pass `num_threads` to `KNNFewShot`. DSPy forwards unknown constructor
arguments to `BootstrapFewShot`, which does not accept it; that is the source of
`KNNFewShot.__init__() got an unexpected keyword argument 'num_threads'`-style
evaluation failures.
