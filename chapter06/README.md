# Chapter 6 notebooks

The completed experiment, interpretation, and manuscript-ready comparison are in
[`CHAPTER_RESULTS.md`](CHAPTER_RESULTS.md). Machine-generated tables and canonical
artifacts remain under `results/`.

Each optimizer has its own concise, executed notebook. Opening a notebook and
choosing **Run All** defaults to `inspect` mode: it reads the checked-in full-run
artifacts, prints score/cost/time/latency, previews the learned prompt and demos,
and makes no API calls. This makes the downloaded notebooks useful even before a
reader installs DSPy or configures credentials.

For a live run, install the repository requirements, configure `OPENAI_API_KEY`,
and launch Jupyter with `CHAPTER06_RUN=smoke` for a bounded code-path check or
`CHAPTER06_RUN=full` for the complete frozen split. The paid/full path is
deliberately opt-in; it is never triggered by the notebook's default execution.

The default configuration targets stable DSPy 3.2.1 and uses:

- `openai/gpt-5.6-luna` for task execution
- `openai/gpt-5.6-sol` for reflection, judging, and prompt proposal

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

Every completed run preserves the console transcript, sanitized LM history,
per-example predictions, metrics, manifest, serialized program, and extracted
prompt under `results/runs/`. The summarizer copies canonical programs to
`optimized_programs/final/`, canonical prompts to `results/final_prompts/`, and
generates the chapter-facing benchmark table in `results/benchmark_summary.md`.
The same rows are also exported as `results/benchmark_summary.csv` and structured
`results/benchmark_summary.json`. It also rebuilds `CHAPTER_RESULTS.md` from the
selected reruns. Completed runs verify prompt/demo equality after serialization
and prediction-label parity on a bounded test subset.

To regenerate and publication-check the notebooks after rebuilding the summary:

```bash
python -m chapter06.build_optimizer_notebooks
python -m chapter06.execute_optimizer_notebooks
python -m chapter06.validate_notebooks
```

The first command intentionally clears outputs, the second really executes every
notebook in safe mode from the repository root, and the final command rejects
unexecuted cells or missing saved output.

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
| `bootstrap-finetune.ipynb` | BootstrapFinetune (CUDA) |
| `better-together.ipynb` | BetterTogether (CUDA) |

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
