# Chapter 6: optimizer notebooks

This download contains one concise, executed notebook per optimizer. Each notebook
explains when to use the optimizer, shows the essential DSPy compile call, reports
the frozen Chapter 6 result, and previews the learned instruction and demonstrations.
Choose **Run All** to inspect the checked-in artifacts locally; the notebooks make
no network calls and require no API key.

The full comparison and its limitations are in [`CHAPTER_RESULTS.md`](CHAPTER_RESULTS.md).
Canonical program snapshots and prompts are under `optimized_programs/final/`
and `results/final_prompts/`. Large provider transcripts, smoke-run duplicates,
caches, model adapters, and temporary training files are intentionally not part of the student
download.

## Benchmark data

The frozen benchmark in `data/ai_vs_human_chapter06.csv` contains 74 passages in
37 human/AI semantic pairs. Pair IDs—not rows—define the split, preventing a source
passage and its rewrite from leaking across train, validation, and test. Exact split
membership and the dataset hash are in `data/ai_vs_human_chapter06_splits.json`;
source and license information is in `data_sources.yaml`.

The test partition is deliberately baseline-adversarial. It is useful for teaching
optimizer behavior, not for estimating general-purpose AI-detector accuracy.

BootstrapFinetune and BetterTogether were run through DSPy with
`Qwen/Qwen2.5-0.5B-Instruct`, Transformers, TRL, and PEFT on Apple Silicon MPS.
Those two rows share the frozen split but are reported separately from the Luna
prompt-optimizer comparison because the base model differs.

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

## DSPy 3.x note for KNNFewShot

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
arguments to `BootstrapFewShot`, which does not accept it.
