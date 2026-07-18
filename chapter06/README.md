# Chapter 6: optimizer notebooks

This download contains one concise, executable notebook per optimizer. Every
notebook loads and validates the same frozen dataset and split, explains the
optimizer, contains its real DSPy execution path, reports final test accuracy,
and previews the published program artifact.

Choose **Run All** for the fast teaching path: it validates the data and displays
the checked-in result without making API calls. To actually compile and evaluate
an optimizer again, set `CHAPTER06_RUN_LIVE=1` before starting Jupyter. Live prompt
runs require `OPENAI_API_KEY`; live weight runs also require the local
PyTorch/Transformers/TRL/PEFT stack and can take several minutes.

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

BootstrapFinetune and BetterTogether run through DSPy with
`Qwen/Qwen2.5-0.5B-Instruct`, DSPy's `LocalProvider`, Transformers, TRL, and PEFT
on Apple Silicon MPS. The thin `MacLocalProvider` subclass changes only local
serving and one TRL argument name; DSPy still owns trace formatting and training.
A stronger Sol teacher produces candidate traces, and a validation guard rejects
one-class trace sets before local training starts. Those two rows share the frozen
split but use a different task model from the Luna prompt optimizers.

The corrected BootstrapFinetune rerun accepted 16 human and 14 AI traces. It
trained successfully through DSPy on MPS and reached 55% final test accuracy
versus the 50% Qwen baseline. Training settings were selected on the training and
validation partitions before the frozen test result was recorded.

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
