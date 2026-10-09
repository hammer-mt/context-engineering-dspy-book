# Chapter 6: optimizer notebooks

This folder contains one short, executable notebook per optimizer covered in
Chapter 6, "Deep Dive into Prompt Optimizers". Every notebook loads and validates
the same frozen dataset and split, explains the optimizer, contains its DSPy
code, reports locked-test accuracy, and previews the saved optimized program.

Choose **Run All** for the fast path: it validates the data and displays the
saved result without making API calls. Each notebook holds the chapter's listings
as code cells. A listing that calls a model API sits unchanged under
`if RUN_LIVE:` and is skipped by default; the saved result is displayed either
way. Where the chapter prints output under a listing, such as an optimizer's
result block, the notebook quotes it in a cell headed **Output printed in the
book**, next to the saved result, so you can compare the two. To compile and
evaluate an optimizer yourself, set `CHAPTER06_RUN_LIVE=1`
before starting Jupyter. Every live run requires `OPENAI_API_KEY` in the
repository `.env` file, including the two fine-tuning notebooks, whose teacher is
`openai/gpt-5.6-sol`; when the key is missing, the first code cell stops with a
message. The two fine-tuning notebooks also train a local model with PyTorch,
Transformers, TRL, and PEFT, and download `Qwen/Qwen2.5-0.5B-Instruct` (about
1 GB) from Hugging Face on their first live run; the published runs took about
17 minutes (BootstrapFinetune) and 29 minutes (BetterTogether) on Apple Silicon.

Most listings evaluate on one thread, as printed (`NUM_THREADS = 1`), so a live
run can be slower than the recorded one: allow five to ten minutes to score a
compiled program, about 20 minutes for the MIPROv2 compile, about 50 minutes for
the SIMBA compile, and about 70 minutes for the COPRO compile. Each notebook
states its own figures.

The chapter imports its detector from this folder (page 135):
`AIDetector` in `optimizer_runtime.py` wraps `dspy.ChainOfThought` around a
`DetectAIText` signature. Its wording differs slightly from the Chapter 3
listing, and every saved program starts from it.

## Results

The full comparison behind Table 6-1 is in [`CHAPTER_RESULTS.md`](CHAPTER_RESULTS.md).
Saved programs, prompts, predictions, hashes, cost, and timing are under
`results/expanded_notebooks/` (GEPA: `results/gepa_light_standard/`).
[`results/README.md`](results/README.md) explains the layout and which folder
holds the numbers printed in the chapter.

The results in `results/` are the runs printed in the chapter. Earlier
exploratory runs are kept on the branch `archive/chapter06-earlier-runs` of the
same repository.

Fine-tuned model weights are not included in the repository; each fine-tuning
run's `model_artifact_manifest.json` records the file hashes and sizes instead.
No credentials are stored. The GEPA folders include the full model request and
response history so the reflection steps can be inspected.

## Benchmark data

The frozen benchmark in `data/ai_vs_human_chapter06_expanded.csv` contains 300
passages in 150 human/AI semantic pairs. Pair IDs, not rows, define the 160-row
train, 60-row validation, and 80-row locked-test split, so a source passage and
its rewrite never land in different partitions. Exact membership and the dataset
hash are in `data/ai_vs_human_chapter06_expanded_splits.json`; source and license
information is in `data_sources_expanded.yaml`.

`data/ai_vs_human_chapter06.csv` is a 74-row seed subset of the same benchmark.
Its rows are included unchanged in the 300-row file, and every optimizer result
in the chapter uses the 300-row file.

The test partition is deliberately hard for the unoptimized baseline. It is
useful for teaching optimizer behavior, not for estimating general-purpose
AI-detector accuracy.

The scripts in `experiments/gepa_expanded/` that build the dataset
(`build_dataset.py`, `prepare_candidates.py`, and `write_sources.py`) document how
the frozen CSV was produced. They cannot be rerun from this repository because
their intermediate inputs are not included. The notebooks need only the frozen
CSV and split file, which they validate by hash.

## Local fine-tuning

BootstrapFinetune and BetterTogether run through DSPy with
`Qwen/Qwen2.5-0.5B-Instruct`, DSPy's `LocalProvider`, Transformers, TRL, and PEFT
on Apple Silicon MPS, falling back to the CPU. The thin `MacLocalProvider`
subclass in `apple_finetune.py` changes only local serving and one TRL argument
name; DSPy still owns trace formatting and training. An `openai/gpt-5.6-sol`
teacher produces candidate traces, and a validation guard rejects one-class trace
sets before local training starts. These two rows share the frozen split but use
a different task model from the `openai/gpt-5.6-luna` prompt optimizers, so they
are compared against their own 51.25% same-model baseline.

The published BootstrapFinetune run accepted 77 human and 63 AI traces and
reached 70% locked-test accuracy. BetterTogether combines a validation-selected
prompt stage with the same weight-training path. In both cases, training and
selection use only the training and validation partitions before the locked test
is evaluated once.

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
| `gepa.ipynb` | GEPA and the chapter's custom `WordLimitProposer` |
| `simba.ipynb` | SIMBA |
| `ensemble.ipynb` | Ensemble |
| `bootstrap-finetune.ipynb` | BootstrapFinetune (local Apple Silicon MPS or CPU) |
| `better-together.ipynb` | BetterTogether `p -> w` (local Apple Silicon MPS or CPU) |
| `gepa-expanded-dataset-experiment.ipynb` | walk-through of the GEPA run: dataset design, baseline, saved results, and an opt-in paid run |

## Printed listings and recorded runs

In live mode the notebooks run the listings exactly as printed. The saved
results were recorded with the runner in `optimizer_runtime.py`
(`run_optimizer`; the GEPA result with the runner in
`experiments/gepa_expanded/`), which passed a few arguments the printed listings
do not show. A live run of a listing is therefore a new experiment, not a repeat
of the recorded one, and its score, cost, and time will differ. Each notebook
repeats the relevant line next to its saved result.

| Notebook | Recorded run, compared with the printed listing |
|---|---|
| `bootstrap-few-shot.ipynb`, `bootstrap-random-search.ipynb` | also passed `max_errors=1` |
| `knn-few-shot.ipynb` | also passed `max_bootstrapped_demos=0, max_labeled_demos=4`, so each prediction made one model call |
| `copro.ipynb` | passed the 60-example validation split as COPRO's `trainset` |
| `miprov2.ipynb` | fixed `seed=42` and used DSPy's default thread count |
| `gepa.ipynb` | same optimizer settings; the metric returned a fixed feedback sentence per outcome instead of `example.notes` |
| `simba.ipynb` | used DSPy's default thread count |
| `better-together.ipynb` | passed `num_threads=1` and a `max_errors` limit to `compile`; the notebook sets both with `dspy.configure` |

The other notebooks (baseline, LabeledFewShot, Ensemble, BootstrapFinetune) were
recorded with the arguments shown in their listings.

## Running an optimizer from the command line

The runner that produced the saved results is also available from the command
line (the saved GEPA result came from the separate runner that
`gepa-expanded-dataset-experiment.ipynb` walks through):

```bash
uv run python -m chapter06.run_live_optimizer --help
```

A live run makes paid API calls. The default `--mode smoke` is a quick trial on
8 training and 4 validation examples that never reads the test split;
`--mode full` uses the whole split. Results are written to
`results/expanded_notebooks/<optimizer>/<mode>/`. A `--mode full` run replaces
the saved files in `full/`, so pass `--artifact-dir` with a folder of your own to
keep the published results intact.

To rebuild `CHAPTER_RESULTS.md` and `results/expanded_notebooks/comparison.json`
from the saved runs (no API calls):

```bash
uv run python -m chapter06.build_expanded_comparison
```

## KNNFewShot in DSPy 3.x

`KNNFewShot` is different from the other few-shot optimizers. Supply `k`,
`trainset`, and `vectorizer` to the constructor, then compile with only the
student, as in the listing on page 140:

```python
# Set up the optimizer and training data
optimizer = dspy.teleprompt.KNNFewShot(
        k=4,
        metric=exact_match,
        trainset=trainset,
        vectorizer=dspy.Embedder(hashed_ngram_embeddings),

    )


optimized_detector = optimizer.compile(
    AIDetector()
)
```

The published KNNFewShot result additionally passed `max_bootstrapped_demos=0,
max_labeled_demos=4`, so the four retrieved neighbors are used directly as labeled
demonstrations with no extra model calls.

Do not pass `num_threads` to `KNNFewShot`. DSPy forwards unknown constructor
arguments to `BootstrapFewShot`, which does not accept it.

## GEPA version

The locked environment uses GEPA 0.1.1, the version DSPy 3.3.0 requires. The
parallel proposal strategies mentioned on page 146 (`SameParentSampling`,
`IndependentSampling`, `PxNSampling`) arrived in GEPA 0.1.4, so they cannot be
imported in this environment. Every GEPA result in the chapter uses classic
single-mutation GEPA, which is what the pinned version runs.
