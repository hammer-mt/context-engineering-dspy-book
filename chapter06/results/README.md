# Chapter 6 results

The results in this folder are the runs printed in the chapter. Earlier
exploratory runs are kept on the branch `archive/chapter06-earlier-runs` of the
same repository.

## Where the numbers printed in the chapter live

| Printed in the book | Saved here |
|---|---|
| Table 6-1 (page 157) and the result block printed for each optimizer | `expanded_notebooks/comparison.json` (also as `expanded_notebooks/comparison.csv` and [`../CHAPTER_RESULTS.md`](../CHAPTER_RESULTS.md)) |
| Every optimizer except GEPA: scores, cost, timing, prompt, program, predictions | `expanded_notebooks/<optimizer>/full/` |
| GEPA: scores, cost, timing, prompt, program, predictions | `gepa_light_standard/` |

Each row of `expanded_notebooks/comparison.json` names the exact program, prompt,
result, and prediction files it was built from. The notebooks read that file, so
the numbers they display are the numbers in the chapter.

## Folder guide

- `expanded_notebooks/<optimizer>/full/` holds the run reported in the chapter:
  `result.json` (scores, usage, cost, timing), `optimized_program.json` (reloadable
  DSPy program), `learned_prompt.json` and `learned_instruction.txt` (the prompt in
  readable form), `validation_predictions.jsonl`, `test_predictions.jsonl`, and
  `run_manifest.json` (dataset and split hashes, seed, versions).
- `expanded_notebooks/run_ledger.json` lists every run saved under
  `expanded_notebooks/` and `gepa_light_standard/` with its cost and timing. Its
  `published_in_chapter` field marks the runs reported in the chapter.
- `gepa_light_standard/` holds the GEPA run reported in the chapter (DSPy's
  `auto='light'` budget): the compile under `run/`, then one validation pass and
  one locked-test pass under `validation/` and `test/`.
- `gepa_expanded/` holds no optimizer run. It records how the benchmark's test
  split was frozen, and the code reads it: `locked_test.json` and
  `freeze_summary.json` record the frozen test split, `baseline_confirmation/`
  is the three-pass check of the unoptimized program made when the split was
  frozen, and `budget_ledger.json` is the spending guard for the opt-in paid
  GEPA run, which starts at zero. The 50.0% majority-vote score in
  `baseline_confirmation/` belongs to that check; the chapter's 53.75% baseline
  is the single pass in `expanded_notebooks/quickstart/full/`.

A run you start yourself with `python -m chapter06.run_live_optimizer` is written
to `expanded_notebooks/<optimizer>/<mode>/`. The default `--mode smoke` is a
quick trial on 8 training and 4 validation rows that never reads the test split.
A `--mode full` run replaces the saved files in `full/`, so pass `--artifact-dir`
with a folder of your own to keep the results printed in the chapter intact.

Each `result.json` and `run_manifest.json` records the DSPy version that
produced the run: 3.2.1 for every optimizer except GEPA, which ran on 3.3.0. The
saved programs load into `AIDetector` under the pinned DSPy 3.3.0 (DSPy prints a
version notice when it loads a file saved by another release), and the notebooks
read them with that version.

The paired statistics in `expanded_notebooks/comparison.json`
(`paired_mcnemar_p_value`, the `paired_bootstrap_ci_*` fields, and
`statistical_analysis`) pair every optimizer's locked-test predictions with the
GPT-5.6-luna baseline predictions in `expanded_notebooks/quickstart/full/`. For
BootstrapFinetune and BetterTogether that is a comparison across models: their
uplift in the chapter is measured against the 51.25% baseline of the local Qwen
student, whose predictions are stored under `baseline` in each run's own
`result.json`. These statistics are not printed in the book.

`expanded_notebooks/ensemble/full/optimized_program.json` holds only the first
of the ensemble's three component programs. DSPy skips sub-programs that are
already marked as compiled when it saves a parent program, so the
BootstrapFewShot and BootstrapFewShotWithRandomSearch components of that run
were not written to the file. The ensemble's scores and predictions are
complete.

To rebuild `comparison.json`, `comparison.csv`, `run_ledger.json`, and
`../CHAPTER_RESULTS.md` from the saved runs without any API calls:

```bash
uv run python -m chapter06.build_expanded_comparison
```
