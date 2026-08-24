# Chapter 6 result artifacts

`expanded_notebooks/comparison.json` is the machine-readable source of truth for
the results presented in the manuscript. Each completed row points to the exact
program, prompt, result, and prediction artifacts used for that published row.
`../CHAPTER_RESULTS.md` is the corresponding human-readable table.

Raw run directories remain checked in for reproducibility and provenance. A raw
directory that is not selected by `comparison.json` is historical evidence, not
an alternative published result. In particular, the manuscript's COPRO row uses
`expanded_notebooks/copro/rerun-20260719-124750/`; the earlier `copro/full/` run is
retained only as history.

`expanded_notebooks/run_ledger.json` inventories completed smoke, full, rerun,
and failed-preflight attempts. Its `manuscript_canonical` field identifies the run
selected for the chapter. GEPA's standard light-budget artifacts live under
`gepa_light_standard/` and its exploratory history remains under `gepa_expanded/`.

Do not add separately copied “final prompt” snapshots. They drift from the saved
programs and make it unclear which run supports the manuscript. Read the prompt
path from the canonical comparison row instead.
