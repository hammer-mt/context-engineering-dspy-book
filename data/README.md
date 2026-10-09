# Datasets

Shared datasets for the AI-versus-human text classifier that the book builds in
Chapter 3 and benchmarks in Chapter 6. The CSV files are small enough to open in
a spreadsheet or load with `pandas.read_csv`; the JSON files record how the
Chapter 6 data is split.

| File | Rows | Used by | What it contains |
|---|---|---|---|
| `ai_vs_human.csv` | 20 (10 AI, 10 human) | Chapter 3, `chapter03/dspy-in-8-steps.ipynb` | The dataset from "Example Dataset" in Chapter 3. Columns: `text`, `is_ai`, `notes`. |
| `ai_vs_human200.csv` | 202 (101 AI, 101 human) | Optional | A larger version of the Chapter 3 dataset with the same three columns. It includes all 20 rows of `ai_vs_human.csv`. |
| `ai_vs_human_chapter06_expanded.csv` | 300 (150 AI, 150 human) | Chapter 6, every optimizer notebook | The benchmark behind the Chapter 6 optimizer comparison: 150 human passages, each paired with an AI rewrite. |
| `ai_vs_human_chapter06_expanded_splits.json` | n/a | Chapter 6 | The fixed train (160 rows), validation (60 rows), and test (80 rows) split of the 300-row file, with the dataset hash. |
| `ai_vs_human_chapter06.csv` | 74 (37 AI, 37 human) | Chapter 6 dataset tooling | The seed subset that the 300-row benchmark grew from. All 74 rows are included unchanged in the 300-row file. |
| `ai_vs_human_chapter06_splits.json` | n/a | Chapter 6 dataset tooling | The split and hash that belong to the 74-row file. |

## Notes

- **Chapter 3.** The book loads `ai_vs_human.csv` from the notebook's own folder,
  so `chapter03/ai_vs_human.csv` is an identical copy of `data/ai_vs_human.csv`.
  To experiment with larger training and validation sets, point `csv_path` in
  the notebook at `../data/ai_vs_human200.csv`. The results printed in the book
  use the 20-row file.
- **Chapter 6.** The two Chapter 6 CSV files add provenance columns to `text`,
  `is_ai`, and `notes`: `pair_id`, `example_id`, `source_id`, `source_url`,
  `source_title`, `source_author`, `license`, `generation_model`, and
  `parent_example_id`. The splits are defined by `pair_id`, so a human passage
  and its AI rewrite always land in the same partition. Sources and licenses
  are listed in [`chapter06/data_sources_expanded.yaml`](../chapter06/data_sources_expanded.yaml),
  and [`chapter06/README.md`](../chapter06/README.md) describes the benchmark.
- Do not edit these files if you want to reproduce the numbers printed in the
  book: the Chapter 6 code checks the dataset hash recorded in the split files.
