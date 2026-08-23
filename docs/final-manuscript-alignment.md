# Final manuscript alignment inventory

This inventory maps every approved item in `Final Manuscript Edit Guide.docx` to the companion repository. The Word manuscript is authoritative and is not stored or modified here. Historical run artifacts retain the dependency versions and outputs that produced them; “current surface” means executable code, notebook source, setup documentation, or generated dependency declarations readers use now.

## Must-fix items

| Guide item | Repository counterpart | Status |
|---|---|---|
| Ch. 2 — five-argument GEPA metric | `chapter02/dspy-tour.ipynb` | Changed `decision_match` to accept `pred_name` and `pred_trace`. |
| Ch. 3 — `dspy.load(..., allow_pickle=True)` | `chapter03/dspy-in-8-steps.ipynb` | Changed both whole-program loads; the second load is a repo-only consistency fix. |
| Ch. 4 — start splitting from `list(dataset)` | `chapter04/error-analysis-router.ipynb` | Already aligned. |
| Ch. 4 — repair locked-out reasoning string | `chapter04/error-analysis-router.ipynb` | Already syntactically valid; the notebook expresses the scenario with a valid dictionary. |
| Ch. 4 — import GSM8K from its exporting module | `chapter04/hf-datasets.ipynb` | Changed to `dspy.datasets.gsm8k`. |
| Ch. 4 — keep the SQuAD comment on one line | `chapter04/hf-datasets.ipynb` | Already aligned. |
| Ch. 4 — read `EvaluationResult.score` | `chapter04/hf-datasets.ipynb` | Already aligned. |
| Ch. 5 — repair `OutputField` description | `chapter05/human-then-llm-judge.ipynb` | Already aligned and syntactically valid. |
| Ch. 6 — imports and `NUM_THREADS` | Chapter 6 optimizer notebooks and `build_optimizer_notebooks.py` | Changed shared setup cells/generator to import the helpers used in printed compile shapes and define `NUM_THREADS = 1`. |
| Ch. 6 — Bootstrap demo counts 2/2 | BootstrapFewShot and BootstrapRS notebooks/runtime | Already aligned. |
| Ch. 6 — KNN hashed embeddings and 0/4 demos | `chapter06/knn-few-shot.ipynb`, `optimizer_runtime.py` | Already aligned. |
| Ch. 6 — remove false `dspy.Evaluate` warning | KNN notebook/docs | Already absent. |
| Ch. 6 — COPRO compiles on validation split | `chapter06/copro.ipynb`, `optimizer_runtime.py` | Already aligned. |
| Ch. 6 — actual COPRO instruction | COPRO checked-in learned prompt/program artifacts | Already aligned with the recorded run; artifacts remain unchanged. |
| Ch. 6 — SIMBA 6/2 and unchanged result | `chapter06/simba.ipynb`, builder/runtime/artifact | Settings were already aligned; explanatory text now states the saved program kept its original instruction and no demos. |
| Ch. 7 — GEPA budget for Flex | `chapter07/modules-tour.ipynb` | Added a complete five-argument metric and `auto="light"` Flex/GEPA example. |
| Ch. 7 — distinct `previous_draft` input | `chapter07/multi-stage-patterns.ipynb` | Changed signature and call. |
| Ch. 7 — local images use `Image.from_path` | `chapter07/multimodal.ipynb` | Corrected prose and example comment. |
| Ch. 9 — RLM uses `max_iters` | `chapter09/financial-analyst.ipynb` | Already aligned. |
| Ch. 10 — explicit SSE serializer | `chapter10/fastapi-invoice-api.ipynb` | Replaced `streaming_response()` with serialization for listener chunks, statuses, predictions, and `[DONE]`. |
| Ch. 10 — catch `dspy.LMError` | `chapter10/fastapi-invoice-api.ipynb` | Removed the LiteLLM exception tuple from the example. |
| Ch. 10 — spaces-only GEPA branch | `chapter10/dspyui-gradio.ipynb` | Already aligned; notebook-wide AST validation guards syntax/indentation. |
| Ch. 11 — iteration 2, not iteration 4 | No table-analysis prose in the companion notebooks | No repository counterpart. |

## Quick-cleanup items

| Guide item | Repository counterpart | Status |
|---|---|---|
| Ch. 1 — remove unsupported 1,500-to-64 claim | No equivalent claim in README or Chapter 1 notebook | No repository counterpart. |
| Preface — evergreen adoption wording | No GitHub-star/download claim in repository docs | No repository counterpart. |
| Ch. 3 — describe the dataset as 10 AI / 10 human | `data/ai_vs_human.csv` | Already aligned; regression test verifies 20 rows split 10/10. |
| Ch. 4 — deterministic HotPotQA/GSM8K examples | `chapter04/hf-datasets.ipynb` | Added the corrected deterministic first-example comments. |
| Ch. 5/10 — internally consistent GPT-5.6-luna arithmetic | No matching detailed token-cost arithmetic in repository artifacts | No repository counterpart; coarse README estimates remain unchanged. |
| Ch. 6 — DSPy 3.3.0 wording | Current Chapter 6 generator/notebook/provider prose | Changed current surfaces to 3.3.0. Historical artifacts still truthfully record 3.2.1. |
| Ch. 6 — BetterTogether can improve | `chapter06/better-together.ipynb`, builder | Softened the use-case wording to testing whether the combination can outperform either method. |
| Ch. 7 — JSONAdapter comment | `chapter07/adapters.ipynb` | Already aligned. |
| Ch. 8 — remove open-source Mem0 `graph_store` | `chapter08/history-mem0-rlm.ipynb` | Config was already clean; added the Mem0 Platform graph-memory note. |
| Ch. 8 — RLM introduced in Chapter 2 | No chapter-number cross-reference in the companion notebook | No repository counterpart. |
| Ch. 9 — table compares three approaches | No Table 9-2 caption in repository artifacts | No repository counterpart. |
| Ch. 10 — Markdown output fields only | `chapter10/fastapi-invoice-api.ipynb` | Corrected prose/comments and adapter behavior so inputs/terminator retain DSPy markers while outputs use Markdown headings. |
| Ch. 10 — replace Chapter 7.4.5 pointer | `chapter10/fastapi-invoice-api.ipynb` | Replaced with “Building Custom Adapters” wording. |
| Ch. 10 — SentimentClassifier not from Chapter 9 | `chapter10/mlflow-tracking.ipynb` | Removed the false attribution. |
| Ch. 11 — `cli_metric` source | `chapter11/image-cli-optimizer.ipynb` | Already aligned: the companion notebook defines `cli_metric`. |

## Repository-only correctness work

- Added `mcp>=1,<2` to the core environment so `uv run python chapter08/mcp_server.py` works after the documented fresh-clone setup, and pinned `huggingface-hub<1` for the locked Transformers/Datasets stack.
- Updated the Chapter 8 MCP notebook bootstrap flow, setup checker, lockfile, generated `requirements.txt`, and README dependency description.
- Added static and executable regression coverage for the alignment contract, including SSE event serialization and Markdown-adapter input/output marker behavior.
