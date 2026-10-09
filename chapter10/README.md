# Chapter 10 — Putting DSPy into Production

The production examples are organized into three notebooks, one for each hands-on section of the chapter:

- `mlflow-tracking.ipynb` — "Monitoring and Observability with MLflow": tracing, optimizer autologging, experiments, and the MLflow MCP server.
- `fastapi-invoice-api.ipynb` — "Deployment with FastAPI": the route, save/load, prompt export, guardrails, async, streaming, fallbacks, and caching.
- `dspyui-gradio.ipynb` — "Building a Gradio User Interface (DSPyUI)": a runnable, minimal DSPyUI workflow.

Complete the [repository setup](../README.md#quick-start-recommended), then run
the notebooks from this directory:

```bash
uv run jupyter lab
```

MLflow needs the local server described in the root README. The Redis and
Gradio sections install their optional packages in the notebook before use.
Live model calls require the corresponding provider keys in the root `.env`.

## Files the notebooks create

Running the notebooks writes a few files into this directory:

- `programs/` — saved program state (`invoice_extractor_optimized.json` from the FastAPI notebook, `latest.json` from DSPyUI).
- `invoice_extractor/` — the whole-program save from "Saving and Loading Programs".
- `evaluation_results.csv` — the artifact logged in "Tracking Experiments".
- `api_server.py` — "FastAPI Route" asks you to save its listing under this name so that `uvicorn api_server:app` can serve it.

All of them are ignored by Git and safe to delete.

## Tests

The checks in `tests/` run offline, without API keys or services. From the
repository root:

```bash
uv run python -m unittest discover -s chapter10/tests -t .
```
