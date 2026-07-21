# Chapter 10 — Putting DSPy into Production

The production examples are organized into three notebooks:

- `mlflow-tracking.ipynb` — tracing, optimizer autologging, experiments, and the MLflow MCP server.
- `fastapi-invoice-api.ipynb` — save/load, prompt export, guardrails, async, streaming, fallbacks, and caching.
- `dspyui-gradio.ipynb` — a runnable, minimal DSPyUI workflow.

Complete the [repository setup](../README.md#quick-start-recommended), then run
the notebooks from this directory:

```bash
uv run jupyter lab
```

MLflow needs the local server described in the root README. The Redis and
Gradio sections install their optional packages in the notebook before use.
Live model calls require the corresponding provider keys in the root `.env`.
