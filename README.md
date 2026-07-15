# context-engineering-dspy-book

Companion code for **Context Engineering with DSPy** (O'Reilly).

## About

*Context Engineering with DSPy* is a practical guide to building reliable AI systems. It moves beyond "prompt engineering" to **Context Engineering**: the art of providing the right information to LLMs in the right format. Using the [DSPy](https://dspy.ai/) framework, you'll build modular, self-improving AI programs that are robust against changes in models and data — DSPy handles the low-level plumbing and prompt optimization so you can focus on designing the logic and flow of your application.

## Chapter coverage

| Chapter | Notebook(s) |
|---|---|
| Ch 1 — Introduction to Context Engineering | `chapter01/hello-dspy.ipynb` |
| Ch 2 — Introduction to DSPy | `chapter02/dspy-tour.ipynb` |
| Ch 3 — DSPy in 8 Steps | `chapter03/dspy-in-8-steps.ipynb`, `chapter03/humanize-quickstart.ipynb` |
| Ch 4 — Strategies for Collecting Datasets | `chapter04/` — 5 notebooks (error analysis, HF, Kaggle, synthetic, PII) |
| Ch 5 — Formalizing Evaluation Metrics | `chapter05/` — 5 notebooks (string/regex, semantic, BLEU/ROUGE/F1, human+LLM judge, rubric+multi-predictor) |
| Ch 6 — Deep Dive into Prompt Optimizers | `chapter06/` — one self-contained notebook per optimizer, plus baseline and platform appendices |
| Ch 7 — Customizing DSPy Programs | `chapter07/` — 8 notebooks (modules, ReAct, PoT/CodeAct/RLM, multi-stage, parallel, multimodal, adapters) |
| Ch 8 — Building AI Agents | `chapter08/` — 7 notebooks (ReAct basics, framework comparison, MCP, RAG in-memory, RAG Qdrant, web search + multi-hop, history/Mem0/RLM) |
| Ch 9 — Real-World Use Cases | `chapter09/` — 7 notebooks (one per use case) |
| Ch 10 — Optimizing Coding Agents with DSPy | `chapter10/` — 5 notebooks (landing-page skill optimizer, image-CLI optimizer, RLM skill discovery, AGENTS.md testing, Clawsona persona) |
| Ch 11 — Getting Ready for Production | `chapter11/` — 3 notebooks (MLflow tracking, FastAPI invoice API, Gradio DSPyUI) |

> **Heads up — Chapter 10 path references.** The printed book occasionally references Ch 10 notebooks as `working/<filename>.ipynb` (from the author's draft repo). In this repo they live at `chapter10/<filename>.ipynb`. The mapping is in [the Ch 10 reference table below](#chapter-10--book-path-mapping).

## Setup

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager (recommended)

### Install

```bash
# 1. Install uv (skip if you already have it)
# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone (current home — may move to oreilly-media in future)
git clone https://github.com/hammer-mt/context-engineering-dspy-book.git
cd context-engineering-dspy-book

# 3. Create venv and install
uv venv .venv
source .venv/bin/activate    # Windows: .\.venv\Scripts\activate
uv pip install -r requirements.txt
```

A few notebooks need additional dependencies (LangGraph, CrewAI, Mem0, Qdrant client, MCP, Redis, Gradio, MLflow MCP extras). Those install at the top of their own notebook so the default install stays small.

### Environment variables

Copy `.env.example` to `.env` and fill in your API keys. The book uses these:

| Variable | Required for |
|---|---|
| `OPENAI_API_KEY` | Most notebooks (Ch 1–11 setup) |
| `ANTHROPIC_API_KEY` | Provider-switch demos (Ch 2), several Ch 10 notebooks, Anthropic prompt-cache examples in Ch 11 |
| `OPENROUTER_API_KEY` | Ch 10 §10.4 notebooks (call `openrouter/anthropic/claude-opus-4.7`) |
| `GOOGLE_API_KEY` | Optional — Gemini provider-switch demos |
| `FAL_KEY` | Ch 9 §9.5 (fashion video generator) only |
| `SERPER_API_KEY` | Ch 8 web search; Ch 9 §9.4 news researcher |

`.env` is gitignored. You should never commit it.

### Security note for contributors and forkers

`.gitignore` prevents `.env` from being staged accidentally, but it does **not** stop `git add -f .env`. Before committing or pushing notebook changes:

1. Clear all output cells (`Cell → All Output → Clear` in Jupyter, or `nbstripout`).
2. Confirm no inline string literals look like API keys: `grep -rE "(sk-[a-zA-Z0-9]{20,}|sk-proj-|AIza[0-9A-Za-z_-]{30,})" chapter*/`.
3. Confirm `.env` is not staged: `git status`.

If you ever accidentally commit a key, **rotate it immediately** at the provider dashboard. Git history retains the leak even after deletion.

## Optional services

A few chapters need an extra service running locally. All commands below use loopback binding so the service is not exposed publicly — do not change `127.0.0.1` to `0.0.0.0` unless you understand the security implications.

| Service | Chapter | Setup |
|---|---|---|
| MLflow | Ch 3 (optional tracing), Ch 11 §11.1 | `mlflow server --backend-store-uri sqlite:///mydb.sqlite --host 127.0.0.1 --port 5000` |
| Qdrant | Ch 8 RAG variant | `docker run --rm -p 127.0.0.1:6333:6333 qdrant/qdrant` |
| Redis | Ch 11 §11.2 cache variant | `docker run --rm -p 127.0.0.1:6379:6379 redis:alpine` |
| Deno | Ch 7 §7.1.7–7.1.8 (PoT, CodeAct); Ch 9 §9.7 financial analyst PoT mode | `curl -fsSL https://deno.land/install.sh \| sh` |
| MCP server | Ch 8 §8.2.5 MCP integration | Reader-provided. See https://modelcontextprotocol.io |
| Playwright browsers | Ch 10 landing-page optimizer | `playwright install` |
| Claude Code CLI | Ch 10 landing-page + image-CLI optimizers | https://docs.claude.com/en/docs/claude-code |
| Apple Silicon MPS or CPU | Ch 6 fine-tuning examples | PyTorch/Transformers/TRL/PEFT support the BootstrapFinetune and BetterTogether compile patterns; inspecting the published results requires no model download |

## Running the notebooks

Each notebook is self-contained. From the repo root:

```bash
jupyter lab
```

Open a chapter directory, pick a notebook, run the first install cell (`%pip install -r ../requirements.txt -q`), then run cells top-to-bottom. Heavy optimizer or agent notebooks ship with smoke-test budget caps in their default cells — full reproduction cells are commented as opt-in so you don't accidentally burn through API budget.

Approximate per-chapter LLM spend if you run every notebook end-to-end with default smoke caps:

| Chapter | Estimated cost |
|---|---|
| Ch 1 | under $0.10 |
| Ch 2 tour | $0.30–0.80 |
| Ch 3 | $0.20–0.50 |
| Ch 4 | $0.30–1.00 (HF + synthetic generation) |
| Ch 5 | $0.30–1.00 (judge training is the heaviest) |
| Ch 6 | $5–15 if every Luna/Sol optimizer notebook is run with smoke caps; full runs can cost substantially more |
| Ch 7 | $0.50–2.00 |
| Ch 8 | $1–3 (framework comparison + multi-hop agents) |
| Ch 9 | $0.50–1.50 (depends on which use cases you run) |
| Ch 10 | $1–3 with smoke caps; full GEPA runs can run $10+ per notebook if opted in |
| Ch 11 | $0.30–1.00 |

## Chapter 10 — book path mapping

The printed book references Ch 10 §10.3 notebooks at paths like `working/clawsona-dspy.ipynb` (from the author's draft repo). In this companion repo they live under `chapter10/`:

| Book reference | Repo location |
|---|---|
| `working/clawsona-dspy.ipynb` | `chapter10/clawsona-dspy.ipynb` |
| `working/skill-discovery-rlm.ipynb` | `chapter10/skill-discovery-rlm.ipynb` |
| `working/test-agents-md.ipynb` | `chapter10/test-agents-md.ipynb` |

The book text will be updated to point at `chapter10/...` in a future printing.

## Contributing

Bug reports, fixes, and improvements welcome. Before opening a PR:

- Clear notebook outputs (see Security note above).
- Match the existing notebook conventions (top markdown cell linking to chapter section, `%pip install -r ../requirements.txt -q` first cell, env-var documentation).
- Keep model slugs aligned with the book unless a documented compatibility update (such as the Chapter 6 Luna/Sol migration) deliberately changes them.
- If you change a class or signature that's duplicated across notebooks (see `docs/duplication-registry.yaml`), update every copy.

## License

MIT — see [LICENSE](./LICENSE).
