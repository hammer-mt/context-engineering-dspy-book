# context-engineering-dspy-book

Companion code for **Context Engineering with DSPy** (O'Reilly).

## About

*Context Engineering with DSPy* is a practical guide to building reliable AI systems. It moves beyond "prompt engineering" to **Context Engineering**: the art of providing the right information to LLMs in the right format. Using the [DSPy](https://dspy.ai/) framework, you'll build modular, self-improving AI programs that are robust against changes in models and data — DSPy handles the low-level plumbing and prompt optimization so you can focus on designing the logic and flow of your application.

## Chapter coverage

This repo currently covers **7 of 11 chapters** — the ones with notebooks the author has already finished. The remaining four (Ch 1, 4, 5, 7, 8, 11) are tracked separately and are coming soon.

| Chapter | Notebook(s) | Status |
|---|---|---|
| Ch 1 — Introduction to Context Engineering | — | coming soon |
| Ch 2 — Introduction to DSPy | `chapter02/dspy-quickstart.ipynb`, `chapter02/formatted_prompt.ipynb` | ready |
| Ch 3 — DSPy in 8 Steps | `chapter03/dspy-in-8-steps.ipynb`, `chapter03/humanize-quickstart.ipynb` | ready |
| Ch 4 — Strategies for Collecting Datasets | — | coming soon |
| Ch 5 — Formalizing Evaluation Metrics | — | coming soon |
| Ch 6 — Deepdive into Prompt Optimizers | `chapter06/` (9 notebooks: benchmark + per-optimizer satellites) | ready |
| Ch 7 — Customizing DSPy Programs | — | coming soon |
| Ch 8 — Building AI Agents | — | coming soon |
| Ch 9 — Real-World Use Cases | `chapter09/` (7 notebooks, one per use case) | ready |
| Ch 10 — Optimizing Coding Agents with DSPy | `chapter10/` (6 notebooks for §10.3.4 + §10.4; §10.2 + §10.3 coming soon) | partial |
| Ch 11 — Getting Ready for Production | — | coming soon |

> **Heads up — Chapter 10 path references.** The printed book occasionally references these Ch 10 notebooks as `working/<filename>.ipynb` (from the author's draft repo). In this repo they live at `chapter10/<filename>.ipynb`. The mapping is in [the Ch 10 reference table below](#chapter-10--book-path-mapping).

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

### Environment variables

Copy `.env.example` to `.env` and fill in your API keys. The book uses these:

| Variable | Required for |
|---|---|
| `OPENAI_API_KEY` | Most notebooks (Ch 2, 3, 6, 9, 10) |
| `ANTHROPIC_API_KEY` | Provider-switch demos (Ch 2) + several Ch 10 notebooks |
| `OPENROUTER_API_KEY` | Ch 10 §10.4 notebooks (clawsona, skill-discovery, etc. — they call `openrouter/anthropic/claude-opus-4.7`) |
| `GOOGLE_API_KEY` | Optional — Gemini provider-switch demos |
| `FAL_KEY` | Ch 9 §9.5 (fashion video generator) only |
| `SERPER_API_KEY` | Ch 9 §9.4 (news researcher) only |

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
| MLflow | Ch 3 (optional tracing) | `mlflow server --backend-store-uri sqlite:///mydb.sqlite --host 127.0.0.1 --port 5000` |
| Arbor | Ch 6 §6.3.3 GRPO only | See [Arbor docs](https://github.com/Ziems/arbor); GPU required |
| Deno | Ch 9 §9.7 financial analyst (PoT mode only) | `curl -fsSL https://deno.land/install.sh \| sh` |
| GPU (CUDA / MPS) | Ch 6 fine-tuning notebooks | Local NVIDIA GPU for `grpo.ipynb`/`better-together.ipynb`; MPS for `finetune-mac-m3.ipynb` |

The model slugs in notebooks (e.g., `openai/gpt-5-mini`, `openai/gpt-5.5-pro`, `anthropic/claude-opus-4.7`, `gemini/gemini-2.5-flash`) are preserved verbatim from the book. Use the slugs as-is.

## Running the notebooks

Each notebook is self-contained. From the repo root:

```bash
jupyter lab
```

Open the chapter directory, pick a notebook, run the first install cell (`%pip install -r ../requirements.txt -q`), then run cells top-to-bottom. Heavy optimizer or agent notebooks ship with smoke-test budget caps in their default cells — full reproduction cells are commented as opt-in so you don't accidentally burn through API budget.

Approximate per-chapter LLM spend if you run every notebook end-to-end with default smoke caps:

| Chapter | Estimated cost |
|---|---|
| Ch 3 | $0.20–0.50 |
| Ch 6 | $2–5 (heavy with full optimizer runs; smoke caps keep it under $5) |
| Ch 9 | $0.50–1.50 (depends on which use cases you run) |
| Ch 10 | $1–3 (smoke caps; full GEPA runs can run $10+ per notebook if opted in) |

## Chapter 10 — book path mapping

The printed book sometimes references Ch 10 §10.4 notebooks at paths like `working/clawsona-dspy.ipynb` (from the author's draft repo). In this companion repo they live under `chapter10/`:

| Book reference | Repo location |
|---|---|
| `working/clawsona-dspy.ipynb` | `chapter10/clawsona-dspy.ipynb` |
| `working/skill-discovery-rlm.ipynb` | `chapter10/skill-discovery-rlm.ipynb` |
| `working/cli-generation-skill.ipynb` | `chapter10/cli-generation-skill.ipynb` |
| `working/optimize-tool-descriptions.ipynb` | `chapter10/optimize-tool-descriptions.ipynb` |
| `working/test-agents-md.ipynb` | `chapter10/test-agents-md.ipynb` |

The book text will be updated to point at `chapter10/...` in a future printing.

## Contributing

Bug reports, fixes, and improvements welcome. Before opening a PR:

- Clear notebook outputs (see Security note above).
- Match the existing notebook conventions (top markdown cell linking to chapter section, `%pip install -r ../requirements.txt -q` first cell, env-var documentation).
- Keep model slugs verbatim — never silently substitute one slug for another.

## License

MIT — see [LICENSE](./LICENSE).
