# context-engineering-dspy-book

Companion code for **Context Engineering with DSPy** (O'Reilly).

## About

*Context Engineering with DSPy* is a practical guide to building reliable AI systems. It moves beyond "prompt engineering" to **Context Engineering**: the art of providing the right information to LLMs in the right format. Using the [DSPy](https://dspy.ai/) framework, you'll build modular, self-improving AI programs that are robust against changes in models and data — DSPy handles the low-level plumbing and prompt optimization so you can focus on designing the logic and flow of your application.

## Chapter coverage

Each chapter of the book has its own folder. Chapter titles are the ones printed in the book.

| Chapter | Folder | Notebooks |
|---|---|---|
| Ch 1 — Introduction to Context Engineering | `chapter01/` | 1 notebook: `hello-dspy.ipynb` |
| Ch 2 — Introduction to DSPy | `chapter02/` | 1 notebook: `dspy-tour.ipynb` |
| Ch 3 — DSPy in Eight Steps | `chapter03/` | 2 notebooks: `dspy-in-8-steps.ipynb` (the chapter's listings) and `humanize-quickstart.ipynb` (a shorter supplementary quickstart; its code is not printed in the book). `ai_vs_human.csv` is the chapter's dataset |
| Ch 4 — Strategies for Collecting Datasets | `chapter04/` | 5 notebooks: `error-analysis-router.ipynb`, `hf-datasets.ipynb`, `kaggle-imdb.ipynb`, `pii-synthesizer.ipynb`, `synthetic-distillation.ipynb` |
| Ch 5 — Evaluation Metrics | `chapter05/` | 5 notebooks: `string-and-regex-metrics.ipynb`, `semantic-similarity.ipynb`, `bleu-rouge-f1.ipynb`, `human-then-llm-judge.ipynb`, `rubric-and-multipredictor.ipynb`. `judge_labels_sample.csv` is sample data for `human-then-llm-judge.ipynb` |
| Ch 6 — Deep Dive into Prompt Optimizers | `chapter06/` | 13 notebooks: the unoptimized baseline (`quickstart-ai-detector.ipynb`); one notebook per optimizer (`labeled-few-shot.ipynb`, `bootstrap-few-shot.ipynb`, `bootstrap-random-search.ipynb`, `knn-few-shot.ipynb`, `copro.ipynb`, `miprov2.ipynb`, `gepa.ipynb`, `simba.ipynb`, `ensemble.ipynb`, `bootstrap-finetune.ipynb`, `better-together.ipynb`); and a GEPA experiment on a larger benchmark (`gepa-expanded-dataset-experiment.ipynb`). Shared code is in `optimizer_runtime.py` and `notebook_support.py`; see [`chapter06/README.md`](./chapter06/README.md) |
| Ch 7 — Custom Modules, Types, and Adapters | `chapter07/` | 9 notebooks: `modules-tour.ipynb`, `react-and-tools.ipynb`, `program-of-thought.ipynb`, `codeact-and-rlm.ipynb`, `flex.ipynb`, `multi-stage-patterns.ipynb`, `parallel-and-majority.ipynb`, `multimodal.ipynb`, `adapters.ipynb`. `contract.txt` and `assets/invoice.md` are short sample documents for `codeact-and-rlm.ipynb` and `multimodal.ipynb` |
| Ch 8 — Building AI Agents | `chapter08/` | 7 notebooks: `react-basics.ipynb`, `framework-comparison.ipynb`, `mcp-integration.ipynb`, `rag-inmemory.ipynb`, `rag-qdrant.ipynb`, `web-search-and-multihop.ipynb`, `history-mem0-rlm.ipynb`; plus the MCP server script `mcp_server.py` |
| Ch 9 — Real-World Use Cases | `chapter09/` | 7 notebooks, one per use case: `sentiment-classifier.ipynb`, `invoice-extraction.ipynb`, `blog-writer.ipynb`, `news-researcher.ipynb`, `video-generator.ipynb`, `customer-service-rag.ipynb`, `financial-analyst.ipynb` |
| Ch 10 — Putting DSPy into Production | `chapter10/` | 3 notebooks: `mlflow-tracking.ipynb`, `fastapi-invoice-api.ipynb`, `dspyui-gradio.ipynb`; see [`chapter10/README.md`](./chapter10/README.md) |
| Ch 11 — Optimizing Coding Agents with DSPy | `chapter11/` | 5 notebooks: `landing-page-skill-optimizer.ipynb`, `image-cli-optimizer.ipynb`, `skill-discovery-rlm.ipynb`, `test-agents-md.ipynb`, `clawsona-dspy.ipynb`. Sample landing pages and a starter skill for the first two notebooks are in `chapter11/` and `chapter11/assets/` |

The datasets for the AI-text detector of Chapters 3 and 6 are in `data/`; [`data/README.md`](./data/README.md) describes each file.

## Setup

These steps follow the Preface of the book ("Software Requirements for This Book"). Run every command from the repository root, where `pyproject.toml`, `uv.lock`, and `requirements.txt` live.

### Prerequisites

- Python 3.12, 3.13, or 3.14
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- An OpenAI API key. Most notebooks call OpenAI models; the keys for other providers are needed only by the examples listed under [Environment variables](#environment-variables).

### Quick start (recommended)

The recommended route uses [uv](https://docs.astral.sh/uv/), a Python project and package manager. If uv is not on your machine yet, see its [installation guide](https://docs.astral.sh/uv/getting-started/installation/), or install it on macOS or Linux with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, run this in PowerShell:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

If PowerShell reports an execution-policy error, use the form given in the uv documentation instead: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.

Next, clone the companion repository and install its locked environment:

```bash
git clone https://github.com/hammer-mt/context-engineering-dspy-book.git
cd context-engineering-dspy-book
uv sync --frozen
```

`uv sync` creates the `.venv` virtual environment for you and installs the exact package versions recorded in `uv.lock`. There is nothing to create or activate by hand.

Copy the environment-variable template:

```bash
cp .env.example .env
```

On Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
```

Open the new `.env` file and fill in the API keys for the examples you plan to run. Then check the installation and your OpenAI key with the setup checker, which reports what it finds without printing any key values:

```bash
uv run python scripts/check_setup.py --require-openai-key
```

Start JupyterLab and open `chapter01/hello-dspy.ipynb`:

```bash
uv run jupyter lab
```

`pyproject.toml` declares support for Python 3.12 through 3.14 and pins DSPy 3.3.0, and `uv.lock` records the complete set of tested package versions. Do not install a different DSPy version on top of this environment: the examples and saved artifacts are validated against DSPy 3.3.0.

With the uv workflow you never need to activate the virtual environment. Put `uv run` in front of a command instead, as in:

```bash
uv run python chapter08/mcp_server.py
```

That script is the small MCP server from Chapter 8, "MCP Integrations". It talks to its client over standard input and output, so it shows no prompt and simply waits for a client; press Ctrl + C to stop it. An `IncompleteFieldDefinitionWarning` from `pydantic_settings` may appear when it starts and can be ignored. You do not have to start the server yourself for the notebook: `chapter08/mcp-integration.ipynb` launches it automatically.

### Installing without uv

If you cannot use uv, create a virtual environment with Python 3.12, 3.13, or 3.14 and install from `requirements.txt`. The [Python `venv` documentation](https://docs.python.org/3/library/venv.html) explains virtual environments in more detail.

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_setup.py --require-openai-key
python -m jupyter lab
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check_setup.py --require-openai-key
python -m jupyter lab
```

`requirements.txt` is a generated, pip-compatible export of `uv.lock`; do not edit it by hand. With an activated environment, leave out the `uv run` prefix from the commands in this README and in the book (for example, `python chapter08/mcp_server.py`).

If installation stops while building a package from source, one of the pinned packages has no prebuilt wheel for your platform and Python version. Try another supported Python version; with uv, for example, run `uv sync --frozen --python 3.12`.

### Environment variables

Keep API keys out of your code. Store them in the `.env` file that you created from `.env.example`. The variables are the ones listed in the Preface, plus the two extra Qdrant settings it mentions:

| Variable | Provider or service | Used by |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | Most notebooks |
| `ANTHROPIC_API_KEY` | Anthropic | Optional. `chapter10/fastapi-invoice-api.ipynb` creates the fallback and prompt-caching models without calling them, so the key is needed only when a request reaches the fallback endpoint or you call the prompt-caching model. `chapter10/dspyui-gradio.ipynb` needs it only if you pick the Anthropic model in the app. `chapter02/dspy-tour.ipynb` creates an Anthropic model object without calling it |
| `GEMINI_API_KEY` | Google Gemini | The model swap at the end of `chapter03/dspy-in-8-steps.ipynb` (skipped when the key is not set); the opt-in audio and video examples in `chapter07/multimodal.ipynb`. Optional in `chapter02/dspy-tour.ipynb` (as for Anthropic) and for the Gemini model choice in `chapter10/dspyui-gradio.ipynb` |
| `OPENROUTER_API_KEY` | OpenRouter | `chapter11/clawsona-dspy.ipynb` (all of its models); `chapter02/dspy-tour.ipynb` (the small model in the "Recursive Language Models" listing and the default small model of Part 2; Part 1 also runs without the key) |
| `FAL_KEY` | FAL.ai | Optional. `chapter09/video-generator.ipynb` runs with local stand-ins for image and video generation; set this key only if you connect those functions to FAL.ai |
| `SERPER_API_KEY` | Serper | Optional. The notebooks search with Tavily; set this key only if you adapt the Chapter 8 web-search tool to Serper |
| `TAVILY_API_KEY` | Tavily | The web-search cells of `chapter08/web-search-and-multihop.ipynb` (its custom multi-hop section runs without the key). Optional in `chapter09/news-researcher.ipynb`, which uses a local search stand-in unless the key is set |
| `QDRANT_API_KEY` | Qdrant Cloud | `chapter08/rag-qdrant.ipynb` with a hosted cluster; not needed for the local Docker server |
| `QDRANT_URL` | Qdrant | `chapter08/rag-qdrant.ipynb` |
| `QDRANT_COLLECTION` | Qdrant | Optional. Collection name for `chapter08/rag-qdrant.ipynb` (default `knowledge_base`) |

You do not need every key. Check the opening cells of a notebook for the providers and services it uses. `.env` is listed in `.gitignore`; never commit it.

A few notebooks also read optional switches, which are explained where they are used: `CHAPTER06_RUN_LIVE` and `RUN_PAID_GEPA` turn on the live Chapter 6 runs (see [Costs](#costs)), and `DSPY_SMALL_MODEL` and `DSPY_STRONG_MODEL` choose other models for Part 2 of `chapter02/dspy-tour.ipynb`.

`python-dotenv` is part of the environment. To load the `.env` file in Python, use:

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

`load_dotenv()` also looks in parent folders, so a notebook started inside a chapter folder finds the `.env` file at the repository root.

You can instead set a variable for the current terminal session only.

On macOS or Linux:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

In Windows Command Prompt:

```bat
set OPENAI_API_KEY=your-api-key-here
```

In Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

### Security note for contributors and forkers

`.gitignore` prevents `.env` from being staged accidentally, but it does **not** stop `git add -f .env`. Before committing or pushing notebook changes:

1. Clear all notebook outputs (`Edit → Clear Outputs of All Cells` in JupyterLab, or `nbstripout`).
2. Confirm no inline string literals look like API keys: `grep -rE "(sk-[a-zA-Z0-9]{20,}|sk-proj-|AIza[0-9A-Za-z_-]{30,})" chapter*/`.
3. Confirm `.env` is not staged: `git status`.

If you ever accidentally commit a key, **rotate it immediately** at the provider dashboard. Git history retains the leak even after deletion.

## Additional software for later chapters

Python and the packages installed above are enough for most examples. A few examples need an extra local tool or service, and you can leave each one until you reach its chapter. The service commands below bind to `127.0.0.1` so that the service is reachable only from your own machine. Do not change `127.0.0.1` to `0.0.0.0` unless you understand the security implications.

### Deno

DSPy's `ProgramOfThought`, `CodeAct`, and `RLM` modules run the Python they generate in a Deno and Pyodide sandbox, and so does `Flex`. These notebooks use the sandbox:

| Chapter | Notebooks that need Deno |
|---|---|
| Ch 2 | `chapter02/dspy-tour.ipynb` (the RLM sections) |
| Ch 7 | `chapter07/program-of-thought.ipynb`, `chapter07/codeact-and-rlm.ipynb`, `chapter07/flex.ipynb`, `chapter07/modules-tour.ipynb` (the opt-in Flex example) |
| Ch 8 | `chapter08/history-mem0-rlm.ipynb` (the RLM section) |
| Ch 9 | `chapter09/financial-analyst.ipynb` |
| Ch 11 | `chapter11/skill-discovery-rlm.ipynb` |

See the [Deno installation guide](https://docs.deno.com/runtime/getting_started/installation/), or install it on macOS or Linux with:

```bash
curl -fsSL https://deno.land/install.sh | sh
```

On Windows PowerShell, use:

```powershell
irm https://deno.land/install.ps1 | iex
```

Open a new terminal and verify the installation:

```bash
deno --version
```

### Docker and Qdrant

Chapter 8 has an optional RAG example backed by Qdrant (`chapter08/rag-qdrant.ipynb`). You can point it at a hosted Qdrant Cloud cluster or run Qdrant on your machine with Docker ([Qdrant local quickstart](https://qdrant.tech/documentation/quick-start/)). Install Docker Desktop (macOS, Windows, or Linux) or Docker Engine (Linux), start Docker, and verify the installation:

```bash
docker --version
```

Start a disposable local Qdrant instance with:

```bash
docker run --rm --name qdrant -p 127.0.0.1:6333:6333 qdrant/qdrant
```

Leave that terminal open while you work; press Ctrl + C to stop the server. The REST API is at `http://localhost:6333` and the dashboard at `http://localhost:6333/dashboard`. Set the following value in your `.env` file when using the local server:

```text
QDRANT_URL=http://127.0.0.1:6333
```

The notebook connects to a collection that already exists and is already populated. Each point's payload must hold its text in a `document` field, and the vectors must match the notebook's configured vectorizer, the FastEmbed model `BAAI/bge-small-en` (384 dimensions); the notebook does not create or fill the collection. It looks for a collection named `knowledge_base` unless you set `QDRANT_COLLECTION`. Because of `--rm`, the local instance keeps nothing after it stops, so re-create its collections the next time you start it. If you have no collection yet, the last section of the notebook, "A small collection to test with", shows code that builds one in this format.

The notebook installs `dspy-qdrant`, `qdrant-client`, and `fastembed` in its own setup cell. If you run the example outside the notebook, install them with:

```bash
uv pip install dspy-qdrant qdrant-client fastembed
```

### Other optional local tools

**MLflow tracking server.** Used for optional tracing in Chapter 3 ("Configuring Our Program") and throughout Chapter 10, "Monitoring and Observability with MLflow" (`chapter10/mlflow-tracking.ipynb`). MLflow is part of the environment. Start the server in a separate terminal:

```bash
uv run mlflow server --backend-store-uri sqlite:///mydb.sqlite
```

Chapter 3 prints the shorter `mlflow server`, and Chapter 10 ("Setting Up MLflow") adds the SQLite backend shown here; either form works with the notebooks. The book and the notebooks show the command without `uv run`, which is the form to use in an activated environment. The server listens on `http://127.0.0.1:5000` and keeps its data in the folder where you start it (`mydb.sqlite`, plus an `mlartifacts/` folder for logged artifacts).

**Redis.** Chapter 10, "Caching in Production", includes an optional cache shared through Redis (`chapter10/fastapi-invoice-api.ipynb`). The notebook shows how to start Redis with Docker:

```bash
docker run --rm -p 127.0.0.1:6379:6379 redis:alpine
```

**Playwright's Chromium browser.** The landing-page optimizer in Chapter 11 (`chapter11/landing-page-skill-optimizer.ipynb`) renders pages in Chromium. Install the browser from the repository environment with:

```bash
uv run playwright install chromium
```

**MCP server.** Chapter 8, "MCP Integrations", uses the server included as `chapter08/mcp_server.py`, which offers two tools, `search_flights` and `book_flight`, that return sample data. Nothing needs to be installed or started: `chapter08/mcp-integration.ipynb` launches it for you.

**spaCy language model.** The privacy guardrails in `chapter09/customer-service-rag.ipynb` use Microsoft Presidio, which is part of the environment, with the spaCy model `en_core_web_lg`, which is not: it is a separate download of about 400 MB. Presidio fetches it the first time the notebook needs it. To install it ahead of time, run:

```bash
uv run python -m spacy download en_core_web_lg
```

In an activated environment, leave out `uv run`. Running `uv sync` again removes the model, and the notebook then downloads it again.

**Local fine-tuning.** The Chapter 6 fine-tuning optimizers (`chapter06/bootstrap-finetune.ipynb` and `chapter06/better-together.ipynb`) train and run `Qwen/Qwen2.5-0.5B-Instruct` on your own machine with PyTorch, Transformers, TRL, and PEFT, all of which are part of the environment. The live path is set up for Apple Silicon (MPS) and falls back to the CPU, which is slower. A live run also calls an `openai/gpt-5.6-sol` teacher, so it needs `OPENAI_API_KEY`, and it downloads the Qwen model from the Hugging Face Hub the first time. The published runs took about 17 minutes (BootstrapFinetune) and 29 minutes (BetterTogether) on Apple Silicon. You can read the published results in those notebooks without training anything.

### Extra packages installed by notebooks

Every notebook outside Chapter 6 begins with a `%pip install` cell that installs `requirements.txt`, for readers who use pip or Google Colab. In the uv environment that cell finds everything already installed. A few notebooks then install additional packages themselves, in the same cell or a later one, which keeps the default environment small:

| Notebook | What its install cell adds |
|---|---|
| `chapter07/multimodal.ipynb` | `attachments soundfile -c ../requirements.txt` |
| `chapter08/framework-comparison.ipynb` | `"langchain>=1" langchain-openai "pydantic-ai>=2" openai-agents crewai` |
| `chapter08/mcp-integration.ipynb` | `"mcp>=1,<2" langchain` (`mcp` is already part of the environment, so the cell adds `langchain`) |
| `chapter08/rag-qdrant.ipynb` | `dspy-qdrant qdrant-client fastembed` |
| `chapter08/history-mem0-rlm.ipynb` | `"mem0ai>=2.1,<3"` |
| `chapter10/fastapi-invoice-api.ipynb` | `redis` |
| `chapter10/mlflow-tracking.ipynb` | `'mlflow[mcp]>=3.5.1'` |
| `chapter10/dspyui-gradio.ipynb` | `gradio -c ../requirements.txt` |

The second column shows the arguments each notebook passes to `%pip install`. Where they include `-c ../requirements.txt`, pip uses the pinned package list as a constraints file, so the new package is installed without changing the versions the other notebooks rely on.

The locked environment includes `pip`, so these cells work when you start Jupyter with `uv run jupyter lab` as well as in an environment made with `python -m venv`.

The frameworks that `chapter08/framework-comparison.ipynb` installs replace the pinned versions of some shared packages; its first cell explains the effect and how to undo it. With uv, `uv sync --frozen` restores the locked environment. Running `uv sync` also removes every package that is not in `uv.lock`, so rerun a notebook's install cell afterward.

If a cell prints `No module named pip`, your environment was created without pip. From a terminal in the repository root, run `uv pip install` followed by the arguments from the table, writing `requirements.txt` in place of `../requirements.txt`, and then restart the notebook kernel. For example:

```bash
uv pip install gradio -c requirements.txt
```

## Running the notebooks

From the repository root, start JupyterLab in the locked environment:

```bash
uv run jupyter lab
```

Open a chapter folder, pick a notebook, and run its cells from top to bottom. The first time a notebook loads a library that draws progress bars, Jupyter may print `TqdmWarning: IProgress not found. Please update jupyter and ipywidgets.` The warning is harmless: progress is shown as plain text. Many notebooks also run in Google Colab once you have cloned the repository there; examples that depend on local services, Docker, browser automation, or files in the repository work best on your own machine.

Standalone scripts are run from the repository root too. A script that imports other modules from its chapter folder is run as a module:

```bash
uv run python -m chapter06.run_live_optimizer --help
```

### Costs

Before you run a notebook, check its opening cells for the keys and services it needs and for any note on cost. The longest and most expensive runs are opt-in:

- **Chapter 6.** The notebooks validate the data and display the published results without calling a model. Set `CHAPTER06_RUN_LIVE=1` before starting Jupyter to compile and evaluate an optimizer yourself. `gepa-expanded-dataset-experiment.ipynb` has its own switch, `RUN_PAID_GEPA=1`.
- **Chapter 11.** `landing-page-skill-optimizer.ipynb` and `image-cli-optimizer.ipynb` default to a budget of three metric calls, which scores the starting point once and confirms that everything is wired up. The larger run is a commented-out cell that you enable yourself: the 200 metric calls printed in the book for the landing-page skill, and 600 for the image CLI, a figure the notebook suggests because the book prints no budget for that run.
- **Flags in other notebooks.** A few notebooks put a long run behind a flag that you set to `True`: `RUN_OPTIMIZATION` in `chapter04/hf-datasets.ipynb` (a MIPROv2 run of about 1,700 model calls), `RUN_FLEX_OPTIMIZATION` in `chapter07/flex.ipynb` and `chapter07/modules-tour.ipynb` (a GEPA compile), and `RUN_LIVE` in `chapter05/rubric-and-multipredictor.ipynb` (one multi-hop question against a retriever). The audio and video model calls in `chapter07/multimodal.ipynb` run only after you supply a recording or a video of your own.

Optimizer cells without such a switch run with the parameters printed in the book. Most of them finish within a few minutes. Three take longer or cost more:

- The GEPA cell in `chapter09/invoice-extraction.ipynb` makes its model calls one after another; allow 15 to 20 minutes.
- `chapter11/clawsona-dspy.ipynb` runs the book's 50-call budget through OpenRouter. The book reports that this run took 19 minutes and cost a few dollars (page 306).
- `chapter11/test-agents-md.ipynb` runs the book's 80-call budget and takes about five minutes the first time.

Approximate LLM spend if you run every notebook in a chapter once, from top to bottom, with its default cells:

| Chapter | Estimated cost | What it covers |
|---|---|---|
| Ch 1 | under $0.10 | One short model call |
| Ch 2 | $0.30–1.50 | A few dozen short calls in each of the two parts |
| Ch 3 | $0.20–1.00 | The GEPA optimizations are the largest part |
| Ch 4 | $0.20–1.00 | With the opt-in MIPROv2 run in `hf-datasets.ipynb` left off; that run adds about $0.60 or more. Dataset downloads are free |
| Ch 5 | $0.10–0.50 | Three of the five notebooks call no chat model. The final GEPA cell in `human-then-llm-judge.ipynb` finishes within seconds when the judge accepts the starter answers; the cost note above that cell explains when it runs longer and costs more |
| Ch 6 | $0 by default | With `CHAPTER06_RUN_LIVE=1`, the book puts most optimizer runs at around $0.60–$1.15 and 5 to 29 minutes each (page 134) |
| Ch 7 | $0.50–2.00 | With the opt-in Flex optimizations left off; the one in `flex.ipynb` adds a few minutes and roughly $0.10–0.50 |
| Ch 8 | $0.30–1.50 | The framework comparison and the agent examples make the most calls. Tavily searches count against your Tavily plan |
| Ch 9 | $0.50–1.50 | The invoice-extraction GEPA run is the largest part |
| Ch 10 | $0.30–1.00 | Launching the DSPyUI app and compiling a program in it costs extra |
| Ch 11 | $2–4, plus the Clawsona run | With the default cells of the four notebooks that use OpenAI. About half of it is `image-cli-optimizer.ipynb`, which generates images for ten validation pages; the book puts that step at $0.05–$0.15 per page (page 296). The Clawsona run is billed by OpenRouter (a few dollars, as above). The larger-budget cells cost far more: each of the 600 metric calls in the image-CLI cell runs the whole CLI, image generation included, on one page |

These figures are rough guides, not quotes. They are based on what one run cost with the models printed in the book, as reported by LiteLLM in the locked environment, and they leave room for provider prices that differ from LiteLLM's price table, for retries, and for longer model answers. Running a notebook a second time usually costs much less, because DSPy answers repeated calls from its cache. Where a notebook gives its own cost note, that note is the more specific guide.

## Notes on differences from the printed listings

The notebooks reproduce the code listings printed in the book verbatim, so you can follow the book line by line. Some printed listings are abbreviated, use placeholder names where real data or functions would go, or depend on a service or a long optimizer run. In those places the notebook shows the listing as printed and adds what it needs to run, with a note that begins **Book vs. notebook** and says what the notebook adds or does differently. The same kind of note marks cells that are not in the book, and places where a run with the pinned package versions behaves differently from what a printed comment or a saved result shows.

[BOOK_NOTES.md](./BOOK_NOTES.md) collects all of these notes, grouped by chapter and notebook, so you can see where a chapter's notebooks differ from the printed pages before you start it. Each note is quoted from its notebook, so "the next cell" or "the cell below" means the cell that follows the note there. Page numbers refer to the print edition.

Where the book prints the output of a listing, the notebook quotes it in a markdown cell headed **Output printed in the book**, placed after the code that produces it, so you can compare it with your own run. The notebooks are saved without cell outputs (the generated Chapter 6 notebooks are the exception), so what you see under a code cell is always from your own run. Model wording, scores, and timings will differ from the printed output.

## Running the tests

The repository includes checks for the setup instructions, the notebooks, and the published results. They run offline and make no model or API calls. Run them from the repository root:

```bash
uv run python -m unittest discover
```

The suites live in `tests/`, `chapter06/tests/`, `chapter09/tests/`, and `chapter10/tests/`. To run one of them, name its folder:

```bash
uv run python -m unittest discover -s chapter06/tests -t .
```

## Contributing

Bug reports, fixes, and improvements are welcome. Before opening a pull request:

- Clear notebook outputs (see the security note above). The twelve generated Chapter 6 notebooks (all except `gepa-expanded-dataset-experiment.ipynb`) are the exception: they are produced by `chapter06/build_optimizer_notebooks.py` and keep the outputs that show the published results. Change the generator instead of editing those notebooks by hand, then run `uv run python -m chapter06.build_optimizer_notebooks` and `uv run python -m chapter06.execute_optimizer_notebooks`.
- Follow the notebook conventions: an opening markdown cell that names the chapter and the printed section title, a setup cell, and a list of the environment variables and services the notebook needs.
- Change dependencies in `pyproject.toml`, run `uv lock`, then regenerate the pip export with `uv export --frozen --no-hashes --no-emit-project --output-file requirements.txt`.
- Keep model identifiers exactly as printed in the book.
- Keep code that reproduces a printed listing exactly as printed. Put additions in a separate cell with a **Book vs. notebook** note, then run `uv run python scripts/build_book_notes.py` to refresh `BOOK_NOTES.md`.
- If you change a class, signature, or function that is defined in more than one notebook, check the other copies listed in `docs/duplication-registry.yaml`. Each copy follows the listing printed in its own chapter, so the copies are not always identical.
- Run the tests.

## License

MIT — see [LICENSE](./LICENSE).
