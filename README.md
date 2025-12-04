# context-engineering-dspy-book
Context Engineering with DSPy (O'Reilly)

## About

Context Engineering with DSPy is a practical guide to building reliable AI systems. It moves beyond "prompt engineering" to **Context Engineering**: the art of providing the right information to LLMs in the right format.

Using the **DSPy** framework, you will learn to build modular, self-improving AI programs that are robust against changes in models and data. DSPy handles the low-level plumbing and prompt optimization, allowing you to focus on designing the logic and flow of your application.

## Setup

All code in this book is written in Python and designed for Jupyter Notebooks or Google Colab.

### Prerequisites

- Python 3.12+
- `uv` package manager (recommended)

### Installation

The project uses `uv` for dependency management.

1. **Install uv** (if not already installed):

   Mac/Linux:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Windows (PowerShell):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone and Setup**:

   ```bash
   # Clone the repository (if you haven't already)
   # git clone https://github.com/oreilly-media/context-engineering-dspy-book
   
   # Create virtual environment
   uv venv .venv
   
   # Activate virtual environment
   # Mac/Linux:
   source .venv/bin/activate
   # Windows:
   # .\.venv\Scripts\activate
   
   # Install dependencies
   uv pip install -r requirements.txt
   ```

   Alternatively, if you want to install DSPy directly:
   ```bash
   uv pip install dspy==3.0.3
   ```

### Environment Variables

You need to set your API keys (e.g., `OPENAI_API_KEY`) to run the examples.

**Using a .env file (Recommended):**

1. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

2. The code will automatically load this using `python-dotenv`.

**Using Terminal:**

Mac/Linux:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```
