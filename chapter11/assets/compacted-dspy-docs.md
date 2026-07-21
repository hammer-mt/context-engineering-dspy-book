# Introduction: What is DSPy?


## Introduction: What is DSPy?

DSPy is a declarative Python framework for building modular, maintainable, and portable AI systems. Unlike traditional prompt engineering, DSPy lets you compose AI software from natural-language modules and structured code, enabling fast iteration and robust integration with a wide range of language models (LMs) and learning strategies.

### Key Concepts

- **Declarative AI Programming:** Write modular, structured code instead of brittle prompt strings.
- **Portability & Reliability:** DSPy programs can be reused and adapted across different LMs and inference strategies.
- **Self-Improvement:** DSPy includes algorithms to automatically optimize prompts and weights for your tasks.

### DSPy Workflow

1. **Programming:** Define your task, constraints, and initial pipeline using DSPy modules.
2. **Evaluation:** Collect a development set, define metrics, and systematically iterate on your system.
3. **Optimization:** Use DSPy optimizers to tune prompts or weights based on your evaluation metrics.

> **Tip:** Start with programming, then evaluation, and only optimize once you have a reliable metric and system.

### Getting Started

Install DSPy:
```bash
pip install -U dspy
```

#### Configure Your Language Model

DSPy supports many LMs. Below are common setup examples:

**OpenAI**
```python
import dspy
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
dspy.configure(lm=lm)
```

**Anthropic**
```python
import dspy
lm = dspy.LM('anthropic/claude-3-opus-20240229', api_key='YOUR_ANTHROPIC_API_KEY')
dspy.configure(lm=lm)
```

**Databricks**
```python
import dspy
lm = dspy.LM('databricks/databricks-meta-llama-3-1-70b-instruct')
dspy.configure(lm=lm)
```

**Local LMs (Ollama)**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama run llama3.2:1b
```
```python
import dspy
lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)
```

**Local LMs (SGLang)**
```bash
pip install "sglang[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Llama-3.1-8B-Instruct
```
```python
lm = dspy.LM("openai/meta-llama/Llama-3.1-8B-Instruct",
             api_base="http://localhost:7501/v1",
             api_key="local", model_type='chat')
dspy.configure(lm=lm)
```

**Other Providers**
DSPy supports dozens of LLM providers via [LiteLLM](https://docs.litellm.ai/docs/providers). Use the appropriate API key and model name:
```python
import dspy
lm = dspy.LM('openai/your-model-name', api_key='PROVIDER_API_KEY', api_base='YOUR_PROVIDER_URL')
dspy.configure(lm=lm)
```

### Direct LM Calls

You can call the configured LM directly:
```python
lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']
```

For more, visit [GitHub](https://github.com/stanfordnlp/dspy) or join the [Discord community](https://discord.gg/XCGy2WDCQB).



================================================================================



# Introduction: Key Features and Objectives


## Introduction: Key Features and Objectives

DSPy is a framework for building modular, programmable language model (LM) systems using core abstractions: **Signatures**, **Modules**, **Optimizers**, and **Assertions**. These enable precise control over LM outputs, supporting both prompt and weight optimization workflows.

### Key Features

- **Modular and Extensible:** DSPy components are highly modular, supporting typed constraints, assertions, and observability. The system is designed for compatibility with external libraries (e.g., LiteLLM) for efficient LM and embedding management, including caching, streaming, and async execution.
- **Programmable Pipelines:** Users define tasks and metrics, compose modules with signatures, and leverage optimizers to automatically generate effective prompts or finetune LM weights.
- **Iterative Optimization:** DSPy encourages iterative evaluation and optimization, allowing continuous improvement of LM systems through prompt/weight refinement and custom metrics.
- **Production-Ready:** Thread-safe and natively supports asynchronous execution, making DSPy suitable for high-throughput and deployment scenarios.
- **Flexible Output and Metrics:** Supports multiple output fields and custom Python metrics for evaluating system performance.

### Objectives

DSPy’s primary goal is to move from ad-hoc prompting to robust, declarative, and modular LM systems. Ongoing and future objectives include:
- Refining abstractions and infrastructure for structured inputs/outputs and LM adapters.
- Enhancing optimizer efficiency and reducing costs.
- Improving fine-tuning, data bootstrapping, and multi-module training.
- Providing comprehensive tutorials from ML workflow to deployment.
- Supporting interactive optimization, tracking, and robust resource management.

DSPy is designed to be user-friendly, extensible, and capable of supporting both rapid prototyping and production-level LM applications.



================================================================================



# Introduction: Installation and Setup


## Introduction: Installation and Setup

To get started with DSPy, ensure you have Python installed, then install the required packages using pip:

```shell
pip install -U dspy mcp
```

This command installs or upgrades DSPy and its dependency `mcp` to the latest versions.



================================================================================



# Introduction: Is DSPy Right for You?


## Introduction: Is DSPy Right for You?

DSPy is designed for NLP and AI researchers or practitioners who need to build complex, modular, and optimizable pipelines for language models. Consider DSPy if your work involves:

- Iterative optimization of prompts or pipelines
- Modular programmatic control over LM interactions
- Adapting to various models or data without relying on fragile prompt strings

**Comparison to Other Frameworks:**

- **Prompt Wrappers (e.g., OpenAI API, MiniChain):**  
  Use string templates for simple tasks. For complex, multi-stage pipelines or tasks requiring retrieval and finetuning, DSPy provides abstraction and optimization beyond what prompt wrappers offer.

- **Application Libraries (e.g., LangChain, LlamaIndex):**  
  These provide pre-built modules and generic prompts for standard applications. DSPy instead offers general-purpose modules that *learn* to prompt or finetune your LM for your specific pipeline and data, giving you a lightweight, customizable programming model.

- **Generation Control Libraries (e.g., Guidance, LMQL, RELM, Outlines):**  
  These focus on constraining LM outputs (e.g., enforcing JSON or regex formats). DSPy optimizes prompts and program structure for your task, including structured outputs if needed. Future versions may integrate constraint mechanisms via Signatures.

Choose DSPy if you want a flexible, automatically-optimizing framework for building and refining custom LM pipelines.



================================================================================



# Introduction: Other


## Introduction: Other

Effective use of DSPy goes beyond mastering its syntax and core concepts. Users should adopt an iterative, exploratory workflow that spans the entire machine learning lifecycle: from data collection, through program design and optimization, to deployment and monitoring. Incremental refinement and continuous evaluation are key to building robust DSPy programs that perform well in production environments.



================================================================================



# Core Concepts: Signatures: Inline Signatures


## Inline Signatures

DSPy Signatures can be defined inline as short strings specifying input and output fields, with optional types. The default type is `str` if unspecified.

**Examples:**
- **Question Answering:** `"question -> answer"` (equivalent to `"question: str -> answer: str"`)
- **Sentiment Classification:** `"sentence -> sentiment: bool"`
- **Summarization:** `"document -> summary"`
- **Multiple Inputs/Outputs:**  
  - `"context: list[str], question: str -> answer: str"`
  - `"question, choices: list[str] -> reasoning: str, selection: int"`

**Tips:**
- Use any valid variable names for fields; keep them semantically meaningful but simple.
- For summarization, `"document -> summary"`, `"text -> gist"`, or `"long_context -> tldr"` are all valid.

### Example: Sentiment Classification

```python
sentence = "it's a charming and often affecting journey."
classify = dspy.Predict('sentence -> sentiment: bool')
result = classify(sentence=sentence)
print(result.sentiment)  # Output: True
```

### Example: Summarization

```python
document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""
summarize = dspy.ChainOfThought('document -> summary')
response = summarize(document=document)
print(response.summary)
# Possible Output:
# The 21-year-old Lee made seven appearances and scored one goal for West Ham last season. He had loan spells in League One with Blackpool and Colchester United, scoring twice for the latter. He has now signed a contract with Barnsley, but the length of the contract has not been revealed.
```

Many DSPy modules (except `dspy.Predict`) may return auxiliary fields by expanding your signature. For example, `dspy.ChainOfThought` adds a `reasoning` field:

```python
print("Reasoning:", response.reasoning)
# Possible Output:
# Reasoning: We need to highlight Lee's performance for West Ham, his loan spells in League One, and his new contract with Barnsley. We also need to mention that his contract length has not been disclosed.
```



================================================================================



# Core Concepts: Signatures: Class-based Signatures


## Core Concepts: Signatures — Class-based Signatures

Class-based signatures in DSPy allow you to define tasks with greater expressiveness and control. They are Python classes that inherit from `dspy.Signature`, and can include:

- **Docstrings**: Clarify the purpose of the signature.
- **Input/Output fields**: Use `dspy.InputField` and `dspy.OutputField`, optionally with a `desc` argument to provide hints or constraints.
- **Type annotations**: Restrict output values (e.g., using `Literal` or `bool`).

### Example: Emotion Classification

```python
from typing import Literal

class Emotion(dspy.Signature):
    """Classify emotion."""
    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"
classify = dspy.Predict(Emotion)
classify(sentence=sentence)
```
_Possible Output:_
```text
Prediction(
    sentiment='fear'
)
```

### Example: Faithfulness to Citations

```python
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""
    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")

context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."
text = "Lee scored 3 goals for Colchester United."

faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
faithfulness(context=context, text=text)
```
_Possible Output:_
```text
Prediction(
    reasoning="Let's check the claims against the context. The text states Lee scored 3 goals for Colchester United, but the context clearly states 'He scored twice for the U's'. This is a direct contradiction.",
    faithfulness=False,
    evidence={'goal_count': ["scored twice for the U's"]}
)
```

### Example: Multi-modal Image Classification

```python
class DogPictureSignature(dspy.Signature):
    """Output the dog breed of the dog in the image."""
    image_1: dspy.Image = dspy.InputField(desc="An image of a dog")
    answer: str = dspy.OutputField(desc="The dog breed of the dog in the image")

image_url = "https://picsum.photos/id/237/200/300"
classify = dspy.Predict(DogPictureSignature)
classify(image_1=dspy.Image.from_url(image_url))
```
_Possible Output:_
```text
Prediction(
    answer='Labrador Retriever'
)
```

**Tip:** Use class-based signatures to clarify tasks and add constraints, but avoid manual keyword tuning—let DSPy optimizers handle it for better performance and transferability.



================================================================================



# Core Concepts: Signatures: Other


## Core Concepts: Signatures: Other

A **Signature** in DSPy is a declarative specification of the input and output behavior for a module. Unlike traditional function signatures that simply describe argument types, DSPy Signatures both declare and initialize the behavior of modules. Field names in Signatures are semantically meaningful—e.g., `question` vs. `answer`, or `sql_query` vs. `python_code`—and define the roles of each input and output in plain English. This approach allows you to specify _what_ the language model should do, rather than _how_ to prompt it, making module behavior explicit and adaptable.



================================================================================



# Core Concepts: Modules: Module Types (Predict, ChainOfThought, ReAct, etc.)


## Core Concepts: Modules: Module Types (Predict, ChainOfThought, ReAct, etc.)

DSPy modules are the primary building blocks for LM-powered programs. Each module abstracts a prompting or reasoning technique (e.g., basic prediction, chain-of-thought, tool use) and is parameterized by a *signature*—a declarative specification of input and output fields. Modules can be composed to form complex systems and are inspired by neural network modules in frameworks like PyTorch.

### Main Module Types

#### `dspy.Predict`
The most basic module. It takes a signature and produces outputs according to the specified schema, without altering the reasoning process.

```python
qa = dspy.Predict('question: str -> response: str')
response = qa(question="What are high memory and low memory on Linux?")
print(response.response)
```

#### `dspy.ChainOfThought`
Encourages the LM to reason step-by-step before producing the final answer. Typically adds a `reasoning` field to the output.

```python
cot = dspy.ChainOfThought('question -> answer')
response = cot(question="Two dice are tossed. What is the probability that the sum equals two?")
print(response.reasoning)
print(response.answer)
```

#### `dspy.ChainOfThoughtWithHint`
Like `ChainOfThought`, but accepts an additional hint to guide reasoning.

```python
generate_answer = dspy.ChainOfThoughtWithHint('question, hint -> answer')
response = generate_answer(
    question="What is the color of the sky?",
    hint="It's what you often see during a sunny day."
)
print(response.answer)
```

#### `dspy.ProgramOfThought`
Prompts the LM to generate code, executes it, and uses the result as the answer.

```python
pot = dspy.ProgramOfThought('question -> answer: float')
result = pot(question="What is 2 + 2?")
print(result.answer)
```

#### `dspy.ReAct`
Implements the ReAct agent pattern, allowing the LM to use external tools/functions as part of its reasoning.

```python
def evaluate_math(expression: str): ...
def search_wikipedia(query: str): ...

react = dspy.ReAct('question -> answer: float', tools=[evaluate_math, search_wikipedia])
pred = react(question="What is 9362158 divided by the year of birth of David Gregory?")
print(pred.answer)
```

#### `dspy.MultiChainComparison`
Runs multiple chains (e.g., multiple `ChainOfThought` completions) and compares their outputs to select the best result.

#### `dspy.Retrieve`
Retrieves relevant passages from a retrieval model to augment LM responses.

```python
retriever = dspy.Retrieve(k=3)
topK_passages = retriever(query="When was the first FIFA World Cup held?").passages
for passage in topK_passages:
    print(passage)
```

#### Function-style Modules
- `dspy.majority`: Returns the most popular response from a set of predictions.

### Usage Pattern

1. **Declare** a module with a signature.
2. **Call** the module with input arguments.
3. **Access** the output fields.

```python
class BasicQA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="short answer")

qa = dspy.ChainOfThought(BasicQA)
response = qa(question="What is the capital of France?")
print(response.reasoning)
print(response.answer)
```

### Tips

- Modules can be configured with arguments like `n` (number of completions), `temperature`, etc.
- Outputs can include reasoning, completions, or structured data, depending on the module and signature.
- Modules are composable: you can build multi-stage pipelines by chaining modules together.

For more advanced usage, see tutorials on multi-stage pipelines, retrieval-augmented generation, and agent-based reasoning in the DSPy documentation.



================================================================================



# Core Concepts: Modules: Composing Modules


## Core Concepts: Modules: Composing Modules

DSPy modules are fully composable using standard Python code. You can freely combine multiple modules—such as chains, signatures, or custom modules—using any control flow (loops, conditionals, etc.). At compile time, DSPy automatically traces all LM calls, enabling optimization and finetuning of the composed system.

Signatures in DSPy are not only for structured I/O; they also allow you to compose multiple signatures into larger modules, which can be compiled into efficient prompts and systems.

**Example: Multi-Hop Search Module**

Below is a simplified example of composing modules in DSPy, inspired by the multi-hop search tutorial:

```python
class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')

    def forward(self, claim: str) -> list[str]:
        notes = []
        titles = []

        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)
        
        return dspy.Prediction(notes=notes, titles=list(set(titles)))
```

**Tips:**
- Compose modules using any Python logic; DSPy will trace and optimize the LM calls.
- Use signatures to define clear interfaces between modules.
- Larger modules built from smaller ones can be compiled and finetuned as a whole.



================================================================================



# Core Concepts: Modules: Module API Overview


## Module API Overview

`dspy.Module` is the base class for all DSPy modules, providing a unified interface for building, executing, and managing DSPy programs. Key methods include:

- `__call__`, `acall`: Synchronous and asynchronous execution of the module.
- `batch`: Process multiple inputs in a batch.
- `deepcopy`, `reset_copy`: Create a deep copy or reset the module's state.
- `save`, `load`: Save or load the module or its state from disk.
- `dump_state`, `load_state`: Serialize or deserialize module parameters and configuration.
- `set_lm`, `get_lm`: Set or retrieve the underlying language model used by the module.
- `parameters`, `named_parameters`: Access module parameters, optionally with names.
- `predictors`, `named_predictors`, `map_named_predictors`: Access or map sub-predictors within the module.
- `named_sub_modules`: List all submodules contained in the module.

These methods enable flexible composition, management, and deployment of DSPy modules in your programs.



================================================================================



# Core Concepts: Modules: Adapters and JSONAdapter


## Core Concepts: Modules: Adapters and JSONAdapter

Adapters in DSPy are utilities designed to standardize and manage the structured input and output between modules and language models. They ensure that data is consistently formatted and parsed, facilitating reliable communication and integration across different components.

### JSONAdapter

The `dspy.JSONAdapter` is a core utility for handling structured data as JSON within DSPy modules. It provides methods to:

- Format prompts as JSON for input to language models.
- Parse outputs from models back into structured Python objects.
- Prepare data for tasks such as few-shot prompting, maintaining conversation history, and finetuning.

By using `JSONAdapter`, you ensure that the input/output between modules and language models remains consistent and machine-readable, reducing errors and simplifying downstream processing.

**Tip:** Use `JSONAdapter` whenever you need reliable, structured communication between DSPy modules and language models, especially for tasks that require precise data formats.



================================================================================



# Core Concepts: Programming in DSPy: Control Flow and Program Structure


## Core Concepts: Programming in DSPy — Control Flow and Program Structure

DSPy encourages constructing control flow and program logic directly in code, moving away from prompt-string engineering. The recommended workflow is:

1. **Define the Task**: Clearly specify the required inputs and outputs.
2. **Build a Simple Pipeline**: Begin with basic modules such as `dspy.ChainOfThought`.
3. **Iterative Development**: Gradually add complexity, testing with diverse examples early to guide future evaluation and optimization.

DSPy's architecture separates:
- **Signatures** (system architecture),
- **Adapters** (input/output formatting),
- **Module Logic** (core computation),
- **Optimization** (performance tuning).

This modularity allows you to swap language models (LMs), adapters, or modules without changing the overall program logic, making DSPy highly portable and maintainable compared to traditional prompt engineering approaches.

**Tip:** Start simple, test early with varied examples, and leverage DSPy's modular design for flexible experimentation.



================================================================================



# Core Concepts: Language Models in DSPy: Configuring LMs


## Core Concepts: Language Models in DSPy — Configuring LMs

To use DSPy, you must first configure a language model (LM). DSPy supports a wide range of providers, including OpenAI, Gemini, Anthropic, Databricks, local LMs (via SGLang or Ollama), and any LiteLLM-compatible provider.

### Basic Configuration

The general pattern for configuring an LM is:

```python
import dspy
lm = dspy.LM('<provider>/<model>', api_key='YOUR_API_KEY', api_base='PROVIDER_URL')
dspy.configure(lm=lm)
```

Set environment variables (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`) as needed for authentication.

#### Provider Examples

- **OpenAI:**
  ```python
  lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
  dspy.configure(lm=lm)
  ```
- **Gemini:**
  ```python
  lm = dspy.LM('gemini/gemini-2.5-pro-preview-03-25', api_key='YOUR_GEMINI_API_KEY')
  dspy.configure(lm=lm)
  ```
- **Anthropic:**
  ```python
  lm = dspy.LM('anthropic/claude-3-opus-20240229', api_key='YOUR_ANTHROPIC_API_KEY')
  dspy.configure(lm=lm)
  ```
- **Databricks:**
  ```python
  lm = dspy.LM('databricks/databricks-meta-llama-3-1-70b-instruct')
  dspy.configure(lm=lm)
  ```
- **Local LMs via SGLang:**
  ```python
  lm = dspy.LM('openai/meta-llama/Meta-Llama-3-8B-Instruct', api_base='http://localhost:7501/v1', api_key='', model_type='chat')
  dspy.configure(lm=lm)
  ```
- **Local LMs via Ollama:**
  ```python
  lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
  dspy.configure(lm=lm)
  ```
- **Other providers via LiteLLM:**
  ```python
  lm = dspy.LM('anyscale/mistralai/Mistral-7B-Instruct-v0.1', api_key='ANYSCALE_API_KEY')
  dspy.configure(lm=lm)
  ```

For full provider support, see the LiteLLM documentation.

### Customizing LM Generation

You can set generation parameters at initialization or per call:

```python
lm = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)
```

- **temperature**: Controls randomness.
- **max_tokens**: Sets response length.
- **stop**: Custom stop sequences.
- **cache**: By default, DSPy caches LM outputs for repeated calls. Set `cache=False` to disable caching.

### Common Errors and Tips

- **"Context too long" errors**: Reduce parameters like `max_bootstrapped_demos`, `max_labeled_demos`, or the number of retrieved passages. Alternatively, increase `max_tokens` if possible.
- **Experiment tracking**: DSPy integrates with MLflow for experiment tracking and tracing (see MLflow section for details).



================================================================================



# Core Concepts: Language Models in DSPy: LM Usage Tracking


## Language Model Usage Tracking

DSPy (v2.6.16+) supports built-in tracking of language model (LM) usage for all module calls.

### Enabling Usage Tracking

Activate tracking by configuring DSPy:

```python
dspy.settings.configure(track_usage=True)
```

You can also specify the LM and enable/disable caching:

```python
dspy.settings.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),
    track_usage=True
)
```

### Accessing Usage Statistics

After running a program, retrieve LM usage from any `dspy.Prediction` object:

```python
usage = prediction_instance.get_lm_usage()
```

The returned dictionary maps LM names to their usage stats, e.g.:

```python
{
    'openai/gpt-4o-mini': {
        'completion_tokens': 61,
        'prompt_tokens': 260,
        'total_tokens': 321,
        'completion_tokens_details': {...},
        'prompt_tokens_details': {...}
    }
}
```

### Example

```python
import dspy

dspy.settings.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),
    track_usage=True
)

class MyProgram(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question -> answer")
        self.predict2 = dspy.ChainOfThought("question, answer -> score")

    def __call__(self, question: str) -> str:
        answer = self.predict1(question=question)
        score = self.predict2(question=question, answer=answer)
        return score

program = MyProgram()
output = program(question="What is the capital of France?")
print(output.get_lm_usage())
```

### Caching and Usage Tracking

If caching is enabled (`cache=True`), repeated calls with the same inputs return cached results and do **not** count toward usage statistics:

```python
dspy.settings.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=True),
    track_usage=True
)

program = MyProgram()
output = program(question="What is the capital of Zambia?")
print(output.get_lm_usage())  # Shows token usage

output = program(question="What is the capital of Zambia?")
print(output.get_lm_usage())  # Shows {}
```

**Note:** Usage tracking is only available in DSPy v2.6.16 and later.



================================================================================



# Core Concepts: Language Models in DSPy: Direct LM Calls


## Core Concepts: Language Models in DSPy — Direct LM Calls

You can call a configured language model (LM) directly in DSPy using a unified API. This approach supports both simple prompts and chat-style messages, and automatically benefits from DSPy's caching utilities.

**Examples:**
```python
lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']

lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']
```

- Use a plain string for standard prompts.
- Use the `messages` argument (a list of role-content dicts) for chat-based models.
- All calls are automatically cached for efficiency.



================================================================================



# Core Concepts: Language Models in DSPy: Multiple LMs and Context Management


## Core Concepts: Language Models in DSPy — Multiple LMs and Context Management

DSPy supports flexible management of multiple language models (LMs) within your workflow. You can:

- **Set a global default LM** using `dspy.configure`.
- **Temporarily override the LM** for a specific code block using `dspy.context`.

Both methods are thread-safe, making them suitable for concurrent or multi-threaded applications.

**Example: Switching Between LMs**
```python
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
response = qa(question="How many floors are in the castle David Gregory inherited?")
print('GPT-4o-mini:', response.answer)

with dspy.context(lm=dspy.LM('openai/gpt-3.5-turbo')):
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print('GPT-3.5-turbo:', response.answer)
```
*Sample Output:*
```
GPT-4o-mini: The number of floors in the castle David Gregory inherited cannot be determined with the information provided.
GPT-3.5-turbo: The castle David Gregory inherited has 7 floors.
```

Use these mechanisms to easily experiment with or deploy multiple LMs in your DSPy applications.



================================================================================



# Core Concepts: Language Models in DSPy: LM Output and Metadata


## Core Concepts: Language Models in DSPy — LM Output and Metadata

Each language model (LM) object in DSPy automatically records a history of all its interactions. This history includes the input prompts, responses, token usage, cost, and other relevant metadata for each call.

To inspect this information:
```python
len(lm.history)           # Number of LM calls made
lm.history[-1].keys()     # Metadata fields for the last call
# Example output:
# dict_keys(['prompt', 'messages', 'kwargs', 'response', 'outputs', 'usage', 'cost'])
```

**Tip:** This metadata is useful for debugging, auditing, and analyzing model usage.

**Advanced:** For custom language models or integrations, subclass `dspy.BaseLM` to implement your own LM or adapter, enabling full compatibility with DSPy signatures and workflows.



================================================================================



# Data Handling in DSPy: Examples: Creating and Manipulating Examples


## Data Handling in DSPy: Creating and Manipulating Examples

DSPy's core data type for representing datapoints is the `Example` object. `Example` objects are similar to Python dictionaries but provide additional utilities for marking input fields and accessing values, making them essential for training, evaluation, and as outputs from DSPy modules (where `Prediction` is a subclass of `Example`).

### Creating Examples

You can create an `Example` directly by specifying fields as keyword arguments:
```python
qa_pair = dspy.Example(question="What is DSPy?", answer="A framework.")
print(qa_pair.question)  # Access a field
```

To mark which fields are inputs (e.g., for prompting), use `.with_inputs()`:
```python
qa_pair = qa_pair.with_inputs("question")
```

Access only the input or label fields:
```python
inputs = qa_pair.inputs()
labels = qa_pair.labels()
```

### Constructing Datasets

Datasets are typically lists of `Example` objects. If you have data as a list of dictionaries (e.g., from a CSV), convert them as follows:
```python
data = [dspy.Example(**d).with_inputs('question') for d in data]
```

#### Example: Loading from CSV with Pandas
```python
import pandas as pd

df = pd.read_csv("sample.csv")
dataset = [
    dspy.Example(context=context, question=question, answer=answer)
    .with_inputs("context", "question")
    for context, question, answer in df.values
]
print(dataset[:3])
```

### Splitting Data

For training and evaluation, split your dataset as needed:
```python
import random
random.shuffle(dataset)
trainset, devset, testset = dataset[:200], dataset[200:500], dataset[500:1000]
```
- **Training/Validation:** Used by optimizers (e.g., 200 examples, split as needed).
- **Development/Test:** Used for system development and final evaluation.

### Tips

- `Example` fields can have any keys and value types (commonly strings).
- Use `.with_inputs("field1", ...)` to specify input fields; others are treated as labels or metadata.
- DSPy modules return `Prediction` objects, which are subclasses of `Example`.



================================================================================



# Data Handling in DSPy: Examples: Specifying Input Keys


## Specifying Input Keys

To define which fields in a DSPy `Example` are treated as inputs, use the `with_inputs()` method. Fields not specified are considered labels or metadata.

```python
# Mark "question" as the input field
qa_pair.with_inputs("question")

# Mark both "question" and "answer" as input fields
qa_pair.with_inputs("question", "answer")
```

**Note:**  
`with_inputs()` returns a new Example instance and does not modify the original object.



================================================================================



# Data Handling in DSPy: Examples: Example Methods and Utilities


## Example Methods and Utilities

The `dspy.Example` class in DSPy is a flexible data container with dictionary-like behavior and several utility methods for handling data fields.

### Key Methods

- `copy()`: Returns a copy of the example.
- `get(key, default=None)`: Retrieves a value by key.
- `inputs()`: Returns a dict of input fields.
- `labels()`: Returns a dict of label fields.
- `items()`, `keys()`, `values()`: Iterate over fields like a dictionary.
- `toDict()`: Converts the example to a standard Python dictionary.
- `with_inputs(*fields)`: Marks specified fields as inputs.
- `without(*fields)`: Returns a copy without specified fields.

### Accessing and Updating Fields

- Access fields with dot notation: `example.field_name`
- Update values by assignment: `example.field_name = new_value`

```python
ex = dspy.Example(article="A", summary="B").with_inputs("article")
print(ex.inputs())   # Example({'article': 'A'})
print(ex.labels())   # Example({'summary': 'B'})
```

Exclude fields with `without()`:

```python
ex = dspy.Example(context="A", question="Q", answer="Ans").with_inputs("context", "question")
print(ex.without("answer"))  # Example({'context': 'A', 'question': 'Q'})
```

### Iterating Over Examples

`Example` supports dictionary-like iteration:

```python
for k, v in article_summary.items():
    print(f"{k} = {v}")
```
**Output**
```
context = This is an article.
question = This is a question?
answer = This is an answer.
rationale = This is a rationale.
```

Use `inputs()` and `labels()` to get only input or label fields as new Example objects.



================================================================================



# Data Handling in DSPy: Datasets: Built-in Datasets


## Built-in Datasets

DSPy provides several built-in datasets, each represented as a list of `Example` objects, making them easy to use or adapt. The main built-in datasets include:

- **HotPotQA**: Multi-hop question answering.
- **GSM8k**: Grade-school math questions.
- **Color**: Basic color dataset.

### Loading and Preparing a Built-in Dataset

To load a built-in dataset, simply import and instantiate it. For example, to load a small subset of HotPotQA:

```python
from dspy.datasets import HotPotQA

dataset = HotPotQA(train_seed=1, train_size=5, eval_seed=2023, dev_size=50, test_size=0)
print(dataset.train)
```

This loads 5 training and 50 development examples as `Example` objects, each containing a `question` and its `answer`. By default, input keys are not set, so you should specify them for use in DSPy modules:

```python
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]
print(trainset)
```

Now, each `Example` has `input_keys={'question'}`.

**Note:** DSPy requires only minimal labeling (typically just the initial input and final output). Intermediate labels are automatically bootstrapped as needed, adapting to changes in your pipeline.



================================================================================



# Data Handling in DSPy: Datasets: Custom Datasets


## Custom Datasets

In DSPy, a custom dataset is simply a list of `Example` objects. There are two main ways to create such datasets:

- **Pythonic Way (Recommended):**  
  Use standard Python logic to construct a list of `Example` objects from your data sources (e.g., CSV, JSON, database). This approach is simple, flexible, and sufficient for most use cases.

- **Advanced: DSPy `Dataset` Class:**  
  For advanced scenarios (e.g., large-scale or streaming data), you can use DSPy's `Dataset` class, which offers additional features and abstractions.

In most cases, start by building your dataset as a Python list of `Example` objects.



================================================================================



# Data Handling in DSPy: Datasets: DSPy Dataset Class Internals


## DSPy Dataset Class Internals

The DSPy `Dataset` base class standardizes data handling by converting raw data into train, dev, and test splits, each as a list of `Example` objects. Accessing `.train`, `.dev`, or `.test` properties shuffles and samples the data (if enabled), then wraps each item as an `Example` with a unique ID and split name.

To create a custom dataset, subclass `Dataset` and implement the `__init__` method to load your data and assign lists of dictionaries (or compatible iterators) to the `_train`, `_dev`, and `_test` attributes. The base class manages further processing and provides a consistent API.

**Example: Subclassing DSPy Dataset**

```python
import pandas as pd
from dspy.datasets.dataset import Dataset

class CSVDataset(Dataset):
    def __init__(self, file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        df = pd.read_csv(file_path)
        self._train = df.iloc[0:700].to_dict(orient='records')
        self._dev = df.iloc[700:].to_dict(orient='records')

dataset = CSVDataset("sample.csv")
print(dataset.train[:3])
```

**Note:** If a split (e.g., `_test`) is not populated, accessing it (e.g., `dataset.test`) will raise an error.

This design allows easy customization of data loading and preprocessing by overriding methods as needed, while avoiding repetitive boilerplate for each new dataset.



================================================================================



# Data Handling in DSPy: Datasets: Loading and Preprocessing Data


## Data Handling in DSPy: Datasets—Loading and Preprocessing Data

DSPy provides a `DataLoader` utility for streamlined dataset loading and preprocessing.

### Initialization

```python
from dspy.datasets import DataLoader
dl = DataLoader()
```

### Loading Datasets

- **From HuggingFace**
  ```python
  dl.from_huggingface(dataset_name, split=..., subset=..., input_keys=...)
  ```
  - Load datasets directly from HuggingFace.
  - Supports split slicing (e.g., `"train[:80%]"`) and subset selection.
  - Returns a dict of splits or a list of `dspy.Example` if a single split is specified.

- **From CSV**
  ```python
  dl.from_csv(filename, fields=..., input_keys=...)
  ```
  - Load data from CSV files.
  - Allows selecting specific columns via `fields`.

### Splitting and Sampling

- **Train/Test Split**
  ```python
  dl.train_test_split(dataset, train_size=...)
  ```
  - Splits a list of `dspy.Example` into training and test sets.

- **Sampling**
  ```python
  dl.sample(dataset, n=...)
  ```
  - Randomly samples `n` examples from a dataset.

Use `input_keys` to specify which fields are used as model inputs.



================================================================================



# Tutorials and Practical Examples: Retrieval-Augmented Generation (RAG): Basic RAG Module


## Tutorials and Practical Examples: Retrieval-Augmented Generation (RAG): Basic RAG Module

This tutorial demonstrates how to build a basic Retrieval-Augmented Generation (RAG) module in DSPy for tech question-answering.

### 1. Install DSPy and Download a Sample Corpus

Install DSPy:
```bash
pip install -U dspy
```

Download a sample tech corpus:
```python
download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl")
```

### 2. Set Up the Retriever

You can use any retriever; here, we use OpenAI embeddings with a local top-K search. For efficient retrieval, install FAISS:
```bash
pip install -U faiss-cpu  # or faiss-gpu if you have a GPU
```
Alternatively, set `brute_force_threshold=30_000` in `dspy.retrievers.Embeddings` to avoid FAISS.

Prepare the corpus and retriever:
```python
import ujson

max_characters = 6000  # truncate long documents
topk_docs_to_retrieve = 5

with open("ragqa_arena_tech_corpus.jsonl") as f:
    corpus = [ujson.loads(line)['text'][:max_characters] for line in f]
    print(f"Loaded {len(corpus)} documents. Will encode them below.")

embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=topk_docs_to_retrieve)
```

### 3. Define the RAG Module

Combine retrieval and generation in a DSPy program:
```python
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)
```

### 4. Run the RAG Module

Ask a question:
```python
rag = RAG()
result = rag(question="what are high memory and low memory on linux?")
print(result)
```

Example output:
```
Prediction(
    reasoning="High Memory and Low Memory in Linux refer to two segments of the kernel's memory space...",
    response="In Linux, High Memory refers to the segment of memory that is not permanently mapped..."
)
```

### 5. Inspect and Evaluate

Inspect the module's reasoning:
```python
dspy.inspect_history()
```

Evaluate performance (RAG typically outperforms a plain CoT module):
```python
evaluate(RAG())
# Output: Average Metric: 166.54 / 300 (55.5%)
```

**Tip:** The RAG module structure allows you to flexibly combine retrieval and generation steps, and optimize the whole pipeline.



================================================================================



# Tutorials and Practical Examples: Retrieval-Augmented Generation (RAG): RAG with Optimization


## RAG with Optimization

To enhance the performance of a RAG module in DSPy, you can optimize its prompts using the `MIPROv2` optimizer. This process systematically refines instructions and incorporates few-shot examples to improve generation quality.

### Example: Optimizing a RAG Module

```python
tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)
optimized_rag = tp.compile(
    RAG(), trainset=trainset,
    max_bootstrapped_demos=2, max_labeled_demos=2,
    requires_permission_to_run=False
)
```

The optimizer updates the prompt, typically adding:
- A clear instruction (e.g., "Using the provided `context` and `question`, analyze step by step to generate a comprehensive response. Explain concepts, highlight distinctions, and address complexities.")
- Two worked-out RAG examples (few-shot demonstrations).

### Evaluation and Inspection

After optimization, evaluate the model on a development set:

```python
evaluate(optimized_rag)
# Output: Average Metric: 183.32 / 300 (61.1%)
```

To compare prompts before and after optimization, use:

```python
dspy.inspect_history(n=2)
```

**Tip:** Always validate the optimized module on held-out data to prevent overfitting.



================================================================================



# Tutorials and Practical Examples: Retrieval-Augmented Generation (RAG): Multi-Hop Retrieval


## Retrieval-Augmented Generation (RAG): Multi-Hop Retrieval

This tutorial demonstrates how to build, optimize, and evaluate a multi-hop retrieval program in DSPy using a Wikipedia corpus and the HoVer dataset.

### Setup

Install dependencies:
```bash
pip install -U dspy bm25s PyStemmer jax[cpu]
```

Download and index Wikipedia abstracts (5M docs) with BM25S for retrieval.

### Data Preparation

Load and index the Wikipedia corpus:
```python
import ujson, bm25s, Stemmer
corpus = [f"{line['title']} | {' '.join(line['text'])}" for line in map(ujson.loads, open("wiki.abstracts.2017.jsonl"))]
stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
retriever = bm25s.BM25(k1=0.9, b=0.4)
retriever.index(corpus_tokens)

def search(query, k):
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    return [corpus[doc] for doc in results[0]]
```

Load the HoVer multi-hop dataset:
```python
from dspy.datasets import DataLoader
import random

hover = DataLoader().from_huggingface(
    dataset_name="hover-nlp/hover", split="train", trust_remote_code=True,
    fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",)
)
hover = [
    dspy.Example(claim=x.claim, titles=list(set(y["key"] for y in x.supporting_facts))).with_inputs("claim")
    for x in hover if x["num_hops"] == 3
]
random.shuffle(hover)
trainset, devset = hover[:200], hover[200:500]
```

### Multi-Hop Retrieval Module

Define a multi-hop retrieval module:
```python
class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')
    def forward(self, claim):
        notes, titles = [], []
        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            pred = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(pred.new_notes)
            titles.extend(pred.titles)
        return dspy.Prediction(notes=notes, titles=list(set(titles)))
```

### Metric and Evaluation

Define a custom recall metric and evaluate:
```python
def top5_recall(example, pred, trace=None):
    gold = example.titles
    recall = sum(x in pred.titles[:5] for x in gold) / len(gold)
    return recall >= 1.0 if trace is not None else recall

evaluate = dspy.Evaluate(devset=devset, metric=top5_recall, num_threads=16)
evaluate(Hop())  # Baseline performance
```

### Optimization

Optimize the multi-hop module using MIPROv2:
```python
gpt4o = dspy.LM('openai/gpt-4o')
tp = dspy.MIPROv2(
    metric=top5_recall, auto="medium", num_threads=16,
    prompt_model=gpt4o, teacher_settings=dict(lm=gpt4o)
)
optimized = tp.compile(Hop(), trainset=trainset, max_bootstrapped_demos=4, max_labeled_demos=4)
evaluate(optimized)  # Improved recall after optimization
```

### Prompt Inspection and Saving

Inspect optimized prompts and save/load the module:
```python
dspy.inspect_history(n=2)  # View optimized prompts for submodules
optimized.save("optimized_hop.json")
loaded = Hop(); loaded.load("optimized_hop.json")
```

### MLflow Integration

Track experiments, metrics, and save/load programs using MLflow for reproducibility. See [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) for details.

---

This workflow illustrates how to compose multi-hop modules, integrate retrieval, define custom metrics, and jointly optimize prompts for complex reasoning tasks in DSPy.



================================================================================



# Tutorials and Practical Examples: ReAct Agents: Building and Optimizing ReAct Agents


## Tutorials and Practical Examples: ReAct Agents – Building and Optimizing ReAct Agents

This section demonstrates how to build, optimize, and evaluate DSPy ReAct agents for retrieval-augmented generation (RAG) tasks, including multi-hop search and answer generation.

### 1. Setup

Install DSPy and (optionally) MLflow for experiment tracking:
```bash
pip install -U dspy
pip install mlflow>=2.20  # optional
```

### 2. Language Model Configuration

Configure a main inference LM and (optionally) a larger teacher LM for optimization:
```python
import dspy
llama3b = dspy.LM('<provider>/Llama-3.2-3B-Instruct', temperature=0.7)
gpt4o = dspy.LM('openai/gpt-4o', temperature=0.7)
dspy.configure(lm=llama3b)
```

### 3. Dataset Preparation

Load and preprocess a multi-hop dataset (e.g., HoVer) as `dspy.Example` objects:
```python
from dspy.datasets import DataLoader
hover = DataLoader().from_huggingface(
    dataset_name="hover-nlp/hover", split="train", trust_remote_code=True,
    fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",)
)
hover = [
    dspy.Example(claim=x.claim, titles=list(set([y["key"] for y in x.supporting_facts]))).with_inputs("claim")
    for x in hover if x["num_hops"] == 3
]
import random; random.shuffle(hover)
trainset, devset, testset = hover[:100], hover[100:200], hover[650:]
```

### 4. Tool Definition

Define retrieval and lookup tools using a ColBERTv2 server:
```python
DOCS = {}
def search(query: str, k: int) -> list[str]:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=k)
    results = [x['text'] for x in results]
    for result in results:
        title, text = result.split(" | ", 1)
        DOCS[title] = text
    return results

def search_wikipedia(query: str) -> list[str]:
    topK = search(query, 30)
    titles = [f"`{x.split(' | ')[0]}`" for x in topK[5:30]]
    return topK[:5] + [f"Other retrieved pages have titles: {', '.join(titles)}."]

def lookup_wikipedia(title: str) -> str:
    if title in DOCS: return DOCS[title]
    results = [x for x in search(title, 10) if x.startswith(title + " | ")]
    return results[0] if results else f"No Wikipedia page found for title: {title}"
```

### 5. Agent Definition

Create a ReAct agent for multi-hop title retrieval:
```python
signature = dspy.Signature("claim -> titles: list[str]", "Find all Wikipedia titles relevant to verifying (or refuting) the claim.")
react = dspy.ReAct(signature, tools=[search_wikipedia, lookup_wikipedia], max_iters=20)
```

#### Minimal Example

For simple QA with retrieval:
```python
import dspy
lm = dspy.LM('openai/gpt-4o-mini')
colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(lm=lm, rm=colbert)
agent = dspy.ReAct("question -> answer", tools=[dspy.Retrieve(k=1)])
prediction = agent(question="Which baseball team does Shohei Ohtani play for?")
print(prediction.answer)
# Output: Shohei Ohtani plays for the Los Angeles Angels.
```
*Note: Update your retrieval source for current answers.*

### 6. Metric and Evaluation

Define a custom recall metric and evaluate:
```python
def top5_recall(example, pred, trace=None):
    gold_titles = example.titles
    recall = sum(x in pred.titles[:5] for x in gold_titles) / len(gold_titles)
    return recall >= 1.0 if trace is not None else recall

evaluate = dspy.Evaluate(devset=devset, metric=top5_recall, num_threads=16)
evaluate(react)
```

### 7. Optimization

Optimize prompts using MIPROv2 and a teacher LM:
```python
tp = dspy.MIPROv2(
    metric=top5_recall, auto="medium", num_threads=16,
    teacher_settings=dict(lm=gpt4o), prompt_model=gpt4o, max_errors=999
)
optimized_react = tp.compile(react, trainset=trainset, max_bootstrapped_demos=3, max_labeled_demos=0)
evaluate(optimized_react)
```

### 8. Prompt Inspection

Inspect optimized prompts and agent reasoning:
```python
optimized_react(claim="...").titles
dspy.inspect_history(n=2)
```

### 9. Saving and Loading Agents

Save and reload optimized agents:
```python
optimized_react.save("optimized_react.json")
loaded_react = dspy.ReAct("claim -> titles: list[str]", tools=[search_wikipedia, lookup_wikipedia], max_iters=20)
loaded_react.load("optimized_react.json")
```
Or use MLflow for experiment tracking:
```python
import mlflow
with mlflow.start_run(run_name="optimized_rag"):
    model_info = mlflow.dspy.log_model(optimized_react, artifact_path="model")
loaded = mlflow.dspy.load_model(model_info.model_uri)
```

---

**Summary:**  
This workflow demonstrates building a DSPy ReAct agent with external tools, optimizing prompts with a teacher LM, evaluating with custom metrics, and saving for reuse. For advanced tracing and reproducibility, integrate with MLflow.



================================================================================



# Tutorials and Practical Examples: ReAct Agents: Advanced Tool Use


## Tutorials and Practical Examples: ReAct Agents – Advanced Tool Use

This tutorial demonstrates how to build and prompt-optimize a DSPy agent for advanced tool use, using the challenging [ToolHop](https://arxiv.org/abs/2501.02506) dataset and strict evaluation criteria.

### Installation

Install DSPy and required dependencies:
```bash
pip install -U dspy func_timeout
```

**Optional:** For tracing and experiment tracking, install and set up [MLflow](https://mlflow.org/):
```bash
pip install mlflow>=2.20
mlflow ui --port 5000  # Run in a separate terminal
```
In your notebook:
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog()
```
See [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) for more.

### Model and Data Preparation

Configure your LLM and download the ToolHop dataset:
```python
import dspy
import ujson
import random

gpt4o = dspy.LM("openai/gpt-4o", temperature=0.7)
dspy.configure(lm=gpt4o)

from dspy.utils import download
download("https://huggingface.co/datasets/bytedance-research/ToolHop/resolve/main/data/ToolHop.json")
data = ujson.load(open("ToolHop.json"))
random.Random(0).shuffle(data)
```

Prepare examples and function sets:
```python
import re
import inspect

examples = []
fns2code = {}

def finish(answer: str):
    """Conclude the trajectory and return the final answer."""
    return answer

for datapoint in data:
    func_dict = {}
    for func_code in datapoint["functions"]:
        cleaned_code = func_code.rsplit("\n\n# Example usage", 1)[0]
        fn_name = re.search(r"^\s*def\s+([a-zA-Z0-9_]+)\s*\(", cleaned_code)
        fn_name = fn_name.group(1) if fn_name else None
        if not fn_name:
            continue
        local_vars = {}
        exec(cleaned_code, {}, local_vars)
        fn_obj = local_vars.get(fn_name)
        if callable(fn_obj):
            func_dict[fn_name] = fn_obj
            assert fn_obj not in fns2code, f"Duplicate function found: {fn_name}"
            fns2code[fn_obj] = (fn_name, cleaned_code)
    func_dict["finish"] = finish
    example = dspy.Example(question=datapoint["question"], answer=datapoint["answer"], functions=func_dict)
    examples.append(example.with_inputs("question", "functions"))

trainset, devset, testset = examples[:100], examples[100:400], examples[400:]
```

### Evaluation Setup

Define strict metric and evaluation helpers:
```python
from func_timeout import func_set_timeout

def wrap_function_with_timeout(fn):
    @func_set_timeout(10)
    def wrapper(*args, **kwargs):
        try:
            return {"return_value": fn(*args, **kwargs), "errors": None}
        except Exception as e:
            return {"return_value": None, "errors": str(e)}
    return wrapper

def fn_metadata(func):
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or "No docstring."
    return dict(function_name=func.__name__, arguments=str(signature), docstring=docstring)

def metric(example, pred, trace=None):
    gold = str(example.answer).rstrip(".0").replace(",", "").lower()
    pred = str(pred.answer).rstrip(".0").replace(",", "").lower()
    return pred == gold  # strict match

evaluate = dspy.Evaluate(
    devset=devset, metric=metric, num_threads=24, 
    display_progress=True, display_table=0, max_errors=999
)
```

### ReAct Agent Definition

Implement a ReAct-based agent with a step limit and function call timeout:
```python
class Agent(dspy.Module):
    def __init__(self, max_steps=5):
        self.max_steps = max_steps
        instructions = (
            "For the final answer, produce short (not full sentence) answers in which you format dates as YYYY-MM-DD, "
            "names as Firstname Lastname, and numbers without leading 0s."
        )
        signature = dspy.Signature(
            'question, trajectory, functions -> next_selected_fn, args: dict[str, Any]', instructions
        )
        self.react = dspy.ChainOfThought(signature)

    def forward(self, question, functions):
        tools = {fn_name: fn_metadata(fn) for fn_name, fn in functions.items()}
        trajectory = []
        for _ in range(self.max_steps):
            pred = self.react(question=question, trajectory=trajectory, functions=tools)
            selected_fn = pred.next_selected_fn.strip('"').strip("'")
            fn_output = wrap_function_with_timeout(functions[selected_fn])(**pred.args)
            trajectory.append(dict(reasoning=pred.reasoning, selected_fn=selected_fn, args=pred.args, **fn_output))
            if selected_fn == "finish":
                break
        return dspy.Prediction(answer=fn_output.get("return_value", ''), trajectory=trajectory)
```

### Baseline Evaluation

Evaluate the agent:
```python
agent = Agent()
evaluate(agent)
# Expected: ~35% accuracy
```

### Prompt Optimization with SIMBA

Optimize the agent using the experimental `dspy.SIMBA` optimizer:
```python
simba = dspy.SIMBA(metric=metric, max_steps=12, max_demos=10)
optimized_agent = simba.compile(agent, trainset=trainset, seed=6793115)
```

Re-evaluate:
```python
evaluate(optimized_agent)
# Expected: ~60% accuracy (substantial improvement)
```

**Summary:**  
This workflow demonstrates how to set up, evaluate, and optimize a DSPy ReAct agent for advanced tool use, leveraging strict evaluation and prompt optimization for significant performance gains.



================================================================================



# Tutorials and Practical Examples: ReAct Agents: MCP Tool Integration


## ReAct Agents: Integrating MCP Tools in DSPy

DSPy supports integrating external tools via the Model Context Protocol (MCP), enabling agents to leverage community-built or custom tools without code duplication. This tutorial demonstrates building a DSPy ReAct agent for airline customer service using MCP tools.

### 1. Set Up the MCP Server

- Use FastMCP to create the server.
- Define data schemas (e.g., `Flight`, `UserProfile`) with Pydantic.
- Implement in-memory databases (Python dicts) for demo data.
- Register tools with `@mcp.tool()` decorators.
- Launch the server with `mcp.run()`.

**Example skeleton:**
```python
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("Airline Agent")

class Flight(BaseModel): ...
user_database = {...}
flight_database = {...}

@mcp.tool()
def fetch_flight_info(...): ...

if __name__ == "__main__":
    mcp.run()
```
This enables DSPy agents to discover and call these tools via MCP.

### 2. Connect DSPy to the MCP Server

- Use `mcp.ClientSession` and `stdio_client` to connect and list tools.
- Convert MCP tools to DSPy tools.

**Example:**
```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import dspy

server_params = StdioServerParameters(
    command="python",
    args=["path_to_your_working_directory/mcp_server.py"],
)

async def get_tools():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [dspy.Tool.from_mcp_tool(session, tool) for tool in tools.tools]
```

### 3. Define the Agent and Run

- Define a DSPy signature for the agent's input/output.
- Configure your language model.
- Build and run the agent with the discovered tools.

**Example:**
```python
class DSPyAirlineCustomerService(dspy.Signature):
    """Airline customer service agent using tools to fulfill user requests."""
    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(desc="Summary of the process and key info (e.g., confirmation number).")

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

async def run(user_request):
    dspy_tools = await get_tools()
    react = dspy.ReAct(DSPyAirlineCustomerService, tools=dspy_tools)
    result = await react.acall(user_request=user_request)
    print(result)

import asyncio
asyncio.run(run("please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam"))
```

The agent will reason about the request, select and call the appropriate tools, and return a structured result including a step-by-step trajectory and a summary.

**Note:** This tutorial must be run locally (not in hosted notebooks). For further inspection of agent reasoning and tool use, see DSPy's Observability Guide and MLflow integration.



================================================================================



# Tutorials and Practical Examples: Entity Extraction: People Extraction with CoNLL-2003


## Entity Extraction: People Extraction with CoNLL-2003

This tutorial demonstrates how to extract "person" entities from the CoNLL-2003 dataset using DSPy.

### Installation

```bash
pip install -U dspy-ai
pip install datasets
```

**Optional: MLflow for Experiment Tracking**
```bash
pip install mlflow>=2.20
mlflow ui --port 5000
```
In your notebook:
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog()
```

### 1. Load and Prepare the CoNLL-2003 Dataset

Load the dataset and extract person entities using NER tags. Prepare examples for DSPy:

```python
from datasets import load_dataset
import dspy

dataset = load_dataset("conll2003")
def extract_people(row):
    return [tok for tok, tag in zip(row["tokens"], row["ner_tags"]) if tag in (1, 2)]

examples = [
    dspy.Example(tokens=row["tokens"], expected_extracted_people=extract_people(row)).with_inputs("tokens")
    for row in dataset["train"].select(range(0, 50))
]
```

### 2. Define the DSPy Entity Extraction Program

Create a signature specifying input/output fields and a program using `ChainOfThought`:

```python
from typing import List

class PeopleExtraction(dspy.Signature):
    """
    Extract contiguous tokens referring to specific people, if any, from a list of string tokens.
    Output a list of tokens. Do not combine multiple tokens into a single value.
    """
    tokens: list[str] = dspy.InputField(desc="tokenized text")
    extracted_people: list[str] = dspy.OutputField(desc="all tokens referring to specific people extracted from the tokenized text")

people_extractor = dspy.ChainOfThought(PeopleExtraction)
```

### 3. Configure DSPy and Language Model

Set the language model (e.g., OpenAI's `gpt-4o-mini`):

```python
lm = dspy.LM(model="openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
```
DSPy will use your `OPENAI_API_KEY` for authentication. You can swap in other providers or local models as needed.

### 4. Inspect and Evaluate the Program

After optimization, inspect how DSPy structures prompts and integrates few-shot examples to guide extraction:

```python
dspy.inspect_history(n=1)
```

This will show the prompt structure, including input fields, rationale, and extracted people, as well as the model's reasoning for each example.

**Example Output:**
```python



================================================================================



# Tutorials and Practical Examples: Entity Extraction: Optimization Workflow


## Entity Extraction: Optimization Workflow

To optimize entity extraction systems in DSPy, follow this workflow:

### 1. Optimize the Model

Use DSPy's `MIPROv2` optimizer to automatically tune your program's prompt and augment few-shot examples with reasoning (via `dspy.ChainOfThought`). This process maximizes correctness on your training set and is fully automated.

```python
mipro_optimizer = dspy.MIPROv2(
    metric=extraction_correctness_metric,
    auto="medium",
)
optimized_people_extractor = mipro_optimizer.compile(
    people_extractor,
    trainset=train_set,
    max_bootstrapped_demos=4,
    requires_permission_to_run=False,
    minibatch=False
)
```

### 2. Evaluate the Optimized Program

After optimization, evaluate the program on a test set to assess improvements and generalization:

```python
evaluate_correctness(optimized_people_extractor, devset=test_set)
# Output: Average Metric: 186 / 200 (93.0%)
```

This demonstrates improved accuracy after optimization.



================================================================================



# Tutorials and Practical Examples: Entity Extraction: Other


## Tutorials and Practical Examples: Entity Extraction—Other

DSPy supports modular and interpretable approaches to entity extraction, allowing you to extract a wide variety of entity types beyond standard examples. You can leverage structured input/output and rapid optimization features without manual prompt engineering.

For more complex entity extraction tasks, consider using advanced modules such as `ReAct`, which enable reasoning and decision-making during extraction. DSPy modules can be easily integrated into larger workflows, making it simple to build scalable and maintainable extraction pipelines.

**Tip:** Experiment with different entity types and combine DSPy modules to suit your specific extraction needs.



================================================================================



# Tutorials and Practical Examples: Math Reasoning


## Tutorials and Practical Examples: Math Reasoning

This section demonstrates how to set up, optimize, and evaluate a math reasoning module in DSPy.

### Workflow

1. **Configure the Language Model:**
   ```python
   dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
   ```

2. **Load the MATH Benchmark:**
   ```python
   from dspy.datasets import MATH
   dataset = MATH(subset='algebra')
   ```

3. **Define a Reasoning Module:**
   ```python
   module = dspy.ChainOfThought("question -> answer")
   ```

4. **Evaluate Zero-Shot Performance:**
   ```python
   evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, num_threads=24)
   evaluate(module)
   ```

5. **(Optional) Integrate with MLflow for Experiment Tracking:**
   ```python
   import mlflow
   mlflow.set_tracking_uri("http://localhost:5000")
   mlflow.set_experiment("DSPy")
   mlflow.dspy.autolog()
   ```

6. **Optimize with MIPROv2 (using a teacher LM for bootstrapped demos):**
   ```python
   optimizer = dspy.MIPROv2(
       metric=dataset.metric,
       auto="medium",
       teacher_settings=dict(lm=gpt4o),
       prompt_model=gpt4o_mini,
       num_threads=24
   )
   optimized_module = optimizer.compile(
       module,
       trainset=dataset.train,
       requires_permission_to_run=False,
       max_bootstrapped_demos=4,
       max_labeled_demos=4
   )
   ```

7. **Re-evaluate the Optimized Module** to observe improved accuracy.

### Best Practices

- Use `dspy.inspect_history()` to review prompt changes after optimization.
- For advanced math reasoning, consider `dspy.ReAct` with a calculator tool or ensemble multiple optimized modules.

### Summary

This workflow covers configuring your LM, loading benchmarks, defining and optimizing a math reasoning module, and tracking experiments with MLflow in DSPy.



================================================================================



# Tutorials and Practical Examples: Classification and Finetuning


## Tutorials and Practical Examples: Classification and Finetuning

DSPy enables practical workflows for classification and language model finetuning, supporting both agent-based and standard classification tasks.

### Bootstrapped Finetuning for Classification

**Requirements:**  
- DSPy >= 2.6.0, local GPU, SGLang for local inference.  
- Install:
  ```bash
  pip install -U dspy>=2.6.0
  pip install "sglang[all]>=0.4.4.post3" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
  pip install -U torch transformers==4.48.3 accelerate trl peft
  ```

**Data Preparation:**  
- Load and convert a dataset (e.g., Banking77) to `dspy.Example` objects:
  ```python
  from dspy.datasets import DataLoader
  from datasets import load_dataset
  CLASSES = load_dataset("PolyAI/banking77", split="train").features['label'].names
  raw_data = [
      dspy.Example(x, label=CLASSES[x.label]).with_inputs("text")
      for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", fields=("text", "label"), input_keys=("text",), split="train")[:1000]
  ]
  unlabeled_trainset = [dspy.Example(text=x.text).with_inputs("text") for x in raw_data[:500]]
  ```

**Define Classification Program:**
  ```python
  from typing import Literal
  classify = dspy.ChainOfThought(f"text -> label: Literal{CLASSES}")
  ```

**Set Up Teacher/Student LMs:**
  ```python
  from dspy.clients.lm_local import LocalProvider
  student_lm = dspy.LM(model="openai/local:meta-llama/Llama-3.2-1B-Instruct", provider=LocalProvider(), max_tokens=2000)
  teacher_lm = dspy.LM('openai/gpt-4o-mini', max_tokens=3000)
  student_classify = classify.deepcopy(); student_classify.set_lm(student_lm)
  teacher_classify = classify.deepcopy(); teacher_classify.set_lm(teacher_lm)
  ```

**Bootstrapped Finetuning:**
- **Unlabeled data:**
  ```python
  dspy.settings.experimental = True
  optimizer = dspy.BootstrapFinetune(num_threads=16)
  classify_ft = optimizer.compile(student_classify, teacher=teacher_classify, trainset=unlabeled_trainset)
  classify_ft.get_lm().launch()
  ```
- **Labeled data with metric:**
  ```python
  metric = lambda x, y, trace=None: x.label == y.label
  optimizer = dspy.BootstrapFinetune(num_threads=16, metric=metric)
  classify_ft = optimizer.compile(student_classify, teacher=teacher_classify, trainset=raw_data[:500])
  classify_ft.get_lm().launch()
  ```

**Evaluation:**
  ```python
  devset = raw_data[500:600]
  evaluate = dspy.Evaluate(devset=devset, metric=metric, display_progress=True, num_threads=16)
  evaluate(classify_ft)
  ```

**Experiment Tracking and Saving:**
- Integrate with MLflow for reproducibility:
  ```python
  import mlflow
  with mlflow.start_run(run_name="optimized_classifier"):
      model_info = mlflow.dspy.log_model(classify_ft, artifact_path="model")
  loaded = mlflow.dspy.load_model(model_info.model_uri)
  ```

**Tips:**
- Set `DSPY_FINETUNEDIR` and `CUDA_VISIBLE_DEVICES` as needed.
- Free GPU memory after use: `classify_ft.get_lm().kill()`.
- Bootstrapped finetuning supports both labeled and unlabeled data, and teacher-student distillation.

### Fine-tuning Agents

You can fine-tune agent modules (e.g., ReAct agents) by defining them as `dspy.Module` classes with submodules:
```python
class Agent(dspy.Module):
    def __init__(self, max_iters=50):
        self.max_iters = max_iters
        self.react = dspy.Predict("task, trajectory, possible_actions: list[str] -> action")
    def forward(self, idx):
        # Initialize environment, loop over steps, call self.react, collect trajectory
        ...
```
**Workflow:**
1. Load training/dev sets.
2. Evaluate zero-shot performance with `dspy.Evaluate`.
3. Optimize prompts via `dspy.MIPROv2`.
4. Fine-tune LM weights using `dspy.BootstrapFinetune` (teacher-student).
5. Save/load programs with `.save()` and `.load()`.

**MLflow** is supported for experiment tracking.

**Further Reading:**  
- See [Drew Breunig's tutorial](https://www.dbreunig.com/2024/12/12/pipelines-prompt-optimization-with-dspy.html) for a practical classification example.
- Refer to the DSPy documentation for advanced agent optimization workflows.



================================================================================



# Tutorials and Practical Examples: Multi-Hop and Research Pipelines


## Tutorials and Practical Examples: Multi-Hop and Research Pipelines

DSPy enables the construction of complex, multi-hop research pipelines by composing modules with custom instructions and iterative logic. Below is a practical example demonstrating how to build a multi-hop research program for claim verification.

### Example: Multi-hop Research Program

Define custom instructions for each reasoning step:
```python
instr1 = "Given a claim and key facts, generate a follow-up search query to find the next essential clue for verifying/refuting the claim."
instr2 = "Given a claim, key facts, and new search results, identify new learnings to extend the key facts about the claim's truth."
```

Compose these into a multi-hop module:
```python
class ResearchHop(dspy.Module):
    def __init__(self, num_docs, num_hops):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought(
            dspy.Signature("claim, key_facts -> followup_search_query", instr1)
        )
        self.append_notes = dspy.ChainOfThought(
            dspy.Signature("claim, key_facts, new_search_results -> new_key_facts", instr2)
        )

    def forward(self, claim: str) -> list[str]:
        key_facts, retrieved_docs = [], []
        for hop_idx in range(self.num_hops):
            query = self.generate_query(claim=claim, key_facts=key_facts).followup_search_query if hop_idx else claim
            search_results = search(query, k=self.num_docs)
            retrieved_docs.extend(search_results)
            if hop_idx == self.num_hops - 1:
                break
            prediction = self.append_notes(
                claim=claim, key_facts=key_facts, new_search_results=search_results
            )
            key_facts.append(prediction.new_key_facts)
        return dspy.Prediction(key_facts=key_facts, retrieved_docs=retrieved_docs)
```

### Defining a Recall Metric for Evaluation

To evaluate pipeline effectiveness, define a recall metric:
```python
def recall(example, pred, trace=None):
    gold_titles = example.titles
    retrieved_titles = [doc.split(" | ")[0] for doc in pred.retrieved_docs]
    return sum(x in retrieved_titles for x in set(gold_titles)) / len(gold_titles)

evaluate = dspy.Evaluate(devset=devset, metric=recall, num_threads=16)
```

This setup allows for iterative reasoning and evidence accumulation, with evaluation tailored to the retrieval task.



================================================================================



# Tutorials and Practical Examples: Image Generation and Multimodal


## Tutorials and Practical Examples: Image Generation and Multimodal

### Iterative Image Prompt Refinement with DSPy

This example shows how to use DSPy for multimodal prompt refinement—automatically improving an image generation prompt until the output matches a target description.

**Setup:**
```bash
pip install -U dspy fal-client pillow dotenv
```
Obtain a Flux Pro API key from [FAL](https://fal.com/flux-pro).

**Environment and Model Configuration:**
```python
import dspy, fal_client, requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv; load_dotenv()

lm = dspy.LM(model="gpt-4o-mini", temperature=0.5)
dspy.settings.configure(lm=lm)
```

**Image Generation Utility:**
```python
def generate_image(prompt):
    request_id = fal_client.submit(
        "fal-ai/flux-pro/v1.1-ultra",
        arguments={"prompt": prompt}
    ).request_id
    result = fal_client.result("fal-ai/flux-pro/v1.1-ultra", request_id)
    url = result["images"][0]["url"]
    return dspy.Image.from_url(url)
```

**Iterative Prompt Refinement Loop:**
```python
check_and_revise_prompt = dspy.Predict(
    "desired_prompt: str, current_image: dspy.Image, current_prompt:str -> feedback:str, image_strictly_matches_desired_prompt: bool, revised_prompt: str"
)

initial_prompt = "A scene that's both peaceful and tense"
current_prompt = initial_prompt

for _ in range(5):
    current_image = generate_image(current_prompt)
    result = check_and_revise_prompt(
        desired_prompt=initial_prompt,
        current_image=current_image,
        current_prompt=current_prompt
    )
    if result.image_strictly_matches_desired_prompt:
        break
    current_prompt = result.revised_prompt
```

**Summary:**  
This workflow demonstrates how DSPy can be used to iteratively critique and revise prompts for image generation, integrating language models and external APIs in a multimodal loop. The same pattern can be generalized to other tasks that require iterative, structured refinement using DSPy.



================================================================================



# Tutorials and Practical Examples: Async DSPy Programming: When to Use Sync vs Async


## Async DSPy Programming: When to Use Sync vs Async

Choosing between synchronous and asynchronous programming in DSPy depends on your application's requirements:

**Use Synchronous Programming When:**
- Prototyping, exploring, or conducting research
- Building small to medium-sized applications
- Simpler code and easier debugging are priorities

**Use Asynchronous Programming When:**
- Building high-throughput or production services (high QPS)
- Efficiently handling multiple concurrent requests is needed
- Working with tools that require async operations
- Scalability and resource utilization are critical

**Benefits of Async in DSPy:**
- Improved performance via concurrent operations
- Better resource utilization and reduced I/O wait times
- Enhanced scalability for multiple requests

**Considerations:**
- Async code is more complex, with harder error handling and debugging
- Potential for subtle bugs
- Async code may differ between notebook environments (e.g., Jupyter) and standard Python scripts

**Recommendation:**  
Start with synchronous programming for most development and switch to async only when you have a clear need for its advantages. This approach lets you focus on core logic before introducing async complexity.



================================================================================



# Tutorials and Practical Examples: Async DSPy Programming: Async Module and Tool Examples


## Async DSPy Programming: Async Module and Tool Examples

Most DSPy built-in modules support asynchronous execution via the `.acall()` method, which mirrors the synchronous `.call()` interface but allows for non-blocking operation. This is especially useful for I/O-bound tasks or integrating with async workflows.

### Asynchronous Built-in Module Example

```python
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
predict = dspy.Predict("question->answer")

async def main():
    output = await predict.acall(question="why did a chicken cross the kitchen?")
    print(output)

asyncio.run(main())
```

### Async Tools with `dspy.Tool`

You can wrap async functions as DSPy tools. When using `acall()`, the tool executes asynchronously:

```python
import asyncio
import dspy
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"

async def foo(x):
    await asyncio.sleep(0.1)
    print(f"I get: {x}")

tool = dspy.Tool(foo)

async def main():
    await tool.acall(x=2)

asyncio.run(main())
```

**Tip:** When using `dspy.ReAct` with tools, calling `acall()` on the ReAct instance will automatically execute all tools asynchronously using their own `acall()` methods.



================================================================================



# Tutorials and Practical Examples: Async DSPy Programming: Custom Async Modules


## Async DSPy Programming: Custom Async Modules

To implement custom asynchronous DSPy modules, define an `aforward()` method (instead of `forward()`) containing your async logic. Use `await` to call other async DSPy modules or methods.

**Example: Chaining Async Operations in a Custom Module**

```python
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class MyModule(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question->answer")
        self.predict2 = dspy.ChainOfThought("answer->simplified_answer")

    async def aforward(self, question, **kwargs):
        # Asynchronously chain predictions
        answer = await self.predict1.acall(question=question)
        return await self.predict2.acall(answer=answer)

async def main():
    mod = MyModule()
    result = await mod.acall(question="Why did a chicken cross the kitchen?")
    print(result)

asyncio.run(main())
```

**Tips:**
- Use `acall()` to invoke modules asynchronously.
- All async DSPy modules must implement `aforward()`.
- You can chain or parallelize async operations as needed within `aforward()`.



================================================================================



# Tutorials and Practical Examples: Async DSPy Programming: Other


## Async DSPy Programming: Other

DSPy natively supports asynchronous programming, enabling the development of efficient and scalable applications. You can leverage async features in both built-in DSPy modules and your own custom implementations to improve performance, especially in I/O-bound or concurrent workloads.



================================================================================



# Tutorials and Practical Examples: Streaming: Output Token Streaming


## Output Token Streaming

DSPy supports token streaming for any module output field of type `str`. This enables real-time consumption of model outputs as they are generated.

### Enabling Streaming

To stream output tokens:

1. Wrap your program with `dspy.streamify`.
2. Create one or more `dspy.streaming.StreamListener` objects for each field you want to stream.

Example:
```python
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
predict = dspy.Predict("question->answer")
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)
```

### Consuming Streamed Output

Since streaming uses an async generator, consume the output in an async context:
```python
import asyncio

async def read_output_stream():
    output_stream = stream_predict(question="Why did a chicken cross the kitchen?")
    async for chunk in output_stream:
        print(chunk)

asyncio.run(read_output_stream())
```
Typical output:
```
StreamResponse(predict_name='self', signature_field_name='answer', chunk='To')
...
Prediction(answer='To get to the other side of the frying pan!')
```

- `StreamResponse`: Wraps each streamed token chunk.
- `Prediction`: Final output of the program.

**Note:** In async environments (e.g., Jupyter), use the generator directly.

### Handling Streamed Chunks

You can distinguish between streamed tokens and final predictions:
```python
async def read_output_stream():
    output_stream = stream_predict(question="Why did a chicken cross the kitchen?")
    async for chunk in output_stream:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(f"Token: {chunk.chunk}")
        elif isinstance(chunk, dspy.Prediction):
            print("Final output:", chunk)
```

#### `StreamResponse` Structure

- `predict_name`: Name of the predictor module.
- `signature_field_name`: Output field being streamed.
- `chunk`: The streamed token(s).

### Streaming with Cache

If a cached result is found, only the final `Prediction` is yielded (no token streaming).

### Streaming Multiple Fields

To stream multiple fields, add a `StreamListener` for each:
```python
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict1 = dspy.Predict("question->answer")
        self.predict2 = dspy.Predict("answer->simplified_answer")
    def forward(self, question: str, **kwargs):
        answer = self.predict1(question=question)
        return self.predict2(answer=answer)

predict = MyModule()
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[
        dspy.streaming.StreamListener(signature_field_name="answer"),
        dspy.streaming.StreamListener(signature_field_name="simplified_answer"),
    ],
)
```
Tokens from both fields will be streamed, each with its corresponding `predict_name` and `signature_field_name`.

### Handling Duplicate Field Names

If multiple modules output fields with the same name, specify both `predict` and `predict_name` in `StreamListener`:
```python
stream_listeners = [
    dspy.streaming.StreamListener(
        signature_field_name="answer",
        predict=predict.predict1,
        predict_name="predict1"
    ),
    dspy.streaming.StreamListener(
        signature_field_name="answer",
        predict=predict.predict2,
        predict_name="predict2"
    ),
]
```
This ensures correct association of streamed tokens with their originating module.

---

**Tip:** The last `StreamResponse` chunk may contain multiple tokens, as listeners buffer tokens until the field is finalized.



================================================================================



# Tutorials and Practical Examples: Streaming: Intermediate Status Streaming


## Streaming: Intermediate Status Streaming

Intermediate status streaming in DSPy allows you to monitor the progress of long-running operations by emitting custom status messages at key points (e.g., when a language model or tool starts or finishes). 

To implement intermediate status streaming:

1. **Subclass `StatusMessageProvider`**: Override hook methods such as `lm_start_status_message`, `tool_start_status_message`, etc., to return custom status strings.
2. **Pass your provider to `dspy.streamify`**: Use your custom provider with `dspy.streamify` alongside your module.

**Example:**
```python
class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    def tool_start_status_message(self, instance, inputs):
        return f"Calling Tool {instance.name} with inputs {inputs}..."
    def tool_end_status_message(self, instance, outputs):
        return f"Tool finished with output: {outputs}!"

stream_predict = dspy.streamify(
    my_module,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="reasoning")],
    status_message_provider=MyStatusMessageProvider(),
)
```

**Available hooks** (override as needed):
- `lm_start_status_message` / `lm_end_status_message`
- `module_start_status_message` / `module_end_status_message`
- `tool_start_status_message` / `tool_end_status_message`

Each hook should return a string describing the current status. In your async output loop, check for `dspy.streaming.StatusMessage` objects to display these updates.



================================================================================



# Tutorials and Practical Examples: Streaming: Synchronous Streaming


## Synchronous Streaming

To stream outputs synchronously (i.e., using a standard Python generator rather than an async generator), set `async_streaming=False` when using `dspy.streamify`. This is useful when you want to process streamed outputs in a blocking, step-by-step manner.

```python
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    async_streaming=False,
)
for chunk in stream_predict(question="..."):
    # process streamed output
```

This approach allows you to iterate over streamed output chunks synchronously, making integration with standard Python code straightforward.



================================================================================



# Tutorials and Practical Examples: Streaming: Other


## Streaming: Other

DSPy supports advanced streaming capabilities to enhance real-time interaction and monitoring in your programs. The two main features are:

- **Output Token Streaming**: Allows you to stream tokens from the model as they are generated, enabling immediate display of partial outputs rather than waiting for the full response.
- **Intermediate Status Streaming**: Provides real-time updates on the internal execution state of your DSPy program, such as indicating when external resources are being called or results are being processed.

These features are useful for building responsive user interfaces and for monitoring long-running or complex DSPy workflows. For implementation details and code examples, refer to the main streaming tutorials.



================================================================================



# Tutorials and Practical Examples: Saving and Loading Programs: State-only Saving


## State-only Saving

Saving only the state of a DSPy program (signature, demos, and configuration) is analogous to weights-only saving in frameworks like PyTorch. This approach excludes the full program architecture, making it suitable for scenarios where you want to preserve learned examples and settings but may reconstruct the program class later.

### Saving State

To save only the state, use the `save` method with `save_program=False`. JSON format is recommended for readability, but use pickle if your program contains non-serializable objects.

```python
compiled_program.save("program.json", save_program=False)   # JSON (recommended)
compiled_program.save("program.pkl", save_program=False)    # Pickle (for non-serializable objects)
```

Optionally, include field metadata (name, type, description, prefix) by setting `save_field_meta=True`:

```python
optimized_program.save("path.json", save_field_meta=True)
```

### Loading State

To load a saved state, first instantiate the program with the same signature, then call `.load()`:

```python
loaded_program = dspy.ChainOfThought("question -> answer")
loaded_program.load("program.json")  # or "program.pkl"
```

If you saved with field metadata, instantiate your program class and load as follows:

```python
loaded_program = YOUR_PROGRAM_CLASS()
loaded_program.load("path.json")
```

This process restores the demos, signature, and configuration, enabling continued use or further optimization of your DSPy program.



================================================================================



# Tutorials and Practical Examples: Saving and Loading Programs: Whole Program Saving


## Whole Program Saving

Starting from DSPy v2.6.0, you can save and reload the entire DSPy program—including its architecture, state, and dependencies—using `cloudpickle`. This is useful for checkpointing, sharing, or deploying trained DSPy programs.

### Saving the Whole Program

Use the `save` method with `save_program=True`, specifying a directory path (not a file), as metadata like dependency versions are also stored:

```python
compiled_dspy_program.save("./dspy_program/", save_program=True)
```

### Loading the Program

Reload the saved program directly with:

```python
loaded_dspy_program = dspy.load("./dspy_program/")
```

### Verifying Program Integrity

You can verify that demos and signatures are preserved:

```python
assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Loaded demo is a dict, original is a dspy.Example.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

With whole program saving, you do not need to reconstruct the program manually; loading restores both architecture and state. Choose this approach when you need to preserve the full program context.



================================================================================



# Tutorials and Practical Examples: Saving and Loading Programs: MLflow Integration


## Saving and Loading Programs: MLflow Integration

DSPy integrates with [MLflow](https://mlflow.org/docs/latest/llms/dspy/index.html) to enable logging of programs, metrics, and execution environments, supporting reproducibility and experiment tracking. Use this integration to systematically record and manage your DSPy workflows. For setup instructions and practical usage examples, refer to the [MLflow Integration guide](https://mlflow.org/docs/latest/llms/dspy/index.html).



================================================================================



# Tutorials and Practical Examples: Saving and Loading Programs: Backward Compatibility


## Backward Compatibility

- **DSPy < 2.7:** Saved programs are *not* guaranteed to be compatible across different DSPy versions. Always use the exact same DSPy version for both saving and loading programs.
- **DSPy ≥ 2.7:** Saved programs are guaranteed to be loadable across all minor releases within the same major version (e.g., you can save in 2.7.0 and load in 2.7.10).

**Tip:** To ensure smooth loading of saved programs, check your DSPy version before saving and loading, especially if collaborating or upgrading environments.



================================================================================



# Tutorials and Practical Examples: Deployment: FastAPI Deployment


## Deployment: FastAPI Deployment

You can serve DSPy programs as REST APIs using FastAPI for lightweight and scalable deployment.

### 1. Install Dependencies

```bash
pip install fastapi uvicorn
export OPENAI_API_KEY="your-openai-api-key"
```

### 2. Minimal FastAPI App

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import dspy

app = FastAPI(title="DSPy Program API", version="1.0.0")

class Question(BaseModel):
    text: str

lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm, async_max_workers=4)
dspy_program = dspy.asyncify(dspy.ChainOfThought("question -> answer"))

@app.post("/predict")
async def predict(question: Question):
    try:
        result = await dspy_program(question=question.text)
        return {"status": "success", "data": result.toDict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3. Streaming Responses (DSPy 2.6.0+)

```python
from dspy.utils.streaming import streaming_response
from fastapi.responses import StreamingResponse

streaming_dspy_program = dspy.streamify(dspy_program)

@app.post("/predict/stream")
async def stream(question: Question):
    stream = streaming_dspy_program(question=question.text)
    return StreamingResponse(streaming_response(stream), media_type="text/event-stream")
```

### 4. Run the Server

```bash
uvicorn fastapi_dspy:app --reload
```

### 5. Test the API

```python
import requests
response = requests.post("http://127.0.0.1:8000/predict", json={"text": "What is the capital of France?"})
print(response.json())
```
Sample output:
```json
{'status': 'success', 'data': {'reasoning': '...', 'answer': 'The capital of France is Paris.'}}
```

**Tips:**
- Use `dspy.asyncify` and `dspy.streamify` for async and streaming endpoints.
- Handle exceptions to return proper HTTP error codes.
- Set your API key in the environment before running the server.



================================================================================



# Tutorials and Practical Examples: Deployment: MLflow Deployment


## Deployment: MLflow Deployment

This section demonstrates how to deploy DSPy programs using MLflow.

### 1. Install MLflow

```bash
pip install mlflow>=2.18.0
```

### 2. Start the MLflow UI

```bash
mlflow ui
```

### 3. Log a DSPy Program

```python
import dspy, mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("deploy_dspy_program")
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
dspy_program = dspy.ChainOfThought("question -> answer")
with mlflow.start_run():
    mlflow.dspy.log_model(
        dspy_program,
        "dspy_program",
        input_example={"messages": [{"role": "user", "content": "What is LLM agent?"}]},
        task="llm/v1/chat",
    )
```

### 4. Serve the Program

```bash
mlflow models serve -m runs:/{run_id}/model -p 6000
```

### 5. Invoke the Model

```bash
curl http://127.0.0.1:6000/invocations -H "Content-Type:application/json" --data '{"messages": [{"content": "what is 2 + 2?", "role": "user"}]}'
```

### 6. Best Practices

- Specify dependencies in `requirements.txt` or `conda.yaml`.
- Use tags and descriptions for model versions.
- Define input schemas/examples.
- Set up logging and monitoring.

### 7. Containerized Deployment (Optional)

```bash
mlflow models build-docker -m "runs:/{run_id}/model" -n "dspy-program"
docker run -p 6000:8080 dspy-program
```

For further details, see the [MLflow DSPy documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).



================================================================================



# Tutorials and Practical Examples: Deployment: Other


## Deployment: Other

DSPy programs can be deployed using various methods, including:

- **FastAPI**: Suitable for lightweight serving of DSPy models as web APIs.
- **MLflow**: Recommended for production-grade deployment with versioning and management capabilities.

Below is a minimal example of initializing a DSPy program, which can then be integrated into your chosen deployment framework:

```python
import dspy
dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))
dspy_program = dspy.ChainOfThought("question -> answer")
```

Integrate the `dspy_program` object into your FastAPI or MLflow workflow as needed.



================================================================================



# Tutorials and Practical Examples: Other


## Tutorials and Practical Examples: Other

This section highlights advanced and miscellaneous DSPy tutorials and practical examples beyond the standard categories.

### Advanced Patterns and Tutorials

- **PAPILLON Tutorial**: Demonstrates advanced DSPy patterns such as multi-stage modules with local LMs and tools, using judge modules as metrics, and optimizing modules via teacher-student setups.
- **Core Development & Deployment**: Tutorials cover MCP tool integration, output refinement (e.g., `BestOfN`, `Refine`), saving/loading, deployment, debugging, observability, streaming, and async programming.
- **Optimization Tutorials**: Include math reasoning, classification finetuning, advanced tool use, and agent finetuning with DSPy optimizers.

Refer to the DSPy documentation for links to these advanced tutorials.

### Practical Examples

#### Ensuring Factual Correctness

```python
import dspy

class FactualityJudge(dspy.Signature):
    """Determine if a statement is factually accurate."""
    statement: str = dspy.InputField()
    is_factual: bool = dspy.OutputField()

factuality_judge = dspy.ChainOfThought(FactualityJudge)

def factuality_reward(args, pred: dspy.Prediction) -> float:
    statement = pred.answer    
    result = factuality_judge(statement)    
    return 1.0 if result.is_factual else 0.0

refined_qa = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"),
    N=3,
    reward_fn=factuality_reward,
    threshold=1.0
)

result = refined_qa(question="Tell me about Belgium's capital city.")
print(result.answer)
```

#### Summarization with Controlled Response Length

```python
import dspy

def ideal_length_reward(args, pred: dspy.Prediction) -> float:
    """
    Reward the summary for being close to 75 words with a tapering off for longer summaries.
    """
    word_count = len(pred.summary.split())
    distance = abs(word_count - 75)
    return max(0.0, 1.0 - (distance / 125))

optimized_summarizer = dspy.BestOfN(
    module=dspy.ChainOfThought("text -> summary"),
    N=50,
    reward_fn=ideal_length_reward,
    threshold=0.9
)

result = optimized_summarizer(
    text="[Long text to summarize...]"
)
print(result.summary)
```

For more advanced use cases and patterns, consult the DSPy documentation and the PAPILLON tutorial.



================================================================================



# Evaluation and Metrics: Defining Metrics: Simple Metrics


## Defining Simple Metrics

In DSPy, a metric is a Python function that quantifies the performance of your system by comparing each data example to its corresponding output. The function typically takes two arguments—`example` (from your dataset) and `pred` (the system's output)—and returns a score (e.g., `bool`, `int`, or `float`). An optional third argument, `trace`, can be used for advanced evaluation or optimization scenarios.

### Basic Metric Example

A simple metric checks if the predicted answer matches the expected answer:

```python
def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()
```

### Built-in Metric Utilities

DSPy provides convenient built-in metrics, such as:
- `dspy.evaluate.metrics.answer_exact_match`
- `dspy.evaluate.metrics.answer_passage_match`

### Custom Metrics with Multiple Properties

Metrics can be more complex, evaluating several conditions. For example, checking both answer correctness and context relevance:

```python
def validate_context_and_answer(example, pred, trace=None):
    answer_match = example.answer.lower() == pred.answer.lower()
    context_match = any((pred.answer.lower() in c) for c in pred.context)
    if trace is None:
        return (answer_match + context_match) / 2.0
    else:
        return answer_match and context_match
```

### Example: Entity Extraction Metric

```python
def extraction_correctness_metric(example, prediction, trace=None):
    return prediction.extracted_people == example.expected_extracted_people
```

### Usage in Evaluation

Metrics are used with DSPy's evaluation utilities:

```python
evaluate_correctness = dspy.Evaluate(
    devset=test_set,
    metric=extraction_correctness_metric,
    num_threads=24,
    display_progress=True,
    display_table=True
)
```

> **Tip:** Start with simple metrics and iterate as you analyze your outputs. Good metrics are critical for effective evaluation and improvement.



================================================================================



# Evaluation and Metrics: Defining Metrics: Advanced Metrics (AI Feedback, Trace)


## Evaluation and Metrics: Advanced Metrics (AI Feedback, Trace)

For complex outputs, such as long-form text, advanced metrics often require assessing multiple qualitative dimensions. Leveraging AI feedback via language models (LMs) enables automated, nuanced evaluation.

### Defining a Metric with AI Feedback

You can define a custom signature for assessment tasks:

```python
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()
```

This signature allows you to query an LM about specific aspects of the output.

#### Example: Multi-Dimensional Metric

The following metric checks if a generated tweet (1) correctly answers a question, (2) is engaging, and (3) fits within 280 characters:

```python
def metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output

    engaging_q = "Does the assessed text make for a self-contained, engaging tweet?"
    correct_q = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"

    correct = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=correct_q)
    engaging = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engaging_q)

    correct, engaging = [m.assessment_answer for m in [correct, engaging]]
    score = (correct + engaging) if correct and (len(tweet) <= 280) else 0

    if trace is not None:
        return score >= 2  # Strict: both criteria must be met
    return score / 2.0    # Otherwise, return a normalized score
```

**Tip:** When compiling (i.e., when `trace is not None`), use stricter criteria to ensure only high-quality outputs pass.



================================================================================



# Evaluation and Metrics: Defining Metrics: Built-in Metrics (SemanticF1, answer_passage_match, etc.)


## Evaluation and Metrics: Built-in Metrics (`SemanticF1`, `answer_passage_match`, etc.)

DSPy provides several built-in metrics to evaluate model outputs beyond exact string matching, focusing on semantic correctness and context relevance.

### `SemanticF1`

`dspy.evaluate.SemanticF1` measures semantic similarity between predicted and reference outputs, making it suitable for tasks where capturing key facts is more important than verbatim matches (e.g., long-form QA). It supports synchronous, asynchronous, and batch evaluation, and can be configured for decompositional scoring.

**Usage Example:**
```python
from dspy.evaluate import SemanticF1
metric = SemanticF1(decompositional=True)
pred = cot(**example.inputs())
score = metric(example, pred)
print(f"Semantic F1 Score: {score:.2f}")
```

To evaluate a model over a dataset with parallelism and progress display:
```python
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24, display_progress=True, display_table=2)
evaluate(cot)
```

For experiment tracking and result visualization with MLflow:
```python
import mlflow
with mlflow.start_run(run_name="rag_evaluation"):
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=24,
        display_progress=True,
        return_all_scores=True,
        return_outputs=True,
    )
    aggregated_score, outputs, all_scores = evaluate(cot)
    mlflow.log_metric("semantic_f1_score", aggregated_score)
    mlflow.log_table(
        {
            "Question": [example.question for example in eval_set],
            "Gold Response": [example.response for example in eval_set],
            "Predicted Response": outputs,
            "Semantic F1 Score": all_scores,
        },
        artifact_file="eval_results.json",
    )
```

You can inspect metric behavior with:
```python
dspy.inspect_history(n=1)
```

### `answer_passage_match`

`dspy.evaluate.answer_passage_match` checks whether the predicted answer appears within any provided passages or contexts. It returns a boolean or float score indicating passage match, and is commonly used for evaluating retrieval-augmented generation (RAG) and open-domain QA tasks.

---

These metrics enable robust, interpretable evaluation for DSPy programs, supporting both detailed analysis and scalable experiment tracking.



================================================================================



# Evaluation and Metrics: Defining Metrics: DSPy Program as Metric


## Evaluation and Metrics: Defining Metrics with DSPy Programs

To evaluate and refine your DSPy pipelines, define a metric—a function that scores system outputs. For simple tasks, this can be a straightforward function (e.g., accuracy). For complex outputs, you can use a DSPy program itself as the metric, such as an LLM-based judge.

### Using a DSPy Program as a Metric

A DSPy program can serve as a metric, especially for tasks where evaluation requires reasoning or judgment (e.g., factuality). Since the metric's output is typically a simple value (like a score or boolean), you can optimize (compile) the metric itself by collecting a small set of labeled examples.

#### Example: LLM as Judge

```python
class FactJudge(dspy.Signature):
    """Judge if the answer is factually correct based on the context."""
    context = dspy.InputField(desc="Context for the prediction")
    question = dspy.InputField(desc="Question to be answered")
    answer = dspy.InputField(desc="Answer for the question")
    factually_correct = dspy.OutputField(desc="Is the answer factually correct based on the context?", prefix="Factual[Yes/No]:")

judge = dspy.ChainOfThought(FactJudge)

def factuality_metric(example, pred):
    factual = judge(context=example.context, question=example.question, answer=pred.answer)
    return int(factual == "Yes")
```

### Optimizing DSPy Metrics

When your metric is a DSPy program, you can compile (optimize) it directly. During compilation, DSPy traces LM calls, allowing you to inspect intermediate steps for validation and debugging.

#### Accessing the Trace

You can use the `trace` argument to validate intermediate outputs:

```python
def validate_hops(example, pred, trace=None):
    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]
    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False
    return True
```

### Function-Based Metrics

For simpler tasks, define a metric as a Python function returning a number or boolean:

```python
def parse_integer_answer(answer, only_first_line=True):
    try:
        if only_first_line:
            answer = answer.strip().split('\n')[0]
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split('.')[0]
        answer = ''.join([c for c in answer if c.isdigit()])
        answer = int(answer)
    except (ValueError, IndexError):
        answer = 0
    return answer

def gsm8k_metric(gold, pred, trace=None) -> int:
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
```

### Tips

- Start with simple metrics and iterate as your system evolves.
- Use a small development set (even 20–200 examples) for evaluation and optimization.
- When using DSPy programs as metrics, leverage the trace for advanced validation and debugging.



================================================================================



# Evaluation and Metrics: Evaluation Utilities: Evaluate Class


## Evaluation Utilities: `Evaluate` Class

The `Evaluate` class in DSPy streamlines the evaluation of programs on a development set, supporting parallel execution and detailed result display.

### Usage

```python
from dspy.evaluate import Evaluate

# Instantiate the evaluator
evaluator = Evaluate(
    devset=YOUR_DEVSET,           # Iterable of evaluation examples
    metric=YOUR_METRIC,           # Function to compute metric per example
    num_threads=NUM_THREADS,      # (Optional) Parallelism
    display_progress=True,        # (Optional) Show progress bar
    display_table=5               # (Optional) Show first N results in a table
)

# Run evaluation
evaluator(YOUR_PROGRAM)
```

- `devset`: Iterable of evaluation examples.
- `metric`: Function taking (example, prediction) and returning a score.
- `num_threads`: Number of parallel threads (default: 1).
- `display_progress`: Whether to show a progress bar.
- `display_table`: Number of results to display in a summary table.

This utility is recommended over manual evaluation loops for efficiency and richer output.



================================================================================



# Evaluation and Metrics: Evaluation Utilities: MLflow for Evaluation


## Evaluation Utilities: MLflow for Evaluation

To track and visualize evaluation results over time, DSPy integrates seamlessly with MLflow. After evaluating your extractor (or other program) on a test set, you can log both summary metrics and detailed outputs to MLflow for experiment tracking.

**Example: Logging Evaluation Results with MLflow**

```python
import mlflow

with mlflow.start_run(run_name="extractor_evaluation"):
    evaluator = dspy.Evaluate(
        devset=test_set,
        metric=extraction_correctness_metric,
        num_threads=24,
        return_all_scores=True,
        return_outputs=True,
    )
    aggregated_score, outputs, all_scores = evaluator(people_extractor)
    mlflow.log_metric("exact_match", aggregated_score)
    mlflow.log_table(
        {
            "Tokens": [ex.tokens for ex in test_set],
            "Expected": [ex.expected_extracted_people for ex in test_set],
            "Predicted": outputs,
            "Exact match": all_scores,
        },
        artifact_file="eval_results.json",
    )
```

- `mlflow.log_metric` records scalar metrics (e.g., accuracy).
- `mlflow.log_table` logs detailed per-example results for later analysis.

For more details, see the [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).



================================================================================



# Optimization in DSPy: DSPy Optimizers Overview: Types of Optimizers


## DSPy Optimizers Overview: Types of Optimizers

DSPy optimizers are algorithms that tune the prompts and/or language model (LM) weights of your DSPy program to maximize a user-defined metric (e.g., accuracy). Optimizers require your DSPy program, a metric function, and a set of training examples (even a small set can suffice). Optimizers can be chained or ensembled for improved results.

### Categories of DSPy Optimizers

#### 1. Automatic Few-Shot Learning
- **LabeledFewShot**: Constructs few-shot examples from labeled data. Requires `k` (number of examples) and a `trainset`.
- **BootstrapFewShot**: Uses a teacher module (defaults to your program) to generate demonstrations for each stage, validated by your metric. Parameters: `max_labeled_demos`, `max_bootstrapped_demos`, and optional custom teacher.
- **BootstrapFewShotWithRandomSearch**: Runs `BootstrapFewShot` multiple times with random search over demos, selecting the best-performing program. Adds `num_candidate_programs` parameter.
- **KNNFewShot**: Selects nearest-neighbor demos for each input using k-NN, then applies `BootstrapFewShot`.

#### 2. Automatic Instruction Optimization
- **COPRO**: Generates and refines instructions for each step using coordinate ascent, guided by the metric. Parameter: `depth` (number of improvement iterations).
- **MIPROv2**: Jointly optimizes instructions and few-shot examples using Bayesian Optimization, considering both data and demonstrations.

#### 3. Automatic Finetuning
- **BootstrapFinetune**: Distills a prompt-based program into LM weight updates, producing a finetuned model for each step.

#### 4. Program Transformations
- **Ensemble**: Combines multiple DSPy programs, using either the full set or a random subset.

### Usage Tips
- Optimizers can require just a trainset, or both trainset and validation set.
- For prompt optimization, a 20% train / 80% validation split is recommended.
- Typical optimization runs are cost-effective, but depend on LM and dataset size.
- Optimizers can be chained (e.g., instruction optimization followed by finetuning) or ensembled.
- For advanced control, use DSPy Assertions or custom metrics.

For detailed API and parameter documentation, see the [DSPy Optimizers API Reference](https://dspy-docs-url/api/optimizers/).



================================================================================



# Optimization in DSPy: DSPy Optimizers Overview: Choosing an Optimizer


## DSPy Optimizers Overview: Choosing an Optimizer

Selecting the right optimizer in DSPy is an iterative process that depends on your data size, optimization goals, and computational resources. Here are general guidelines to help you get started:

- **Very few examples (~10):** Use `BootstrapFewShot`.
- **More data (≥50 examples):** Try `BootstrapFewShotWithRandomSearch`.
- **Instruction optimization only (0-shot prompts):** Use `MIPROv2` configured for 0-shot optimization.
- **Longer optimization runs (≥40 trials) with sufficient data (≥200 examples):** Use `MIPROv2` for more thorough optimization.
- **Efficient programs with large LMs (≥7B parameters):** Consider finetuning a smaller LM for your task using `BootstrapFinetune`.

Experimentation is key—expect to iterate on both optimizer choice and configuration to achieve the best results for your task.

### Next Steps and Further Optimization

After initial optimization, you can further improve your system by:
- Exploring alternative program architectures (e.g., using LMs to generate search queries).
- Trying different prompt or weight optimizers.
- Scaling inference with ensembling or other techniques.
- Reducing costs by distilling to smaller LMs.

Always analyze your system outputs to identify bottlenecks, refine your evaluation metrics, and collect more or higher-quality data as needed.



================================================================================



# Optimization in DSPy: DSPy Optimizers Overview: How MIPROv2 Works


## Optimization in DSPy: DSPy Optimizers Overview — How MIPROv2 Works

`MIPROv2` is a DSPy optimizer that jointly optimizes few-shot examples and natural-language instructions for each predictor in your LM program using Bayesian Optimization. Its workflow consists of:

1. **Bootstrapping Few-Shot Examples:** Runs the program on many training inputs, collecting traces and retaining those with correct outputs or high metric scores. Multiple candidate sets of labeled demos are created.
2. **Proposing Instruction Candidates:** For each predictor, generates instruction candidates by analyzing the dataset, program code, selected demos, and random tips, using a prompt model.
3. **Bayesian Optimization (Discrete Search):** Iteratively proposes and evaluates combinations of instructions and demos on minibatches of the validation set. A surrogate model guides the search, and the best candidates are periodically evaluated on the full set. The combination with the highest validation performance is selected.

MIPROv2 can be composed with other DSPy optimizers (e.g., using its output as input to `BootstrapFinetune` or ensembling top candidates with `Ensemble`), supporting both inference-time and pre-inference optimization.

For further details and benchmarks, see the [MIPROv2 paper](https://arxiv.org/abs/2406.11695).



================================================================================



# Optimization in DSPy: DSPy Optimizers Overview: Optimizer Usage Patterns


## DSPy Optimizers Overview: Optimizer Usage Patterns

DSPy provides a suite of optimizers for prompt and weight tuning, each following a common interface: instantiate the optimizer with a metric and configuration, then call `.compile()` with your DSPy program and a training set. Typical usage costs depend on the language model and dataset size.

### Common Usage Pattern

```python
optimizer = OptimizerType(metric=your_metric, **config)
optimized_program = optimizer.compile(your_dspy_program, trainset=your_trainset)
optimized_program.save("path.json")  # Save for reuse
```

### Optimizer Types and Examples

- **LabeledFewShot**: Constructs few-shot prompts from labeled data.
  ```python
  from dspy.teleprompt import LabeledFewShot
  optimizer = LabeledFewShot(k=8)
  compiled = optimizer.compile(student=your_dspy_program, trainset=trainset)
  ```

- **BootstrapFewShot**: Bootstraps demos using a metric and optionally a teacher LM.
  ```python
  from dspy.teleprompt import BootstrapFewShot
  optimizer = BootstrapFewShot(metric=your_metric, max_bootstrapped_demos=4, max_labeled_demos=16)
  compiled = optimizer.compile(student=your_dspy_program, trainset=trainset)
  ```

- **BootstrapFewShotWithRandomSearch**: Randomly selects demos and searches over candidate programs.
  ```python
  from dspy.teleprompt import BootstrapFewShotWithRandomSearch
  optimizer = BootstrapFewShotWithRandomSearch(metric=your_metric, max_bootstrapped_demos=2, num_candidate_programs=8)
  compiled = optimizer.compile(student=your_dspy_program, trainset=trainset, valset=devset)
  ```

- **Ensemble**: Combines multiple compiled programs.
  ```python
  from dspy.teleprompt.ensemble import Ensemble
  ensemble = Ensemble(reduce_fn=dspy.majority)
  compiled_ensemble = ensemble.compile([prog1, prog2, prog3])
  ```

- **BootstrapFinetune**: Finetunes LM weights using bootstrapped demos.
  ```python
  from dspy.teleprompt import BootstrapFinetune
  optimizer = BootstrapFinetune(metric=your_metric)
  finetuned = optimizer.compile(your_dspy_program, trainset=finetune_data, target=model_to_finetune)
  ```

- **COPRO**: Optimizes instructions via coordinate ascent.
  ```python
  from dspy.teleprompt import COPRO
  optimizer = COPRO(prompt_model=prompt_lm, metric=your_metric, breadth=2, depth=2)
  compiled = optimizer.compile(your_dspy_program, trainset=trainset)
  ```

- **MIPRO / MIPROv2**: Bayesian optimization of instructions and demos.
  ```python
  from dspy.teleprompt import MIPROv2
  optimizer = MIPROv2(metric=your_metric, auto="light")
  optimized = optimizer.compile(your_dspy_program, trainset=trainset, max_bootstrapped_demos=3, max_labeled_demos=4)
  ```

- **Signature Optimizer**: Optimizes signature types for better structure.
  ```python
  from dspy.teleprompt.signature_opt_typed import optimize_signature
  compiled = optimize_signature(student=TypedChainOfThought("question -> answer"), evaluator=Evaluate(...), n_iterations=50).program
  ```

- **KNNFewShot**: Uses nearest neighbor retrieval for demo selection.
  ```python
  from dspy.teleprompt import KNNFewShot
  optimizer = KNNFewShot(k=3, trainset=trainset, vectorizer=Embedder(...))
  compiled = optimizer.compile(student=ChainOfThought("question -> answer"))
  ```

- **BootstrapFewShotWithOptuna**: Uses Optuna for demo hyperparameter search.
  ```python
  from dspy.teleprompt import BootstrapFewShotWithOptuna
  optimizer = BootstrapFewShotWithOptuna(metric=your_metric, max_bootstrapped_demos=2, num_candidate_programs=8)
  compiled = optimizer.compile(student=your_dspy_program, trainset=trainset, valset=devset)
  ```

### Full Example: Optimizing a Math Program with MIPROv2

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import MIPROv2

lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
dspy.configure(lm=lm)

optimizer = MIPROv2(metric=gsm8k_metric, auto="medium")
optimized_program = optimizer.compile(
    dspy.ChainOfThought("question -> answer"),
    trainset=gsm8k.train,
    requires_permission_to_run=False,
)
optimized_program.save("optimized.json")
```

### Additional Tips

- Use `.save(path)` and `.load(path)` to persist and reload optimized programs.
- For advanced configuration, refer to each optimizer's documentation.
- Typical optimization runs cost a few cents to a few dollars, depending on the LM and data size.



================================================================================



# Optimization in DSPy: DSPy Optimizers Overview: Optimizer Research Directions


## Optimizer Research Directions

DSPy optimizer research targets three core areas:
- **Quality:** Enhancing performance across language models, aiming for improvements even from weak zero-shot baselines.
- **Cost:** Lowering requirements for labeled/unlabeled data and minimizing inference costs.
- **Robustness:** Ensuring optimizers generalize to unseen data and are resilient to metric or label errors.

Current and future work includes:
- Systematic benchmarking of optimizers.
- Achieving substantial quality improvements over existing methods (e.g., MIPROv2, BetterTogether).
- Creating optimizers that perform well with limited data, weak initial programs, or minimal supervision.



================================================================================



# Optimization in DSPy: DSPy Optimizers Overview: Assertion-Driven Optimization


## Assertion-Driven Optimization in DSPy

DSPy supports assertion-driven optimization, particularly with the `BootstrapFewShotWithRandomSearch` optimizer. Assertions can be used in two main modes:

- **Compilation with Assertions**: Assertions guide the teacher model during the bootstrapping of few-shot examples. The student model learns from these robust, assertion-validated examples but does not use assertions during inference (i.e., no assertion-aware backtracking or retry).
- **Compilation + Inference with Assertions**: Both teacher and student models use assertions. The teacher provides assertion-driven examples, and the student further leverages assertions for optimization at inference time.

**Example:**

```python
teleprompter = BootstrapFewShotWithRandomSearch(
    metric=validate_context_and_answer_and_hops,
    max_bootstrapped_demos=max_bootstrapped_demos,
    num_candidate_programs=6,
)

# Compilation with Assertions (student does not use assertions at inference)
compiled_with_assertions_baleen = teleprompter.compile(
    student=baleen,
    teacher=baleen_with_assertions,
    trainset=trainset,
    valset=devset,
)

# Compilation + Inference with Assertions (student uses assertions at inference)
compiled_baleen_with_assertions = teleprompter.compile(
    student=baleen_with_assertions,
    teacher=baleen_with_assertions,
    trainset=trainset,
    valset=devset,
)
```

Use the appropriate setup depending on whether you want assertion-driven optimizations only during compilation or also during inference.



================================================================================



# Optimization in DSPy: DSPy Optimizers Overview: Online RL with GRPO


## Optimization in DSPy: DSPy Optimizers Overview — Online RL with `dspy.GRPO`

DSPy provides experimental support for online reinforcement learning (RL) over multi-module programs via the `dspy.GRPO` optimizer. This enables fine-tuning of small local language models (e.g., 1.7B–7B parameters) to optimize for custom evaluation metrics, using external RL servers such as Arbor. This feature is highly experimental and intended for advanced users.

### Workflow Overview

1. **Set up the RL environment:**
   - Install [Arbor](https://github.com/arbor-ai/arbor) and start the RL server:
     ```bash
     pip install arbor-ai
     python -m arbor.cli serve --arbor-config arbor.yaml
     ```
     Example `arbor.yaml`:
     ```yaml
     inference:
       gpu_ids: '0'
     training:
       gpu_ids: '1, 2'
     ```
2. **Configure your local LM in DSPy:**
   ```python
   import dspy
   from dspy.clients.lm_local_arbor import ArborProvider

   port = 7453
   local_lm = dspy.LM(
       model="openai/arbor:Qwen/Qwen2.5-7B-Instruct",
       provider=ArborProvider(),
       temperature=0.7,
       api_base=f"http://localhost:{port}/v1/",
       api_key="arbor",
   )
   dspy.configure(lm=local_lm)
   ```
3. **Prepare your multi-module DSPy program** (e.g., for multi-hop retrieval or privacy-preserving tasks).
4. **Specify evaluation metrics** (e.g., recall, quality, privacy leakage).
5. **Evaluate the zero-shot baseline.**
6. **Optimize with `dspy.GRPO`:**
   ```python
   from dspy.teleprompt.grpo import GRPO

   program = ResearchHop(num_docs=4, num_hops=2)
   program.set_lm(local_lm)

   train_kwargs = {
       "per_device_train_batch_size": 2,
       "gradient_accumulation_steps": 4,
       "learning_rate": 2e-5,
       "bf16": True,
       "lora": True,
       # ...other settings...
   }

   compiler = GRPO(
       metric=recall,
       multitask=True,
       num_dspy_examples_per_grpo_step=6,
       num_samples_per_input=8,
       num_train_steps=500,
       num_threads=24,
       train_kwargs=train_kwargs,
   )

   optimized_program = compiler.compile(
       student=program,
       trainset=trainset,
       valset=devset,
   )

   # Use the optimized program
   example = devset[0]
   optimized_program(**example.inputs())
   ```
   *In experiments, ~18 hours of GRPO training improved recall from 61.8% to 66.2% on the dev set.*

### Tutorial: Multi-Hop Research Example

**Dependencies:**
```bash
pip install -U bm25s PyStemmer "jax[cpu]"
```

**Download and index Wikipedia abstracts:**
```python
from dspy.utils import download
download("https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz")
!tar -xzvf wiki.abstracts.2017.tar.gz

import ujson, bm25s, Stemmer
corpus = []
with open("wiki.abstracts.2017.jsonl") as f:
    for line in f:
        line = ujson.loads(line)
        corpus.append(f"{line['title']} | {' '.join(line['text'])}")
stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
retriever = bm25s.BM25(k1=0.9, b=0.4)
retriever.index(corpus_tokens)
```

**Load the HoVer dataset:**
```python
import random
from dspy.datasets import DataLoader

kwargs = dict(fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",))
hover = DataLoader().from_huggingface(dataset_name="hover-nlp/hover", split="train", trust_remote_code=True, **kwargs)
hpqa_ids = set()
hover = [
    dspy.Example(claim=x.claim, titles=list(set([y["key"] for y in x.supporting_facts]))).with_inputs("claim")
    for x in hover
    if x["num_hops"] == 3 and x["hpqa_id"] not in hpqa_ids and not hpqa_ids.add(x["hpqa_id"])
]
random.Random(0).shuffle(hover)
trainset, devset, testset = hover[:600], hover[600:900], hover[900:]
```

**BM25 Search Function:**
```python
def search(query: str, k: int) -> list[str]:
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = {corpus[doc]: float(score) for doc, score in zip(results[0], scores[0])}
    return list(run.keys())
```

### Tips and Caveats

- `dspy.GRPO` is highly experimental and under active development.
- Requires running a local RL server (Arbor) and a compatible local LM.
- While `dspy.GRPO` enables online RL for arbitrary LM programs, prompt optimizers like `dspy.MIPROv2` or `dspy.SIMBA` are usually more cost-effective for most use cases.
- For the latest usage patterns and updates, refer to the [official DSPy tutorials](https://github.com/stanfordnlp/dspy).



================================================================================



# Optimization in DSPy: DSPy Optimizers Overview: Saving and Loading Optimized Programs


## Saving and Loading Optimized Programs

To reuse optimized DSPy programs without re-optimizing, you can save and load them using DSPy's built-in methods or integrate with MLflow for reproducibility and collaboration.

### Saving and Loading with DSPy

Save an optimized program:
```python
optimized_program.save("optimized_program.json")
```

Load the saved program:
```python
loaded_program = ProgramClass()  # Replace with your program's class
loaded_program.load("optimized_program.json")
result = loaded_program(input_args)  # Use as usual
```
Example:
```python
optimized_rag.save("optimized_rag.json")
loaded_rag = RAG()
loaded_rag.load("optimized_rag.json")
result = loaded_rag(question="cmd+tab does not work on hidden or minimized windows")
```

### Saving and Loading with MLflow

To enable reproducibility, environment tracking, and collaboration, use MLflow integration:

```python
import mlflow

with mlflow.start_run(run_name="optimized_program"):
    model_info = mlflow.dspy.log_model(optimized_program, artifact_path="model")

loaded = mlflow.dspy.load_model(model_info.model_uri)
```

MLflow tracks environment metadata, program performance, and simplifies sharing with collaborators.

For more details, see the [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).



================================================================================



# Optimization in DSPy: DSPy Optimizers Overview: Deployment and Reproducibility


## Optimization in DSPy: Optimizers Overview – Deployment and Reproducibility

### Saving and Loading Compiled Programs

To ensure reproducibility, save a compiled DSPy program using `.save("path.json")` and load it into a new instance with `.load("path.json")`:

```python
compiled = teleprompter.compile(CoT(), trainset=trainset, valset=devset)
compiled.save('compiled_cot.json')
cot = CoT(); cot.load('compiled_cot.json')
```

### Exporting for Deployment

Deployment is achieved by saving the compiled program as above. The exported JSON can be loaded in any compatible environment.

### Custom Data Retrieval

For advanced retrieval within DSPy pipelines, integrate external libraries such as [RAGatouille](https://github.com/bclavie/ragatouille) to enable search over your own data (e.g., using ColBERT).

### Cache Control

- **Disable LM Caching:**  
  ```python
  dspy.LM('openai/gpt-4o-mini', cache=False)
  ```
- **Set Cache Directory:**  
  ```python
  os.environ["DSPY_CACHEDIR"] = "/your/cache/dir"
  ```
- **Serverless Environments:**  
  For AWS Lambda and similar, disable both `DSP_*` and `DSPY_*` caches to avoid issues.

These practices ensure your DSPy optimizers are reproducible, portable, and ready for reliable deployment.



================================================================================



# API Reference: dspy.Predict


## API Reference: dspy.Predict

`dspy.Predict` is a core DSPy module for making predictions with language models. It supports:

- **Synchronous prediction:**  
  - `__call__(...)` / `forward(...)`
- **Asynchronous prediction:**  
  - `acall(...)` / `aforward(...)`
- **Batch processing:**  
  - `batch(...)`
- **Language model configuration:**  
  - `set_lm(...)`
- **State management:**  
  - `save(...)` / `load(...)`

Use `dspy.Predict` as a foundational component for building custom DSPy modules and pipelines.



================================================================================



# API Reference: dspy.ReAct


## API Reference: `dspy.ReAct`

`dspy.ReAct` is a core class in DSPy for implementing the ReAct (Reasoning and Acting) paradigm, enabling step-wise reasoning and action selection with language models. Below is a concise reference of its main methods and attributes.

### Methods

- **`__call__(...)`**  
  Invokes the ReAct module on input data.

- **`acall(...)`**  
  Asynchronous version of `__call__`.

- **`batch(...)`**  
  Processes a batch of inputs in parallel.

- **`deepcopy()`**  
  Returns a deep copy of the ReAct instance.

- **`dump_state()`**  
  Serializes the current state of the module.

- **`forward(...)`**  
  Core forward computation logic.

- **`get_lm()`**  
  Returns the underlying language model.

- **`load(...)`**  
  Loads a saved ReAct module from disk.

- **`load_state(...)`**  
  Loads the module state from a serialized object.

- **`map_named_predictors(...)`**  
  Applies a function to all named predictors.

- **`named_parameters()`**  
  Returns an iterator over named parameters.

- **`named_predictors()`**  
  Returns an iterator over named predictors.

- **`named_sub_modules()`**  
  Returns an iterator over named sub-modules.

- **`parameters()`**  
  Returns an iterator over module parameters.

- **`predictors()`**  
  Returns an iterator over predictors.

- **`reset_copy()`**  
  Resets and returns a copy of the module.

- **`save(...)`**  
  Saves the module to disk.

- **`set_lm(...)`**  
  Sets the underlying language model.

- **`truncate_trajectory(...)`**  
  Truncates the reasoning/action trajectory.

### Tips

- Use `save()` and `load()` to persist and restore module state.
- Use `set_lm()` and `get_lm()` to manage the language model backend.
- Use batch processing methods for efficiency on multiple inputs.

For detailed signatures and source code, refer to the DSPy documentation or source files.



================================================================================



# API Reference: dspy.Example and dspy.Prediction


## API Reference: dspy.Example and dspy.Prediction

### dspy.Example

`dspy.Example` serves as a base class for representing data examples in DSPy. It provides methods such as:
- `inputs()`: Returns the input fields.
- `labels()`: Returns the label fields.
- `toDict()`: Converts the example to a dictionary.
- `with_inputs()`: Returns a copy with updated inputs.

### dspy.Prediction

`dspy.Prediction` is a subclass of `dspy.Example` used to encapsulate the outputs of DSPy modules. It inherits all methods from `Example`, making it easy to manipulate, inspect, and convert predicted outputs within DSPy pipelines.

**Typical usage:**  
Use `dspy.Prediction` objects to represent and process the results produced by DSPy modules, leveraging the inherited methods for convenient access and transformation.



================================================================================



# API Reference: dspy.LM


## API Reference: dspy.LM

`dspy.LM` is the foundational class for integrating language models within DSPy. It provides a unified interface for configuring, invoking, and managing LMs in both synchronous and asynchronous contexts.

**Key Methods:**
- `__call__`, `forward`: Synchronously invoke the language model.
- `acall`, `aforward`: Asynchronously invoke the language model.
- `copy()`: Create a duplicate of the LM instance.
- `dump_state()`: Serialize and export the current LM state.
- `finetune()`: Finetune the LM, if supported by the backend.
- `inspect_history()`: Access recent LM calls and associated metadata.
- `kill()`, `launch()`: Start or stop the LM process (for local models).
- `update_global_history()`: Update the global usage history.

Use `dspy.LM` to configure and interact with language models across DSPy modules and pipelines, supporting flexible integration and management of LM state and execution.



================================================================================



# API Reference: dspy.Adapter, JSONAdapter, TwoStepAdapter


## API Reference: dspy.Adapter, JSONAdapter, TwoStepAdapter

### dspy.Adapter

`dspy.Adapter` is the foundational class for formatting, parsing, and managing structured input/output in DSPy modules. It standardizes prompt construction, output parsing, and conversation history management. Key methods include:

- `format`: Formats input data for prompts.
- `parse`: Parses model outputs into structured data.
- `format_conversation_history`: Prepares conversation context.
- `__call__` / `acall`: Synchronous and asynchronous execution interfaces.

Use `Adapter` subclasses to ensure consistent I/O formatting between DSPy modules and language models.

### dspy.JSONAdapter

`dspy.JSONAdapter` is a subclass of `dspy.Adapter` that specializes in JSON-based I/O formatting and parsing. Use this adapter when you require structured JSON input/output between modules and language models.

### dspy.TwoStepAdapter

`dspy.TwoStepAdapter` extends the Adapter pattern for two-step or multi-stage prompting scenarios. It provides:

- All core Adapter methods (`format`, `parse`, `format_conversation_history`)
- Specialized formatting for user/assistant messages and field descriptions

Use `TwoStepAdapter` for advanced prompt structuring in workflows that require multiple interaction steps.

**Tip:** Choose the adapter that matches your module's I/O requirements: use `Adapter` or `JSONAdapter` for standard cases, and `TwoStepAdapter` for multi-stage prompts.



================================================================================



# API Reference: dspy.BestOfN and dspy.Refine


## API Reference: dspy.BestOfN and dspy.Refine

### Overview

`dspy.BestOfN` and `dspy.Refine` are modules for robust prediction selection in DSPy. Both run a given module multiple times (up to `N` attempts), using a `reward_fn` to select the best output or the first to meet a specified `threshold`. They differ in their approach to iterative improvement.

- **BestOfN**: Runs the module with different temperature settings, selecting the best result according to `reward_fn` or the first that meets `threshold`.
- **Refine**: Extends `BestOfN` by introducing a feedback loop—after each unsuccessful attempt (except the last), it generates feedback on the previous output and uses this as a hint for the next run, enabling iterative refinement.

### BestOfN

Runs the provided module up to `N` times, each with a different temperature. Returns the first prediction meeting the `threshold` or the one with the highest reward if none meet it.

**Example:**
```python
import dspy

def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_3 = dspy.BestOfN(
    module=dspy.ChainOfThought("question -> answer"),
    N=3,
    reward_fn=one_word_answer,
    threshold=1.0
)

result = best_of_3(question="What is the capital of Belgium?")
print(result.answer)  # Brussels
```

**Error Handling:**  
By default, errors are tolerated up to `N` attempts. Control this with `fail_count`:
```python
best_of_3 = dspy.BestOfN(
    module=qa,
    N=3,
    reward_fn=one_word_answer,
    threshold=1.0,
    fail_count=1
)
# Raises an error after the first failure
```

### Refine

`Refine` adds an automatic feedback loop: after each failed attempt (except the last), it generates feedback about the previous output and uses it as a hint for the next run, supporting iterative improvement.

**Example:**
```python
refine = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"),
    N=3,
    reward_fn=one_word_answer,
    threshold=1.0
)
result = refine(question="What is the capital of Belgium?")
```

**Error Handling:**  
Set `fail_count` to control how many failures are allowed before stopping (e.g., `fail_count=1` stops after the first error).

### Summary Table

| Module     | Approach                                  | Feedback Loop | Error Handling    |
|------------|-------------------------------------------|---------------|------------------|
| BestOfN    | Multiple attempts, select best/threshold  | No            | `fail_count`     |
| Refine     | Multiple attempts, iterative refinement   | Yes           | `fail_count`     |

### Version Note

In DSPy 2.6+, `dspy.Suggest` and `dspy.Assert` are replaced by `dspy.Refine` and `dspy.BestOfN`.



================================================================================



# API Reference: dspy.Parallel


## API Reference: dspy.Parallel

`dspy.Parallel` is a DSPy module designed for parallel execution of submodules or tasks. It provides `__call__` and `forward` methods to facilitate running operations concurrently. Use `dspy.Parallel` when you need to process multiple inputs or submodules at the same time within a DSPy program.



================================================================================



# API Reference: dspy.MultiChainComparison


## API Reference: dspy.MultiChainComparison

`dspy.MultiChainComparison` is a DSPy module designed to compare outputs from multiple reasoning chains within a DSPy program. It enables evaluation and comparison of different reasoning paths, supporting standard module methods such as `__call__`, `acall`, `batch`, `forward`, `save`, `load`, and language model (LM) configuration. Use this module to systematically assess and select among alternative reasoning strategies.



================================================================================



# API Reference: dspy.BootstrapFewShot, BootstrapFewShotWithRandomSearch, BootstrapFinetune, MIPROv2, COPRO, Ensemble, KNNFewShot, SIMBA, GRPO


## API Reference: dspy.BootstrapFewShot, BootstrapFewShotWithRandomSearch, BootstrapFinetune

### dspy.BootstrapFewShot

A DSPy optimizer for automatic few-shot prompt synthesis using labeled data and/or teacher modules.

**Main methods:**
- `compile(student, trainset, ...)`: Compile and optimize a DSPy program using few-shot demonstrations.
- `get_params()`: Retrieve optimizer parameters.

### dspy.BootstrapFewShotWithRandomSearch

Variant of BootstrapFewShot that uses random search for prompt selection.

**Main methods:**
- `compile(student, trainset, ...)`: Compile and optimize using random search over few-shot demos.
- `get_params()`: Retrieve optimizer parameters.

### dspy.BootstrapFinetune

Optimizer for finetuning language models using DSPy programs.

**Main methods:**
- `compile(student, trainset, ...)`: Compile and optimize a DSPy program via finetuning.
- `convert_to_lm_dict()`: Convert the optimizer state to a format suitable for language models.
- `finetune_lms()`: Finetune language models with the optimized program.
- `get_params()`: Retrieve optimizer parameters.

> For usage patterns and tutorials, refer to the DSPy documentation and relevant examples.



================================================================================



# API Reference: Other


## Other Utilities

### `dspy.disable_logging()`

Disables all DSPy logging output.  
Use this function as a quick way to silence DSPy logs without manually configuring the logging library.

**Example:**
```python
import dspy
dspy.disable_logging()
```



================================================================================



# Advanced Features: Streaming and Observability: Streaming API and Status Messages


## Advanced Features: Streaming and Observability — Streaming API and Status Messages

DSPy supports real-time status updates during streaming via the `dspy.streaming.StatusMessageProvider` class. By subclassing and overriding its hook methods, you can customize status messages for various execution stages:

- `lm_start_status_message` / `lm_end_status_message`
- `module_start_status_message` / `module_end_status_message`
- `tool_start_status_message` / `tool_end_status_message`

To use custom status messages, pass your `StatusMessageProvider` instance to `dspy.streamify`. This allows you to control the feedback shown during streaming execution.

**Example:**
```python
class MyStatusProvider(dspy.streaming.StatusMessageProvider):
    def lm_start_status_message(self, *args, **kwargs):
        return "Language model started..."
    # Override other hooks as needed

provider = MyStatusProvider()
dspy.streamify(..., status_message_provider=provider)
```



================================================================================



# Advanced Features: Streaming and Observability: MLflow Tracing


## Advanced Features: Streaming and Observability—MLflow Tracing

DSPy offers seamless integration with [MLflow](https://mlflow.org/docs/latest/llms/tracing/index.html) for automatic tracing of LLM predictions, requiring no API key. This enables robust experiment tracking and observability for your DSPy programs.

### Enabling MLflow Tracing

1. **Install MLflow (v2.18.0 or later):**
   ```bash
   pip install -U mlflow>=2.18.0
   ```
2. **Enable DSPy autologging:**
   ```python
   import mlflow
   mlflow.dspy.autolog()
   mlflow.set_experiment("DSPy")  # Optional: organize traces by experiment
   ```
3. **Run your DSPy agent/program as usual.** MLflow will automatically record traces of LLM predictions and program executions.

4. **View traces in the MLflow UI:**
   ```bash
   mlflow ui --port 5000
   ```

For a detailed walkthrough, see the [DSPy + MLflow deployment tutorial](../deployment/#deploying-with-mlflow).

### Observability and Optimization Ecosystem

DSPy supports integration with multiple observability tools (MLflow, Phoenix Arize, LangWatch, Weights & Biases Weave) for real-time tracking, prompt inspection (`inspect_history`), and iterative optimization. Future releases will deepen these integrations, add cost management, and support interactive, human-in-the-loop optimization workflows.



================================================================================



# Advanced Features: Streaming and Observability: Custom Logging and Callbacks


## Advanced Features: Streaming and Observability: Custom Logging and Callbacks

DSPy supports advanced observability via a flexible callback mechanism, allowing you to implement custom logging and monitoring for modules, language models, and adapters.

### Callback Handlers

Extend `dspy.utils.callback.BaseCallback` to hook into various execution events:

| Handler | Triggered When |
|:--|:--|
| `on_module_start` / `on_module_end` | A `dspy.Module` is invoked |
| `on_lm_start` / `on_lm_end` | A `dspy.LM` is invoked |
| `on_adapter_format_start` / `on_adapter_format_end` | An adapter formats input |
| `on_adapter_parse_start` / `on_adapter_parse_end` | An adapter parses output |

### Example: Custom Logging for Agent Steps

```python
import dspy
from dspy.utils.callback import BaseCallback

class AgentLoggingCallback(BaseCallback):
    def on_module_end(self, call_id, outputs, exception):
        step = "Reasoning" if self._is_reasoning_output(outputs) else "Acting"
        print(f"== {step} Step ===")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        print("\n")

    def _is_reasoning_output(self, outputs):
        return any(k.startswith("Thought") for k in outputs.keys())

dspy.configure(callbacks=[AgentLoggingCallback()])
```

**Sample Output:**
```
== Reasoning Step ===
  Thought_1: I need to find the current team that Shohei Ohtani plays for in Major League Baseball.
  Action_1: Search[Shohei Ohtani current team 2023]

== Acting Step ===
  passages: ["Shohei Ohtani ..."]
...
```

> **Tip:**  
> When handling input or output data in callbacks, avoid mutating them in-place. Always create a copy before making changes to prevent unintended side effects in the program.



================================================================================



# Advanced Features: Streaming and Observability: Debugging Techniques


## Advanced Features: Streaming and Observability: Debugging Techniques

As DSPy systems become more complex, robust debugging and observability are essential for diagnosing issues and maintaining reliability. DSPy provides several tools and techniques for this purpose:

### Inspecting LLM Calls with `inspect_history`

Use `dspy.inspect_history(n=...)` to print the most recent LLM invocations, including system messages, inputs, and responses:

```python
dspy.inspect_history(n=5)
```

**Limitations:**  
- Only logs LLM calls (not retrievers, tools, or other components).
- Does not capture metadata (parameters, latency, module relationships).
- Can be hard to organize with multiple concurrent questions.

For deeper tracing and observability, consider more advanced solutions.

### Adjusting Logging Verbosity

DSPy leverages Python's standard `logging` library. To increase debugging output, set the logging level to `DEBUG`:

```python
import logging
logging.getLogger("dspy").setLevel(logging.DEBUG)
```

To reduce log output, use `WARNING` or `ERROR`:

```python
import logging
logging.getLogger("dspy").setLevel(logging.WARNING)
```

### Advanced Observability

For comprehensive tracing—including metadata and component relationships—explore MLflow Tracing and custom logging with callbacks. These approaches provide greater transparency and are recommended for diagnosing complex or production systems.



================================================================================



# Advanced Features: Assertions and Reliability: dspy.Assert and dspy.Suggest


## Advanced Features: Assertions and Reliability: `dspy.Assert` and `dspy.Suggest`

DSPy provides assertion utilities to enforce and guide computational constraints within LM pipelines:

- **`dspy.Assert(constraint: bool, msg: Optional[str] = None, backtrack: Optional[module] = None)`**  
  Enforces that `constraint` is `True`. If the assertion fails, DSPy triggers a retry or backtracking mechanism (optionally using the provided `backtrack` module), modifying the pipeline signature with previous outputs and feedback. If the constraint cannot be satisfied after `max_backtracking_attempts`, a `dspy.AssertionError` is raised and execution halts.

- **`dspy.Suggest(constraint: bool, msg: Optional[str] = None, backtrack: Optional[module] = None)`**  
  Works like `dspy.Assert`, but does not halt execution after retries. Instead, persistent failures are logged and the pipeline continues. This provides soft guidance rather than strict enforcement.

**Usage Tips:**
- Use `dspy.Assert` as a strict checker during development to ensure pipeline correctness.
- Use `dspy.Suggest` during evaluation or production for best-effort guidance without interrupting execution.

**Example:**
```python
dspy.Assert(len(output) < 100, msg="Output too long")
dspy.Suggest("keyword" in output, msg="Keyword missing from output")
```



================================================================================



# Advanced Features: Assertions and Reliability: Using Assertions in Programs


## Advanced Features: Assertions and Reliability: Using Assertions in Programs

Assertions in DSPy enable you to enforce constraints and improve the reliability of your pipelines. By inserting `dspy.Assert` or `dspy.Suggest` after model generations, you can validate outputs and trigger automatic retries if constraints are not met.

### Enforcing Constraints

Define validation functions to check conditions on model outputs. For example, to ensure a generated query is under 100 characters and distinct from previous queries:

```python
def validate_query_distinction_local(prev_queries, query):
    # Returns True if query is distinct from previous queries
    if not prev_queries:
        return True
    if dspy.evaluate.answer_exact_match_str(query, prev_queries, frac=0.8):
        return False
    return True

# Inside your module's forward method:
dspy.Suggest(
    len(query) <= 100,
    "Query should be short and less than 100 characters",
    target_module=self.generate_query
)
dspy.Suggest(
    validate_query_distinction_local(prev_queries, query),
    "Query should be distinct from: " + "; ".join(f"{i+1}) {q}" for i, q in enumerate(prev_queries)),
    target_module=self.generate_query
)
```

### Activating Assertion-Driven Reliability

To enable automatic backtracking and retries when assertions fail, activate assertions in your program using one of the following methods:

- **Wrap your module:**
  ```python
  from dspy.primitives.assertions import assert_transform_module, backtrack_handler
  robust_module = assert_transform_module(YourModule(), backtrack_handler)
  ```
- **Or call:**
  ```python
  robust_module = YourModule().activate_assertions()
  ```

When an assertion fails, DSPy modifies the prompt to include the failed output and user instruction, then retries generation, enabling robust, constraint-driven pipelines with minimal manual intervention.



================================================================================



# Advanced Features: Cost Tracking and LM Usage


## Advanced Features: Cost Tracking and LM Usage

### Tracking LLM Call Costs

DSPy allows you to monitor the cost (in USD) of language model calls, as calculated by LiteLLM for supported providers. To compute the total cost of all LLM calls made via a model instance `lm`, use:

```python
cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])
```

This sums the `cost` field from each entry in the `lm.history`, ensuring only non-`None` values are included.



================================================================================



# Ecosystem and Community: DSPy Ecosystem Overview


## Ecosystem and Community: DSPy Ecosystem Overview

DSPy is a modular, open-source framework originating from Stanford NLP (2022), designed to accelerate innovation in large language model (LM) program architectures, inference strategies, and optimizers. Its ecosystem thrives on community contributions, leading to rapid improvements and broad adoption. Notable advances include optimizers such as MIPROv2, BetterTogether, and LeReT, as well as program architectures like STORM, IReRa, and DSPy Assertions. The DSPy ecosystem continually evolves through active participation from researchers and practitioners, supporting both research and production use cases.



================================================================================



# Ecosystem and Community: Integrations and Related Tools


## Ecosystem and Community: Integrations and Related Tools

DSPy supports integration with a variety of platforms and tools, including:

- Databricks
- Langchain
- Weaviate
- Qdrant
- Weights & Biases
- Milvus
- Neo4j
- Lightning AI
- Haystack
- Arize
- LlamaIndex
- Langtrace
- Langfuse
- OpenLIT
- Relevance AI

For an up-to-date and comprehensive list of integrations, related tools, and community resources, visit the [Awesome DSPy](https://github.com/ganarajpr/awesome-dspy/tree/master) repository.



================================================================================



# Ecosystem and Community: Contributing and Community Resources


## Ecosystem and Community: Contributing and Community Resources

DSPy is an open-source project that encourages community involvement. To contribute, refer to the [contributing guide](https://github.com/stanfordnlp/dspy/blob/main/CONTRIBUTING.md) for guidelines and best practices.

For learning materials, tutorials, and community discussions, explore the following curated resources:
- [Awesome DSPy repository](https://github.com/ganarajpr/awesome-dspy/tree/master): A comprehensive list of blogs, videos, and podcasts about DSPy.
- [Weaviate DSPy directory](https://weaviate.io/developers/weaviate/more-resources/dspy): Additional DSPy resources and community links.

These resources provide up-to-date information and practical guidance for both new and advanced users.



================================================================================



# Ecosystem and Community: Notable Papers and Open-Source Projects


## Ecosystem and Community: Notable Papers and Open-Source Projects

DSPy is widely adopted in both research and open-source communities. Below are key papers and projects that showcase its capabilities:

### Notable Papers Using DSPy

- **[STORM](https://arxiv.org/abs/2402.14207):** Automated generation of Wikipedia-like articles.
- **[PATH](https://arxiv.org/abs/2406.11706):** Prompt optimization for training information retrieval (IR) models.
- **[UMD's Suicide Detection System](https://arxiv.org/abs/2406.06608):** Surpassed expert human prompt engineering for sensitive detection tasks.
- **[DSPy Guardrails](https://boxiyu.github.io/assets/pdf/DSPy_Guardrails.pdf):** Enhanced LLM safety by reducing adversarial attack success rates.

For more research and up-to-date examples, consult the official DSPy documentation.

### Open-Source Projects Using DSPy

- **[STORM](https://github.com/stanford-oval/storm):** Wikipedia-style article generation.
- **[DSPy Redteaming](https://github.com/haizelabs/dspy-redteam):** Tools for LLM safety and adversarial testing.
- **[DSPy Theory of Mind](https://github.com/plastic-labs/dspy-opentom):** Modeling reasoning about beliefs and intentions.
- **[Text2SQL Optimization](https://github.com/jjovalle99/DSPy-Text2SQL):** Improving language model performance on text-to-SQL tasks.
- **[DSPy with FastAPI and Gradio](https://github.com/diicellman/dspy-rag-fastapi), [Gradio RAG](https://github.com/diicellman/dspy-gradio-rag):** Retrieval-augmented generation (RAG) and serving demos.

For a broader collection of community projects and demos, see the [Awesome DSPy](https://github.com/ganarajpr/awesome-dspy/tree/master) list.



================================================================================



# Ecosystem and Community: Production and Real-World Use Cases


## Ecosystem and Community: Production and Real-World Use Cases

DSPy is widely adopted in production and research across diverse industries, powering applications in chatbots, code synthesis, retrieval-augmented generation (RAG), prompt optimization, legal and medical pipelines, finance, e-commerce, and more. Below are selected examples of real-world use cases:

- **JetBlue:** Multiple chatbot applications
- **Replit:** Code LLM pipelines for synthesizing diffs
- **Databricks:** LM Judges, RAG, classification, and production solutions
- **Sephora:** Agent use cases (undisclosed)
- **Zoro UK:** E-commerce and structured shopping
- **VMware:** RAG and prompt optimization
- **Haize Labs:** Automated LLM red-teaming
- **Plastic Labs:** R&D pipelines for Honcho
- **PingCAP:** Knowledge graph construction
- **Salomatic:** Medical report enrichment
- **Truelaw:** Custom legal LLM pipelines
- **STChealth:** Entity resolution with human-readable rationale
- **Moody's:** RAG optimization and agentic finance systems
- **Normal Computing:** Translating chip specs to formal languages
- **Procure.FYI:** Processing technology spending and pricing data
- **RadiantLogic:** AI data assistant for query routing and summarization
- **Raia:** AI-powered personal healthcare agents
- **Hyperlint:** Technical documentation generation
- **Starops & Saya:** Research document and article generation
- **Tessel AI:** Human-machine interaction enhancements
- **Dicer.ai:** Marketing AI for ad optimization
- **Howie:** Automated meeting scheduling via email
- **Isoform.ai:** Custom integrations
- **Trampoline AI:** Data augmentation and LM pipelines
- **Pretrain:** Automated AI performance optimization from user examples

This list is a subset of public or approved industry use cases; DSPy is used in hundreds of other applications. To contribute your use case, submit a pull request to the DSPy documentation repository.



================================================================================



# Ecosystem and Community: Team and Organization


## Ecosystem and Community: Team and Organization

DSPy is a research-driven framework developed to advance language model (LM) program abstractions, prompt optimization, and reinforcement learning (RL). The project is led by Omar Khattab (Stanford & Databricks) and mentored by experts from Stanford, UC Berkeley, CMU, and industry partners. DSPy benefits from a broad, active community of contributors and is widely used by researchers, engineers, and hobbyists in both academic and production environments.