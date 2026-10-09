"""Generate the Chapter 6 optimizer notebooks.

Each notebook runs the listings printed in Chapter 6 of the book. ``LISTINGS``
holds them exactly as printed; ``make_notebook`` places each one in a code cell,
indented under ``if RUN_LIVE:`` when it calls a model API, so that a default
"Run All" is free and shows the saved result instead. ``PRINTED_OUTPUT`` holds
the output the chapter prints under its listings; each block is quoted in a
markdown cell next to the cell it belongs to, so a reader can compare.

After changing this file, regenerate the notebooks and refresh their outputs
from the repository root::

    python -m chapter06.build_optimizer_notebooks
    python -m chapter06.execute_optimizer_notebooks
    python -m chapter06.validate_notebooks
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent, indent
from typing import Any


CHAPTER_DIR = Path(__file__).resolve().parent
COMPARISON_PATH = CHAPTER_DIR / "results" / "expanded_notebooks" / "comparison.json"


# The Chapter 6 listings, exactly as printed (line breaks, comments, and
# indentation included). Do not reformat them.
LISTINGS: dict[str, str] = {
    # "Setting Up the AI Detection Module", page 135
    "load-dataset": r'''# Load the frozen Chapter 6 dataset and pair-grouped split
from chapter06.optimizer_runtime import (
    AIDetector,
    exact_match,
    hashed_ngram_embeddings,
    load_frozen_examples,
    split_summary,
)

NUM_THREADS = 1
splits = load_frozen_examples()
trainset = splits["train"]
valset = splits["validation"]
testset = splits["test"]

print(split_summary(splits))''',
    # "Setting Up the AI Detection Module", page 135
    "baseline-validation": r'''# Configure the task model and create the unoptimized program
task_lm = dspy.LM(
    "openai/gpt-5.6-luna",
    cache=False,
)

dspy.configure(lm=task_lm)
detector = AIDetector()

# Measure the baseline on validation
validation_evaluator = dspy.Evaluate(
    devset=valset,
    metric=exact_match,
    num_threads=1,
    display_progress=True,
    display_table=False,
)

validation_result = validation_evaluator(detector)''',
    # "Setting Up the AI Detection Module", page 136
    "baseline-test": r'''# The unoptimized baseline is frozen, so release the locked test once
test_evaluator = dspy.Evaluate(
    devset=testset,
    metric=exact_match,
    num_threads=1,
    display_progress=True,
    display_table=False,
)

test_result = test_evaluator(detector)''',
    # "LabeledFewShot", page 137
    "labeled-few-shot": r'''# Set up the optimizer
optimizer = dspy.teleprompt.LabeledFewShot(k=4)

# Compile with training data
optimized_detector = optimizer.compile(
        AIDetector(),
        trainset=trainset
    )''',
    # "BootstrapFewShot", page 138
    "bootstrap-few-shot": r'''# Set up the optimizer
optimizer = dspy.teleprompt.BootstrapFewShot(
        metric=exact_match,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        max_rounds=1,
    )

# Compile with training data
optimized_detector = optimizer.compile(
    AIDetector(),
    trainset=trainset,
)''',
    # "BootstrapFewShotWithRandomSearch (BootstrapRS)", page 139
    "bootstrap-random-search": r'''# Set up the optimizer
optimizer = dspy.BootstrapRS( # aliased to BootstrapFewShotWithRandomSearch
        metric=exact_match,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        max_rounds=1,
        num_candidate_programs=8,
        num_threads=NUM_THREADS,
    )

# Compile with training data
optimized_detector = optimizer.compile(
    AIDetector(),
    trainset=trainset,
    valset=valset,
)''',
    # "KNNFewShot", page 140
    "knn-few-shot": r'''# Set up the optimizer and training data
optimizer = dspy.teleprompt.KNNFewShot(
        k=4,
        metric=exact_match,
        trainset=trainset,
        vectorizer=dspy.Embedder(hashed_ngram_embeddings),

    )


optimized_detector = optimizer.compile(
    AIDetector()
)''',
    # "COPRO", page 142
    "copro": r'''# Set up the optimizer
optimizer = dspy.COPRO(
        prompt_model=teacher_lm,
        metric=exact_match,
        breadth=4,  # Number of candidates per iteration
        depth=2,    # Number of refinement iterations
        init_temperature=1.0,
        track_stats=True,
    )

# Compile with training data
optimized_detector = optimizer.compile(
    AIDetector(),
    trainset=trainset,
    eval_kwargs={
        "num_threads": NUM_THREADS,
        "display_progress": True,
        "display_table": False,
    },
)''',
    # "MIPROv2", page 143
    "miprov2": r'''optimizer = dspy.MIPROv2(
        metric=exact_match,
        prompt_model=teacher_lm,
        task_model=task_lm,
        auto="light",  # Use light preset for faster benchmarking
        num_threads=NUM_THREADS,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        track_stats=True,
    )
optimized_detector = optimizer.compile(
    AIDetector(),
    trainset=trainset,
    valset=valset,
)''',
    # "GEPA", page 145
    "gepa": r'''# Exact-match metric with feedback for GEPA
def exact_match_with_feedback(
    example,
    response,
    trace=None,
    pred_name=None,
    pred_trace=None,
):
    score = (
        1
        if example.is_ai == response.is_ai
        else 0
    )

    if pred_name:
        return dspy.Prediction(
            score=score,
            feedback=example.notes,
        )
    else:
        return score


optimizer = dspy.GEPA(
    metric=exact_match_with_feedback,
    auto="light",
    reflection_lm=teacher_lm,
    reflection_minibatch_size=3,
    num_threads=4,
    candidate_selection_strategy="pareto",
    use_merge=False,  # Quicker for this single-predictor benchmark
    track_best_outputs=True,
    track_stats=True,
    seed=42,
)

optimized_detector = optimizer.compile(
    AIDetector(),
    trainset=trainset,
    valset=valset,
)''',
    # "Custom instruction proposers", pages 148-149
    "gepa-word-limit-proposer": r'''from gepa.core.adapter import ProposalFn

class GenerateWordLimitedInstruction(dspy.Signature):
    """Given a current instruction and feedback, generate an improved
    instruction under a word limit."""
    current_instruction = dspy.InputField()
    feedback_summary = dspy.InputField()
    max_words = dspy.InputField()
    improved_instruction = dspy.OutputField()

class WordLimitProposer(ProposalFn):
    def __init__(self, max_words: int = 50):
        self.max_words = max_words
        self.improver = dspy.ChainOfThought(GenerateWordLimitedInstruction)
    def __call__(self, candidate, reflective_dataset, components_to_update):
        updated = {}
        for name in components_to_update:
            if name not in candidate or name not in reflective_dataset:
                continue
            feedback = "\n".join(
                f"Example {i+1}: {ex.get('Feedback','')}"
                for i, ex in enumerate(reflective_dataset[name]))
            updated[name] = self.improver(
                current_instruction=candidate[name],
                feedback_summary=feedback,
                max_words=self.max_words).improved_instruction
        return updated

optimizer = dspy.GEPA(
    metric=exact_match_with_feedback,
    max_full_evals=3,
    num_threads=4,
    reflection_lm=teacher_lm,
    instruction_proposer=WordLimitProposer(
        max_words=50
    ),
)''',
    # "SIMBA", page 149
    "simba": r'''# Set up the optimizer
optimizer = dspy.teleprompt.SIMBA(
        metric=exact_match,
        bsize=8,  # Mini-batch size
        num_candidates=4,  # Number of candidates per iteration
        max_steps=6,  # Number of optimization steps
        max_demos=2,
        prompt_model=teacher_lm,
        num_threads=NUM_THREADS,
    )
# Compile with training data
optimized_detector = optimizer.compile(
    AIDetector(),
    trainset=trainset,
    seed=42,
)''',
    # "BootstrapFinetune", page 151
    "bootstrap-finetune": r'''# Set up the custom optimizer
optimizer = BalancedBootstrapFinetune(
    metric=exact_match,
    train_kwargs=training_config,
    exclude_demos=True,
    num_threads=1,
    min_examples_per_class=2,
)

optimized_detector = optimizer.compile(
    detector,
    trainset=trainset,
    teacher=sol_teacher,
)''',
    # "BetterTogether", page 153
    "better-together": r'''optimizer = dspy.BetterTogether(
    metric=exact_match,
    p=prompt_optimizer,
    w=weight_optimizer,
)

optimized_detector = optimizer.compile(
    detector,
    trainset=trainset,
    teacher=sol_teacher,
    valset=valset,
    strategy="p -> w",
    seed=42,
    optimizer_compile_args={"p": {"teacher": sol_teacher}},
)''',
    # "Program Transformations with Ensemble", page 155
    "ensemble-programs": r'''# Program 1: four randomly selected labeled examples
program_labeled = dspy.LabeledFewShot(k=4).compile(
    AIDetector(),
    trainset=trainset,
)

# Program 2: successful bootstrapped traces
program_bootstrapped = dspy.BootstrapFewShot(
    metric=exact_match,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    max_rounds=1,
    max_errors=1,
).compile(
    AIDetector(),
    trainset=trainset,
)

# Program 3: validation-selected bootstrapped configuration
program_searched = dspy.BootstrapFewShotWithRandomSearch(
    metric=exact_match,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    num_candidate_programs=8,
    num_threads=1,
    max_errors=1,
).compile(
    AIDetector(),
    trainset=trainset,
    valset=valset,
)''',
    # "Program Transformations with Ensemble", pages 155-156
    "ensemble-vote": r'''def boolean_majority(predictions):
    return dspy.majority(
        predictions,
        field="is_ai",
        normalize=lambda value: str(value).lower(),
    )

programs = [
    program_labeled,
    program_bootstrapped,
    program_searched,
]

ensemble = dspy.Ensemble(reduce_fn=boolean_majority)
ensemble_program = ensemble.compile(programs)''',
    # "Program Transformations with Ensemble", page 156
    "ensemble-sampled": r'''# Randomly run 2 of the 3 programs for each query
ensemble_sampled = dspy.Ensemble(
    reduce_fn=boolean_majority,
    size=2,
)

ensemble_sampled_program = ensemble_sampled.compile(
    [
        program_labeled,
        program_bootstrapped,
        program_searched,
    ]
)''',
}


# Output the chapter prints under its listings, exactly as printed (line breaks
# included). Do not reformat it. Each entry is (page, printed lines); a block
# that continues on the next page has (first page, last page) as its page.
PRINTED_OUTPUT: dict[str, tuple[int | tuple[int, int], str]] = {
    # "Setting Up the AI Detection Module", page 135
    "split-summary": (
        135,
        r'''train=160 (human=80, AI=80); validation=60 (human=30, AI=30); test=80
(human=40, AI=40)''',
    ),
    # "Setting Up the AI Detection Module", page 135
    "baseline-validation": (
        135,
        r'''Average Metric: 33.00 / 60 (55.0%)''',
    ),
    # "Setting Up the AI Detection Module", page 136
    "baseline-test": (
        136,
        r'''Average Metric: 43.00 / 80 (53.8%)''',
    ),
    # "LabeledFewShot", page 137
    "labeled-few-shot": (
        137,
        r'''============================================================
OPTIMIZER: LabeledFewShot
============================================================
Validation Accuracy:   63.33%
Locked Test Accuracy:  67.50% (54/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       +13.75 percentage points
------------------------------------------------------------
Optimization Cost:     $0.0000
Evaluation Cost:       $0.2263
Optimization Time:     0.0s
Mean / p95 Latency:    1.621s / 2.482s
============================================================''',
    ),
    # "BootstrapFewShot", page 138
    "bootstrap-few-shot": (
        138,
        r'''============================================================
OPTIMIZER: BootstrapFewShot
============================================================
Validation Accuracy:   66.67%
Locked Test Accuracy:  67.50% (54/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       +13.75 percentage points
------------------------------------------------------------
Optimization Cost:     $0.0030
Evaluation Cost:       $0.1919
Optimization Time:     4.5s
Mean / p95 Latency:    1.647s / 2.715s
============================================================''',
    ),
    # "BootstrapFewShotWithRandomSearch (BootstrapRS)", pages 139-140
    "bootstrap-random-search": (
        (139, 140),
        r'''============================================================
OPTIMIZER: BootstrapFewShotWithRandomSearch
============================================================
Validation Accuracy:   61.67%
Locked Test Accuracy:  65.00% (52/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       +11.25 percentage points
------------------------------------------------------------
Optimization Cost:     $0.8766
Evaluation Cost:       $0.1902
Optimization Time:     1119.1s
Mean / p95 Latency:    1.579s / 2.663s
============================================================''',
    ),
    # "KNNFewShot", page 141
    "knn-few-shot": (
        141,
        r'''============================================================
OPTIMIZER: KNNFewShot
============================================================
Validation Accuracy:   71.67%
Locked Test Accuracy:  72.50% (58/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       +18.75 percentage points
------------------------------------------------------------
Optimization Cost:     $0.0000
Evaluation Cost:       $0.2380
Optimization Time:     0.0s
Mean / p95 Latency:    1.830s / 2.837s
============================================================''',
    ),
    # "COPRO", page 142
    "copro": (
        142,
        r'''============================================================
OPTIMIZER: COPRO
============================================================
Validation Accuracy: 55.0% (33/60)
Locked Test Accuracy: 56.3% (45/80)
Same-Model Baseline:  53.8% (43/80)
Accuracy Uplift:      +2.5 percentage points
------------------------------------------------------------
Total Tokens:         383,539
  - Prompt:           266,047
  - Completion:       117,492
Optimization Cost:    $0.8156
Evaluation Cost:      $0.2113
Total Recorded Cost:  $1.0269
Optimization Time:    1155.8s (19.3 minutes)
Mean / p95 Latency:   2.310s / 3.242s
============================================================''',
    ),
    # "COPRO", page 143
    "copro-instruction": (
        143,
        r'''Analyze the supplied passage for linguistic, stylistic, and structural signals
associated with AI-generated versus human-written text. Consider factors such
as formulaic phrasing, repetition, coherence patterns, specificity, natural
variation, and errors, while avoiding reliance on topic or unsupported
assumptions. Make a forced-choice classification based only on the passage.
Output exactly one label: AI-generated or Human-written.''',
    ),
    # "MIPROv2", page 144
    "miprov2": (
        144,
        r'''============================================================
OPTIMIZER: MIPROv2
============================================================
Validation Accuracy: 80.0% (48/60)
Locked Test Accuracy: 66.3% (53/80)
Same-Model Baseline:  53.8% (43/80)
Accuracy Uplift:      +12.5 percentage points
------------------------------------------------------------
Total Tokens:         565,949
  - Prompt:           471,085
  - Completion:       94,864
Optimization Cost:    $0.8548
Evaluation Cost:      $0.2138
Total Recorded Cost:  $1.0685
Optimization Time:    285.2s (4.8 minutes)
Mean / p95 Latency:   2.098s / 2.870s
============================================================''',
    ),
    # "GEPA", page 147
    "gepa": (
        147,
        r'''============================================================
OPTIMIZER: GEPA
============================================================
Validation Accuracy:   80.00% (48/60)
Locked Test Accuracy:  80.00% (64/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       +26.25 percentage points
------------------------------------------------------------
Optimization Cost:     $0.5824
Evaluation Cost:       $0.0396
Total Recorded Cost:   $0.6221
Optimization Time:     616.7s (10.3 minutes)
Mean / p95 Latency:    2.423s / 3.452s
============================================================''',
    ),
    # "GEPA", page 147
    "gepa-instruction": (
        147,
        r'''Classify whether a supplied passage is likely AI-generated or AI-rewritten
according to the stylistic calibration below.
Input:
text: the passage to classify.
Output:
reasoning: A concise explanation grounded in specific wording, sentence
structure, rhythm, transitions, repetition, and level of source-specific
texture.
is_ai: True if the passage is likely AI-generated or AI-rewritten; otherwise
False.
Return only these two fields.
Evaluation method:
Judge the passage comparatively and holistically. Do not treat polished grammar,
concision, technical terminology, or the mere presence of domain-specific names
as decisive. Instead, determine whether the details retain locally authored
irregularities or have been organized into globally smoothed, balanced,
self-contained explanatory prose.''',
    ),
    # "SIMBA", page 150
    "simba": (
        150,
        r'''============================================================
OPTIMIZER: SIMBA
============================================================
Validation Accuracy:   51.67%
Locked Test Accuracy:  47.50% (38/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       -6.25 percentage points
------------------------------------------------------------
Optimization Cost:     $1.1413
Evaluation Cost:       $0.1589
Optimization Time:     321.3s
Mean / p95 Latency:    1.711s / 2.408s
============================================================''',
    ),
    # "BootstrapFinetune", page 152
    "bootstrap-finetune": (
        152,
        r'''============================================================
OPTIMIZER: BootstrapFinetune
============================================================
Student Model:         Qwen/Qwen2.5-0.5B-Instruct
Teacher Model:         openai/gpt-5.6-sol
Validation Accuracy:   70.00%
Locked Test Accuracy:  70.00% (56/80)
Same-Model Baseline:   51.25% (41/80)
Accuracy Uplift:       +18.75 percentage points
Accepted Traces:       77 human / 63 AI
------------------------------------------------------------
Optimization Cost:     $0.8651
Evaluation Cost:       $0.0000
Optimization Time:     1026.3s
Mean / p95 Latency:    1.526s / 1.992s
============================================================''',
    ),
    # "BetterTogether", pages 153-154
    "better-together": (
        (153, 154),
        r'''============================================================
OPTIMIZER: BetterTogether
============================================================
Student Model:         Qwen/Qwen2.5-0.5B-Instruct
Teacher Model:         openai/gpt-5.6-sol
Validation Accuracy:   60.00%
Locked Test Accuracy:  65.00% (52/80)
Same-Model Baseline:   51.25% (41/80)
Accuracy Uplift:       +13.75 percentage points
Accepted Traces:       77 human / 55 AI
------------------------------------------------------------
Optimization Cost:     $0.8445
Evaluation Cost:       $0.0000
Optimization Time:     1740.0s
Mean / p95 Latency:    1.550s / 1.922s
============================================================''',
    ),
    # "Program Transformations with Ensemble", page 156
    "ensemble": (
        156,
        r'''============================================================
TRANSFORMATION: Ensemble
============================================================
Validation Accuracy:   65.00%
Locked Test Accuracy:  70.00% (56/80)
Same-Model Baseline:   53.75% (43/80)
Accuracy Uplift:       +16.25 percentage points
------------------------------------------------------------
Optimization Cost:     $0.8881
Evaluation Cost:       $0.5875
Optimization Time:     1151.2s
Mean / p95 Latency:    4.803s / 6.601s
============================================================''',
    ),
}


SKIP_PAID = (
    "Skipped: this listing calls the model API. Set CHAPTER06_RUN_LIVE=1 "
    "before starting Jupyter to run it; the saved result is shown below."
)
SKIP_FINETUNE = (
    "Skipped: this listing calls the model API and fine-tunes a local model. "
    "Set CHAPTER06_RUN_LIVE=1 before starting Jupyter to run it; the saved "
    "result is shown below."
)
SKIP_LOCAL_SETUP = (
    "Skipped: this cell sets up the local student model and the GPT-5.6-sol "
    "teacher. Set CHAPTER06_RUN_LIVE=1 before starting Jupyter to run it."
)
SKIP_DEPENDENT = (
    "Skipped: this listing builds on the live-only cells above. Set "
    "CHAPTER06_RUN_LIVE=1 before starting Jupyter to run it."
)
SHOW_LIVE_PROGRAM = (
    "predictor = optimized_detector.detect.predict\n"
    'print(f"Instruction: {predictor.signature.instructions}")\n'
    'print(f"Demonstrations: {len(predictor.demos)}\\n")'
)
SKIP_SCORING = (
    "Skipped: scoring calls the model. Set CHAPTER06_RUN_LIVE=1 before "
    "starting Jupyter to run it; the saved result is shown below."
)

LOCAL_LIVE_REQUIREMENTS = (
    "A live run needs `OPENAI_API_KEY` for the GPT-5.6-sol teacher and the local "
    "PyTorch/Transformers/TRL/PEFT stack, which is part of the repository "
    "environment. The first live run downloads the `Qwen/Qwen2.5-0.5B-Instruct` "
    "student (about 1 GB) from Hugging Face."
)
LOCAL_OUTPUT_NOTE = (
    "The fine-tuned model is written to a new folder in your system's temporary "
    'directory; `training_config["output_dir"]` holds the path.'
)
INSTRUCTION_SCORE_NOTE = (
    "The cell first prints the instruction your live run selected and the number "
    "of demonstrations it attached. Every prediction is a fresh model call: the "
    "60 validation and 80 locked-test predictions run one at a time, which takes "
    "five to ten minutes, depending on the instruction. The summary lines are "
    "printed when both passes finish."
)
LOCAL_SCORE_NOTE = (
    "The 60 validation and 80 locked-test predictions run on the fine-tuned "
    "local student, one at a time, so this cell makes no API call. It takes a "
    "few minutes on Apple Silicon, and the summary lines are printed when both "
    "passes finish."
)

PRINTED_SPLIT_NOTE = (
    "The page wraps the line after `test=80`; the cell above prints it on one line. "
    "The split is frozen, so every run prints these counts."
)
PRINTED_METRIC_NOTE = (
    "Every call is fresh and uncached, so the count in a live run can differ by a "
    "few examples."
)
PRINTED_RESULT_NOTE = (
    "These are the numbers of the saved result above; a live run is a new "
    "experiment, so its scores, cost, and timings will differ."
)
PRINTED_ROUNDED_RESULT_NOTE = (
    "These are the numbers of the saved result above, which shows the percentages "
    "to two decimal places; a live run is a new experiment, so its scores, cost, "
    "and timings will differ."
)

RECORDED_BY_RUNNER = (
    "The recorded run was produced by `run_optimizer()` in "
    "`chapter06/optimizer_runtime.py`."
)
DEFAULT_RUN_NOTE = (
    "The saved program was compiled without the locked test: the optimizer saw "
    "training and validation data only, and the locked test was scored once "
    "afterwards."
)

LOCAL_STUDENT_SETUP = '''
import tempfile

from chapter06.apple_finetune import TransformersLocalLM, make_model_spec
from chapter06.optimizer_runtime import (
    BalancedBootstrapFinetune,
    finetune_training_config,
)

# Student: Qwen2.5-0.5B-Instruct, served locally through Transformers
student_lm = TransformersLocalLM(
    model=make_model_spec("Qwen/Qwen2.5-0.5B-Instruct"),
    device=os.getenv("CHAPTER06_FINETUNE_DEVICE", "auto"),
    max_context_tokens=1024,
    max_tokens=96,
    cache=False,
)
detector = AIDetector()
detector.set_lm(student_lm)

# Teacher: the same program running on GPT-5.6-sol
teacher_lm = dspy.LM("openai/gpt-5.6-sol", cache=False)
sol_teacher = AIDetector()
sol_teacher.set_lm(teacher_lm)

# Training settings handed to DSPy's local fine-tuning provider
training_config = finetune_training_config(
    Path(tempfile.mkdtemp(prefix="chapter06-finetune-"))
)
'''.strip("\n")


NOTEBOOKS: dict[str, dict[str, Any]] = {
    "quickstart-ai-detector.ipynb": {
        "title": "Unoptimized baseline",
        "optimizer": "quickstart",
        "book_section": '"Setting Up the AI Detection Module" (pages 134–136)',
        "idea": "The control condition: run the original ChainOfThought program without changing its prompt, demonstrations, or weights.",
        "use_when": "Always measure this first. It tells you whether optimization beats the program you would otherwise deploy.",
        "changes": "Nothing. There is no optimizer and no `compile` step, so optimization time and cost are zero.",
        "config": [
            "GPT-5.6-luna is the task model, with `cache=False` so every call is fresh and its latency and cost are visible",
            "validation is scored first; the locked test is scored once afterwards",
        ],
        "models": None,
        "printed_split": ("split-summary", PRINTED_SPLIT_NOTE),
        "listings": [
            {
                "key": "baseline-validation",
                "heading": "Measure the baseline on validation",
                "intro": "Page 135. The listing configures GPT-5.6-luna as the task model, creates the unoptimized `AIDetector`, and scores it on the 60 validation examples. That is 60 model calls.",
                "guard": SKIP_PAID,
                "printed": ("baseline-validation", PRINTED_METRIC_NOTE),
            },
            {
                "key": "baseline-test",
                "heading": "Release the locked test once",
                "intro": "Page 136. The baseline has no optimizer to tune, so the 80-example locked test can be scored straight away. This result is the same-model baseline used for uplift in every other notebook.",
                "guard": SKIP_PAID,
                "printed": ("baseline-test", PRINTED_METRIC_NOTE),
            },
        ],
        "score": None,
        "reading": "The baseline score is the denominator for uplift: every Luna optimizer notebook reports its locked-test accuracy against this result.",
        "apply": "Measure the unoptimized program on your own frozen split before you compile anything. Score validation first, release the test set once, and keep that locked-test score as the same-model baseline for every optimizer you try.",
        "run_note": "The saved result is one fresh, uncached validation pass followed by one locked-test pass.",
        "recorded_note": "In live mode the two listings above print DSPy's own `Average Metric` lines. The saved result records the same two passes together with their latency and cost.",
    },
    "labeled-few-shot.ipynb": {
        "title": "LabeledFewShot",
        "optimizer": "labeled-few-shot",
        "book_section": '"LabeledFewShot" (page 137)',
        "idea": "Sample labeled training examples and attach them directly as demonstrations; no model calls are needed during compilation.",
        "use_when": "You have trustworthy labels, want the cheapest few-shot baseline, and do not need generated reasoning traces.",
        "changes": "Demonstrations only; the original instruction remains unchanged.",
        "config": [
            "`k=4` labeled demonstrations, sampled from the frozen training split",
            "no metric and no search: the examples are attached without being tested",
        ],
        "models": "task",
        "listings": [
            {
                "key": "labeled-few-shot",
                "heading": "Compile with LabeledFewShot",
                "intro": "Page 137. Compiling only samples four labeled examples, so this cell makes no model call and runs in the default mode too.",
                "guard": None,
            },
        ],
        "score": {"program": "optimized_detector"},
        "reading": "Inspect the four saved demonstrations. This method can improve in-context calibration, but the sampled examples are not selected against the validation set.",
    },
    "bootstrap-few-shot.ipynb": {
        "title": "BootstrapFewShot",
        "optimizer": "bootstrap-few-shot",
        "book_section": '"BootstrapFewShot" (pages 138–139)',
        "idea": "Run the program over training examples, keep the traces that pass the metric, and use them as demonstrations.",
        "use_when": "Labels exist but worked reasoning traces may teach the model more than labels alone.",
        "changes": "Adds demonstrations to the prompt: bootstrapped traces first, with labeled training examples filling any slot that bootstrapping leaves empty. The saved program holds two bootstrapped demonstrations. It does not search instruction text.",
        "config": [
            "`max_bootstrapped_demos=2` and `max_labeled_demos=2`",
            "one bootstrap round (`max_rounds=1`)",
            "no separate teacher model: the task model generates the traces",
        ],
        "models": "task",
        "listings": [
            {
                "key": "bootstrap-few-shot",
                "heading": "Compile with BootstrapFewShot",
                "intro": "Page 138. Bootstrapping runs the program on training examples, so compiling makes a handful of model calls.",
                "guard": SKIP_PAID,
            },
        ],
        "score": {"program": "optimized_detector"},
        "reading": "Compare its demonstrations with LabeledFewShot: bootstrapped examples include model-produced reasoning that passed the exact-match metric.",
        "recorded_note": "The recorded run also passed `max_errors=1` to `BootstrapFewShot`, which stops the compile after a single failed call. The printed listing leaves DSPy's default error limit in place; demonstrations are selected the same way.",
    },
    "bootstrap-random-search.ipynb": {
        "title": "BootstrapFewShotWithRandomSearch (BootstrapRS)",
        "optimizer": "bootstrap-random-search",
        "book_section": '"BootstrapFewShotWithRandomSearch (BootstrapRS)" (pages 139–140)',
        "idea": "Build several bootstrapped demonstration sets and choose among them using the frozen validation split.",
        "use_when": "BootstrapFewShot is promising and you can afford multiple candidate programs to reduce dependence on one sample of demonstrations.",
        "changes": "Demonstrations only, but candidate selection adds a validation-driven search loop.",
        "config": [
            "8 candidate programs (`num_candidate_programs=8`)",
            "`max_bootstrapped_demos=2` and `max_labeled_demos=2` for every candidate; the saved program holds one bootstrapped and one labeled demonstration",
            "one evaluation thread (`NUM_THREADS = 1`)",
        ],
        "models": "task",
        "listings": [
            {
                "key": "bootstrap-random-search",
                "heading": "Compile with BootstrapRS",
                "intro": "Page 139. Every candidate program is scored on the validation split, so this is the first expensive compile in the chapter: the recorded run took about 18.7 minutes. DSPy scores 11 programs in total: a program with no demonstrations, a labeled-only program, an unshuffled bootstrap, and the 8 shuffled candidates.",
                "guard": SKIP_PAID,
            },
        ],
        "score": {"program": "optimized_detector"},
        "reading": "The extra compile spend buys candidate selection, not a new instruction. Check whether the held-out gain justifies the search relative to plain BootstrapFewShot.",
        "recorded_note": "The recorded run also passed `max_errors=1`, which stops the search after a single failed call. The printed listing leaves DSPy's default error limit in place; candidates are built and selected the same way.",
    },
    "knn-few-shot.ipynb": {
        "title": "KNNFewShot",
        "optimizer": "knn-few-shot",
        "book_section": '"KNNFewShot" (pages 140–141)',
        "idea": "Retrieve the training examples nearest to each input and compile a small few-shot program for it at inference time.",
        "use_when": "Different inputs benefit from different demonstrations and a retriever over your training examples is available.",
        "changes": "The instruction stays fixed; the demonstrations are selected by similarity for each input.",
        "config": [
            "`k=4` nearest training examples for each input",
            "`hashed_ngram_embeddings` builds local 512-dimensional hashed unigram/bigram vectors, so no embedding API is called",
            "`trainset` and `vectorizer` go to the constructor; `compile` takes only the program",
        ],
        "models": "task",
        "listings": [
            {
                "key": "knn-few-shot",
                "heading": "Compile with KNNFewShot",
                "intro": "Page 140. Building the optimizer embeds the training split locally and compiling makes no model call, so this cell runs in the default mode too.",
                "guard": None,
            },
        ],
        "score": {
            "program": "optimized_detector",
            "note": "As printed, each prediction first bootstraps demonstrations on its four retrieved neighbors, so it makes about five model calls: roughly 700 calls and close to half an hour for the 60 validation and 80 locked-test predictions. DSPy prints a progress bar and a `Bootstrapped ...` line for every prediction; the summary lines come last.",
        },
        "reading": "Compile cost is nearly zero, but retrieval and demonstration selection move into inference: every prediction first looks up its four nearest training examples. The saved program therefore stores no demonstrations, which is why the preview below reports `Demonstrations: 0`; KNNFewShot chooses them again for every input.",
        "recorded_note": "The recorded 72.50% run additionally passed `max_bootstrapped_demos=0, max_labeled_demos=4`, so the four retrieved neighbors were used directly as labeled demonstrations with one model call per prediction. As printed, DSPy also bootstraps traces on the retrieved neighbors for every prediction, so a live run makes several model calls per prediction and costs more than the saved result.",
    },
    "copro.ipynb": {
        "title": "COPRO",
        "optimizer": "copro",
        "book_section": '"COPRO" (pages 141–143)',
        "idea": "Propose instruction variants, evaluate them, and iteratively refine the best candidates.",
        "use_when": "The instruction is likely the bottleneck and you want a direct, interpretable prompt search without a demonstration search.",
        "changes": "Instruction text only; the preview at the end of this notebook shows the wording the recorded run selected.",
        "config": [
            "`breadth=4` candidates per iteration and `depth=2` refinement iterations",
            "GPT-5.6-sol proposes instructions; GPT-5.6-luna runs the task",
            "candidates are scored with the exact-match metric",
        ],
        "models": "task+teacher",
        "listings": [
            {
                "key": "copro",
                "heading": "Compile with COPRO",
                "intro": "Page 142. COPRO evaluates every proposed instruction, so this compile is slow: the recorded run took about 19.3 minutes. While it runs, DSPy prints the teacher model's most recent prompt and reply after each candidate is scored; that output is expected.",
                "guard": SKIP_PAID,
            },
        ],
        "score": {
            "program": "optimized_detector",
            "prelude": SHOW_LIVE_PROGRAM,
            "note": INSTRUCTION_SCORE_NOTE,
        },
        "reading": "Read the learned instruction before the score. COPRO's result is easy to inspect because any gain must come from wording rather than from demonstrations.",
        "printed_result_note": PRINTED_ROUNDED_RESULT_NOTE,
        "printed_program": (
            "copro-instruction",
            "This is the instruction of the saved program previewed above; the saved "
            "text also marks the two labels as code with backticks. A live run writes "
            "its own instruction.",
        ),
        "recorded_note": "The recorded COPRO result was compiled with the 60-example validation split passed as COPRO's `trainset`. The printed listing passes the 160-example training split, so a live run scores 8 candidate instructions on 160 examples each, one prediction at a time: allow about 70 minutes for the compile cell, compared with the recorded 19.3 minutes, and expect a different instruction and score.",
    },
    "miprov2.ipynb": {
        "title": "MIPROv2",
        "optimizer": "miprov2",
        "book_section": '"MIPROv2" (pages 143–144)',
        "idea": "Jointly propose instructions and demonstrations, then use validation feedback to search their combinations.",
        "use_when": "Both prompt wording and examples may matter and you can spend more compile calls on a joint search.",
        "changes": "Instruction plus up to two bootstrapped and two labeled demonstrations.",
        "config": [
            '`auto="light"` search budget',
            "up to two bootstrapped and two labeled demonstrations",
            "GPT-5.6-sol proposes instructions; GPT-5.6-luna runs the task",
        ],
        "models": "task+teacher",
        "listings": [
            {
                "key": "miprov2",
                "heading": "Compile with MIPROv2",
                "intro": "Page 143. The light preset still runs a full instruction and demonstration search against the validation split.",
                "guard": SKIP_PAID,
            },
        ],
        "score": {
            "program": "optimized_detector",
            "prelude": SHOW_LIVE_PROGRAM,
            "note": INSTRUCTION_SCORE_NOTE,
        },
        "reading": "MIPROv2 searches instructions and demonstration sets together, so read the two as one result. The saved program pairs a rewritten instruction with no demonstrations: the combination the search selected on validation uses none.",
        "printed_result_note": PRINTED_ROUNDED_RESULT_NOTE,
        "recorded_note": "The recorded run fixed `seed=42` and left the evaluation thread count at DSPy's default. The printed listing uses DSPy's default seed and `NUM_THREADS = 1`, so a live run evaluates sequentially, takes roughly 20 minutes to compile instead of the recorded 285 seconds, and can select a different program.",
    },
    "gepa.ipynb": {
        "title": "GEPA",
        "optimizer": "gepa",
        "book_section": '"GEPA" (pages 144–148) and "Custom instruction proposers" (pages 148–149)',
        "idea": "Use textual feedback on failures to evolve the instruction, keeping a Pareto frontier of the best candidates along the way.",
        "use_when": "Your metric can explain errors, not merely score them, and you want a detailed instruction that encodes those lessons.",
        "changes": "Instruction text; the recorded run selected a long instruction with no demonstrations.",
        "config": [
            '`auto="light"` search budget',
            "reflection minibatches of three examples",
            "Pareto candidate selection, with merging disabled for this single-predictor program",
            "the metric returns a score and, while GEPA is reflecting, textual feedback",
            "`seed=42`",
        ],
        "models": "task+teacher",
        "listings": [
            {
                "key": "gepa",
                "heading": "Compile with GEPA",
                "intro": "Page 145. The listing defines the feedback metric, builds the optimizer, and compiles. The metric passes each example's `notes` to GEPA as feedback whenever `pred_name` is set.",
                "guard": SKIP_PAID,
            },
        ],
        "score": {
            "program": "optimized_detector",
            "prelude": SHOW_LIVE_PROGRAM,
            "note": INSTRUCTION_SCORE_NOTE,
        },
        "reading": "The learned instruction is long, so the preview below is truncated. Open the prompt snapshot listed above for the complete text, and the `optimizer_trace/` folder next to it for the candidates GEPA tried.",
        "printed_program": (
            "gepa-instruction",
            "The page prints the beginning of the saved instruction previewed above, "
            "without the list markers, backticks, and blank lines of the saved text. A "
            "live run writes its own instruction.",
        ),
        "run_note": 'The saved result is one `auto="light"` compile with `seed=42`, started from the unoptimized detector, followed by one fresh, uncached validation pass and one locked-test pass.',
        "saved_by": "The recorded run was produced by the runner in `chapter06/experiments/gepa_expanded/` and is walked through in `gepa-expanded-dataset-experiment.ipynb`.",
        "recorded_note": "The saved GEPA program was compiled with the optimizer settings printed in the listing, but its metric returned a fixed feedback sentence for each outcome instead of `example.notes`. The `notes` column of this dataset holds one short provenance sentence per passage, so a live run of the listing produces a different instruction and score.",
        "extras": [
            {
                "key": "gepa-word-limit-proposer",
                "heading": "Custom instruction proposers",
                "intro": "Pages 148–149. The listing defines a proposer that asks the reflection model to keep each proposed instruction within a 50-word budget, then builds a GEPA optimizer that uses it. It was not used for the benchmark results in this chapter. The listing stops once the optimizer is built; to try it, compile and evaluate it as a separate program.\n\nThe listing reuses `exact_match_with_feedback` and `teacher_lm` from the cells above, so it sits under the same `if RUN_LIVE:` guard. Building the optimizer makes no model call.",
                "guard": SKIP_DEPENDENT,
            },
        ],
    },
    "simba.ipynb": {
        "title": "SIMBA",
        "optimizer": "simba",
        "book_section": '"SIMBA" (pages 149–150)',
        "idea": "Sample mini-batches, identify the examples the program finds hardest, and add reflective rules or demonstrations in a sequence of improvement steps.",
        "use_when": "You want iterative, example-driven improvement and can tolerate a relatively expensive reflective search.",
        "changes": "SIMBA may add rules to the instruction or attach demonstrations. The saved program keeps the original instruction and stores no demonstrations; it differs from the baseline program only in its sampling settings.",
        "config": [
            "mini-batches of eight examples (`bsize=8`)",
            "four candidates per step and six optimization steps",
            "at most two demonstrations",
            "GPT-5.6-sol writes the proposed rules; `seed=42` at compile time",
        ],
        "models": "task+teacher",
        "listings": [
            {
                "key": "simba",
                "heading": "Compile with SIMBA",
                "intro": "Page 149. SIMBA does not take a validation set; it samples mini-batches from the training split.",
                "guard": SKIP_PAID,
            },
        ],
        "score": {
            "program": "optimized_detector",
            "prelude": SHOW_LIVE_PROGRAM,
            "note": INSTRUCTION_SCORE_NOTE,
        },
        "reading": "The saved program keeps the original instruction and has no demonstrations, and it scored below the unoptimized baseline on both the validation split and the locked test: optimization is an experiment, not a guarantee of improvement.",
        "recorded_note": "The recorded run did not pass `num_threads`, so DSPy's default thread count applied. With `NUM_THREADS = 1` as printed, a live run evaluates sequentially: allow about 50 minutes for the compile cell, compared with the recorded 321 seconds.",
    },
    "ensemble.ipynb": {
        "title": "Ensemble",
        "optimizer": "ensemble",
        "book_section": '"Program Transformations with Ensemble" (pages 154–157)',
        "idea": "Compile three different few-shot programs and reduce their boolean predictions with majority voting.",
        "use_when": "Errors are costly enough to justify several model calls per prediction and the component programs make different mistakes.",
        "changes": "Nothing is learned: Ensemble combines three compiled few-shot programs into one program and votes on their predictions.",
        "config": [
            "three component programs: LabeledFewShot, BootstrapFewShot, and BootstrapFewShotWithRandomSearch",
            "a majority-vote reducer over the boolean `is_ai` field",
            "all three components run for every example",
        ],
        "models": "task",
        "listings": [
            {
                "key": "ensemble-programs",
                "heading": "Compile three programs",
                "intro": "Page 155. Each program gets its demonstrations a different way. Program 3 is the expensive step: it searches eight candidate programs against the validation split.",
                "guard": SKIP_PAID,
            },
            {
                "key": "ensemble-vote",
                "heading": "Vote with a boolean reducer",
                "intro": "Pages 155–156. The reducer votes on the `is_ai` field, and `dspy.Ensemble` wraps the three programs into one. Building the ensemble makes no model call, but it needs the three programs compiled above, so it sits under the same `if RUN_LIVE:` guard.",
                "guard": SKIP_DEPENDENT,
            },
        ],
        "score": {
            "program": "ensemble_program",
            "note": "Each prediction runs all three component programs, so the 60 validation and 80 locked-test predictions make 420 model calls and take roughly a quarter of an hour. The summary lines are printed when both passes finish.",
        },
        "reading": "Accuracy is only half the result: compare mean and p95 latency with the single-program optimizers, because every prediction fans out to three model calls.\n\nThe program snapshot holds only the first component. When DSPy saves a parent program it skips sub-programs that are already marked as compiled, so the BootstrapFewShot and BootstrapFewShotWithRandomSearch components of the recorded ensemble are not in the file. The preview below therefore shows the four labeled demonstrations of the LabeledFewShot component; the scores above come from all three components voting.",
        "extras": [
            {
                "key": "ensemble-sampled",
                "heading": "Optional: a sampled ensemble",
                "intro": "Page 156. With `size=2`, each query runs two of the three programs, chosen at random, which reduces model calls at the cost of greater variance. This variation was not used for the saved benchmark result.\n\nThe listing reuses `boolean_majority` and the three programs from the cells above, so it sits under the same `if RUN_LIVE:` guard. Building the sampled ensemble makes no model call.",
                "guard": SKIP_DEPENDENT,
            },
        ],
    },
    "bootstrap-finetune.ipynb": {
        "title": "BootstrapFinetune (Apple Silicon / MPS)",
        "optimizer": "bootstrap-finetune",
        "book_section": '"BootstrapFinetune" (pages 151–153)',
        "idea": "Bootstrap successful teacher traces into training data, then update model weights rather than only prompt state.",
        "use_when": "You control a trainable model and want to distill accepted DSPy traces into a reusable local adapter.",
        "changes": "A PEFT LoRA adapter for Qwen2.5-0.5B-Instruct; the prompt remains separately inspectable.",
        "config": [
            "`BalancedBootstrapFinetune`, a small subclass of `dspy.BootstrapFinetune` that refuses to train unless at least two accepted traces exist for each class",
            "GPT-5.6-sol teacher; Qwen2.5-0.5B-Instruct student trained locally",
            "10 epochs, batch size 1, gradient accumulation 4, PEFT LoRA, learning rate 2e-4",
            "`MacLocalProvider`, a thin `LocalProvider` subclass, serves the student with Transformers on MPS and adapts one TRL argument name",
        ],
        "models": "local",
        "platform_note": """
        ## LocalProvider on Apple Silicon and with standard SGLang

        DSPy 3.3.0's `LocalProvider` has two separate jobs. Its **training** path
        already uses Transformers and TRL and selects CUDA, then MPS, then CPU.
        We keep that path: DSPy formats the accepted traces, masks non-assistant
        tokens, constructs the PEFT trainer, trains, and saves the merged model.
        This notebook passes the device explicitly: MPS when it is available,
        otherwise the CPU (set `CHAPTER06_FINETUNE_DEVICE` to `mps` or `cpu` to
        choose).

        Its standard **serving** path starts `python -m sglang.launch_server`.
        That is the path to use on an SGLang-compatible CUDA system:

        ```python
        from dspy.clients.lm_local import LocalProvider

        student_lm = dspy.LM(
            "openai/local:Qwen/Qwen2.5-0.5B-Instruct",
            provider=LocalProvider(),
            cache=False,
            max_tokens=96,
        )
        ```

        Apple Silicon needs a different serving backend. This chapter's
        `MacLocalProvider(LocalProvider)` overrides only `launch` and `kill` so
        the model is loaded by Transformers on MPS. Its `finetune` method calls
        `LocalProvider.finetune` unchanged except for translating DSPy 3.3.0's
        `max_seq_length` keyword to the `max_length` name used by the pinned TRL
        0.24.0. There is no replacement training loop. The complete small shim is
        in `chapter06/apple_finetune.py`; the optimizer call below is identical on
        both platforms.
        """,
        "local_setup_note": 'The chapter calls `BalancedBootstrapFinetune` "the custom optimizer" and uses `training_config`, `sol_teacher`, and a `detector` bound to the local student without printing their definitions. The cell below sets them up so the listing that follows runs as printed: the student is `Qwen/Qwen2.5-0.5B-Instruct` served locally, and `sol_teacher` is the same program running on GPT-5.6-sol. The cell imports PyTorch and the local fine-tuning code, so it sits under the `if RUN_LIVE:` guard.',
        "local_setup": LOCAL_STUDENT_SETUP
        + '''
dspy.configure(lm=student_lm)

print(f"student model: {student_lm.model}")
print(f"teacher model: {teacher_lm.model}")''',
        "listings": [
            {
                "key": "bootstrap-finetune",
                "heading": "Compile with BootstrapFinetune",
                "intro": f"Page 151. GPT-5.6-sol generates a trace for every training example, the traces that pass the metric become fine-tuning data, and the Qwen student is trained locally. The recorded run took about 17 minutes on Apple Silicon.\n\n{LOCAL_LIVE_REQUIREMENTS} {LOCAL_OUTPUT_NOTE}",
                "guard": SKIP_FINETUNE,
            },
        ],
        "score": {
            "program": "optimized_detector",
            "prelude": 'print(f"Accepted traces: {optimizer.accepted_label_counts}")',
            "note": LOCAL_SCORE_NOTE,
        },
        "reading": "Check the accepted human/AI trace counts before the score. A balanced accepted set is a prerequisite for reading this fine-tune as a classification experiment.",
    },
    "better-together.ipynb": {
        "title": "BetterTogether (Apple Silicon / MPS)",
        "optimizer": "better-together",
        "book_section": '"BetterTogether" (pages 153–154)',
        "idea": "Run prompt optimization and weight optimization in sequence, evaluate the intermediate programs, and keep the best one.",
        "use_when": "You have both a useful prompt optimizer and a trainable local model, and want to test whether combining them outperforms either technique alone.",
        "changes": "Prompt demonstrations first, then a Qwen LoRA adapter trained on the accepted traces.",
        "config": [
            "BootstrapFewShotWithRandomSearch is the prompt optimizer (`p`); BalancedBootstrapFinetune is the weight optimizer (`w`)",
            'explicit `"p -> w"` strategy with `seed=42`',
            "Qwen2.5-0.5B-Instruct student served locally on MPS; GPT-5.6-sol teacher",
            "10 weight-training epochs",
        ],
        "models": "local",
        "local_setup_note": "The chapter uses `prompt_optimizer`, `weight_optimizer`, `sol_teacher`, and a `detector` bound to the local student without printing their definitions; the cell below sets them up so the listing that follows runs as printed. The recorded run passed `num_threads=1` and a `max_errors` limit directly to `compile`; here the same two values are set with `dspy.configure`, so the printed call behaves like the recorded run. The cell imports PyTorch and the local fine-tuning code, so it sits under the `if RUN_LIVE:` guard.",
        "local_setup": LOCAL_STUDENT_SETUP
        + '''

# A malformed answer from the small local model counts as an error, so allow
# one per example and evaluate on a single thread.
max_errors = len(trainset) + len(valset)
dspy.configure(lm=student_lm, num_threads=1, max_errors=max_errors)

# p: select demonstrations with BootstrapFewShotWithRandomSearch
prompt_optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=exact_match,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    num_candidate_programs=4,
    num_threads=1,
    max_errors=max_errors,
)

# w: fine-tune the local student on the accepted traces
weight_optimizer = BalancedBootstrapFinetune(
    metric=exact_match,
    train_kwargs=training_config,
    exclude_demos=False,
    num_threads=1,
    min_examples_per_class=2,
)

print(f"student model: {student_lm.model}")
print(f"teacher model: {teacher_lm.model}")''',
        "listings": [
            {
                "key": "better-together",
                "heading": "Compile with BetterTogether",
                "intro": f"Page 153. The `p -> w` strategy first selects demonstrations, then fine-tunes the Qwen student on GPT-5.6-sol's accepted traces. The recorded run took about 29 minutes on Apple Silicon.\n\n{LOCAL_LIVE_REQUIREMENTS} {LOCAL_OUTPUT_NOTE}",
                "guard": SKIP_FINETUNE,
            },
        ],
        "score": {
            "program": "optimized_detector",
            "prelude": 'if getattr(optimized_detector, "flag_compilation_error_occurred", False):\n    raise RuntimeError("BetterTogether stopped before completing p -> w")\nprint(f"Accepted traces: {weight_optimizer.accepted_label_counts}")',
            "note": LOCAL_SCORE_NOTE,
        },
        "reading": "Read the validation-selected stage alongside the same-model baseline. If candidates tie or regress, keeping an earlier stage is a legitimate optimizer outcome, not a reason to consult the locked test.",
    },
}

# The printed listing(s) each notebook compiles with, joined in book order.
for _spec in NOTEBOOKS.values():
    _spec["compile"] = "\n\n".join(
        LISTINGS[item["key"]] for item in _spec["listings"]
    )


def _text(value: str) -> str:
    return dedent(value).strip("\n")


def _cell_source(value: str) -> list[str]:
    return value.strip("\n").splitlines(keepends=True)


def _markdown(value: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _cell_source(value)}


def _code(value: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _cell_source(value),
    }


def _skip_branch(message: str) -> str:
    words = message.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if len(candidate) > 66 and current:
            lines.append(current + " ")
            current = word
        else:
            current = candidate
    lines.append(current)
    quoted = "\n".join(f'        "{line}"' for line in lines)
    return f"else:\n    print(\n{quoted}\n    )"


def _guarded(body: str, skip_message: str) -> str:
    """Indent ``body`` unchanged under the live guard, with a skip message."""

    return f"if RUN_LIVE:\n{indent(body, '    ')}\n{_skip_branch(skip_message)}"


def _printed_output_cell(key: str, note: str) -> dict[str, Any]:
    """Quote output the chapter prints, so a reader can compare it with a run."""

    page, text = PRINTED_OUTPUT[key]
    where = f"page {page}" if isinstance(page, int) else f"pages {page[0]}–{page[1]}"
    return _markdown(
        f"**Output printed in the book ({where}):**\n\n"
        f"```text\n{text}\n```\n\n{note}"
    )


def _listing_cells(item: dict[str, Any], *, level: str = "##") -> list[dict[str, Any]]:
    listing = LISTINGS[item["key"]]
    source = _guarded(listing, item["guard"]) if item.get("guard") else listing
    cells = [
        _markdown(f"{level} {item['heading']}\n\n{item['intro']}"),
        _code(source),
    ]
    if item.get("printed"):
        cells.append(_printed_output_cell(*item["printed"]))
    return cells


def _with_cell_ids(notebook: dict[str, Any]) -> dict[str, Any]:
    for index, cell in enumerate(notebook["cells"], start=1):
        cell["id"] = f"chapter06-cell-{index:02d}"
    return notebook


def _keys_and_cost(spec: dict[str, Any]) -> str:
    """Say what a notebook needs and what its recorded run cost.

    The figures are read from the saved comparison, so they are the ones the
    notebook displays under "Saved result".
    """

    summary = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    row = next(row for row in summary["rows"] if row["optimizer"] == spec["optimizer"])
    optimization = row["optimization_cost_usd"]
    evaluation = row["evaluation_cost_usd"]
    seconds = row["optimization_time_seconds"]
    duration = f"{seconds:.1f} seconds"
    if seconds >= 60:
        duration += f" ({seconds / 60:.1f} minutes)"
    if spec.get("models") == "local":
        live = (
            "A live run (`CHAPTER06_RUN_LIVE=1`) needs `OPENAI_API_KEY` for the "
            "GPT-5.6-sol teacher and trains the Qwen student on your machine."
        )
        recorded = (
            f"The recorded run cost ${optimization:.4f} in teacher calls and took "
            f"{duration} to compile; the student is trained and evaluated locally "
            "at no API cost."
        )
    else:
        live = (
            "A live run (`CHAPTER06_RUN_LIVE=1`) makes paid model calls and needs "
            "`OPENAI_API_KEY`."
        )
        if spec["optimizer"] == "quickstart":
            recorded = f"The recorded baseline evaluation cost ${evaluation:.4f}."
        elif optimization == 0:
            recorded = (
                "Compiling made no model call in the recorded run; evaluating the "
                f"compiled program cost ${evaluation:.4f}."
            )
        else:
            recorded = (
                f"The recorded run cost ${optimization:.4f} to optimize, which took "
                f"{duration}, and ${evaluation:.4f} to evaluate."
            )
    return (
        "**Keys and cost:** The default **Run All** is free: it needs no API key, "
        "makes no model call, and shows the saved result of the recorded run. "
        f"{live} {recorded} A live run is a new experiment, so its score, cost, "
        "and time will differ."
    )


def _intro_cell(spec: dict[str, Any]) -> dict[str, Any]:
    config = "\n".join(f"- {item}" for item in spec["config"])
    return _markdown(
        "\n\n".join(
            [
                f"# {spec['title']}",
                f"Companion notebook for Chapter 6, {spec['book_section']}.",
                spec["idea"],
                f"**Use it when:** {spec['use_when']}",
                f"**What compilation changes:** {spec['changes']}",
                f"Important configuration in this benchmark:\n\n{config}",
                _text(
                    """
                    Every notebook in this folder loads the chapter's 300-passage benchmark
                    and its frozen, pair-grouped split: 160 training, 60 validation, and 80
                    locked-test examples. Both members of a human/AI pair stay in the same
                    partition. Optimizer choices use validation only; the locked test is
                    scored once, after the program is frozen. These scores compare optimizer
                    behavior; they are not a general AI-detector leaderboard.
                    """
                ),
                _keys_and_cost(spec),
            ]
        )
    )


def _setup_cell(spec: dict[str, Any]) -> dict[str, Any]:
    runtime_imports = "format_result, published_result"
    if spec.get("score"):
        runtime_imports += ", score_summary"
    return _code(
        _text(
            f"""
            import os
            import sys
            from pathlib import Path

            import dspy
            from dotenv import load_dotenv

            cwd = Path.cwd().resolve()
            REPO_ROOT = cwd if (cwd / "chapter06").is_dir() else cwd.parent
            if not (REPO_ROOT / "chapter06" / "results" / "expanded_notebooks" / "comparison.json").exists():
                raise RuntimeError("Run this notebook from the repository root or the chapter06 directory.")
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            load_dotenv(REPO_ROOT / ".env")

            from chapter06.notebook_support import artifact_paths, learned_program_preview, verify_prompt_artifact
            from chapter06.optimizer_runtime import {runtime_imports}

            OPTIMIZER = {spec["optimizer"]!r}
            RUN_LIVE = os.getenv("CHAPTER06_RUN_LIVE", "0") == "1"
            if RUN_LIVE and not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "CHAPTER06_RUN_LIVE=1 needs OPENAI_API_KEY. Add it to the .env file "
                    "in the repository root and restart the kernel."
                )
            print(f"optimizer={{OPTIMIZER!r}}; live={{RUN_LIVE}}")
            """
        )
    )


def _dataset_cells(spec: dict[str, Any]) -> list[dict[str, Any]]:
    cells = [
        _markdown(
            _text(
                """
                ## Load the frozen dataset

                The next cell is the setup listing from Chapter 6, "Setting Up the AI Detection
                Module" (page 135). Every notebook in this folder starts with it.

                Cells that call a model API hold the chapter listing unchanged, indented under
                `if RUN_LIVE:`. By default they are skipped, so **Run All** is free and works
                offline, and the saved result of the recorded run is displayed instead. To run
                the listings yourself, set `CHAPTER06_RUN_LIVE=1` before starting Jupyter.
                Every live run needs `OPENAI_API_KEY` in the repository `.env` file; if the
                key is missing, the setup cell above stops with a message.
                """
            )
        ),
        _code(LISTINGS["load-dataset"]),
    ]
    if spec.get("printed_split"):
        cells.append(_printed_output_cell(*spec["printed_split"]))
    return cells


def _model_cells(spec: dict[str, Any]) -> list[dict[str, Any]]:
    models = spec.get("models")
    if models == "task":
        return [
            _markdown(
                "> **Book vs. notebook:** The chapter creates `task_lm` and configures DSPy in "
                "its baseline listing (page 135). The cell below repeats those two steps so "
                "this notebook runs on its own. Creating the model object makes no API call."
            ),
            _code(
                _text(
                    """
                    task_lm = dspy.LM("openai/gpt-5.6-luna", cache=False)
                    dspy.configure(lm=task_lm)
                    print(f"task model: {task_lm.model}")
                    """
                )
            ),
        ]
    if models == "task+teacher":
        return [
            _markdown(
                "> **Book vs. notebook:** The chapter creates `task_lm` in its baseline listing "
                "(page 135) and uses `teacher_lm` without printing its definition. The cell "
                "below defines both so the listing that follows runs as printed: GPT-5.6-luna "
                "runs the task and GPT-5.6-sol is the teacher (page 157). Creating the model "
                "objects makes no API call."
            ),
            _code(
                _text(
                    """
                    task_lm = dspy.LM("openai/gpt-5.6-luna", cache=False)
                    teacher_lm = dspy.LM("openai/gpt-5.6-sol", cache=False)
                    dspy.configure(lm=task_lm)
                    print(f"task model: {task_lm.model}")
                    print(f"teacher model: {teacher_lm.model}")
                    """
                )
            ),
        ]
    if models == "local":
        cells: list[dict[str, Any]] = []
        if spec.get("platform_note"):
            cells.append(_markdown(_text(spec["platform_note"])))
        cells.append(_markdown(f"> **Book vs. notebook:** {spec['local_setup_note']}"))
        cells.append(_code(_guarded(spec["local_setup"], SKIP_LOCAL_SETUP)))
        return cells
    return []


DEFAULT_SCORE_NOTE = (
    "Every prediction is a fresh model call: the 60 validation and 80 locked-test "
    "predictions run one at a time, which takes about five minutes (longer when the "
    "compiled prompt is long). The summary lines are printed when both passes finish."
)


def _score_cells(spec: dict[str, Any]) -> list[dict[str, Any]]:
    score = spec.get("score")
    if not score:
        return []
    program = score["program"]
    body = f"print(score_summary({program}, valset, testset))"
    if score.get("prelude"):
        body = f"{score['prelude']}\n{body}"
    return [
        _markdown(
            f"## Score the compiled program\n\n"
            f"In live mode the next cell scores `{program}`, the program the listing above "
            f"produced, on the validation split and then once on the locked test. "
            f"{score.get('note', DEFAULT_SCORE_NOTE)}\n\n"
            f"A call that fails, or an answer that cannot be read as true or false, counts "
            f"as a wrong prediction; the summary then ends with a `Failed Calls` line. If "
            f"the first three calls fail outright, or no validation prediction can be "
            f"scored, the cell stops and shows the first error."
        ),
        _code(_guarded(body, SKIP_SCORING)),
    ]


def _saved_result_cells(spec: dict[str, Any]) -> list[dict[str, Any]]:
    cells = [
        _markdown(
            "## Saved result\n\n"
            "The block below is the recorded run for this notebook, in the layout of the result "
            "blocks printed in the chapter. Every notebook uses the same layout, so a block can "
            "show details the chapter prints for some optimizers only: the count behind the "
            "validation percentage, token totals, the combined cost, the time in minutes, and "
            "percentages to two decimal places. It is read from "
            "`results/expanded_notebooks/comparison.json`, so it is the same whether or not you "
            f"run live. {spec.get('saved_by', RECORDED_BY_RUNNER)}"
        ),
        _code(
            _text(
                """
                print(format_result(published_result(OPTIMIZER)))
                print()
                print(artifact_paths(OPTIMIZER))
                """
            )
        ),
    ]
    # The result block the chapter prints for this optimizer, where it has one.
    if spec["optimizer"] in PRINTED_OUTPUT:
        cells.append(
            _printed_output_cell(
                spec["optimizer"],
                spec.get("printed_result_note", PRINTED_RESULT_NOTE),
            )
        )
    return cells


def _reading_cells(spec: dict[str, Any]) -> list[dict[str, Any]]:
    parts = [
        "## Read the result",
        spec["reading"],
        spec.get("run_note", DEFAULT_RUN_NOTE),
    ]
    if spec.get("recorded_note"):
        parts.append(f"> **Book vs. notebook:** {spec['recorded_note']}")
    parts.append("The next cell previews the saved program.")
    cells = [
        _markdown("\n\n".join(parts)),
        _code(
            _text(
                """
                print(learned_program_preview(OPTIMIZER))
                print()
                print("Saved program matches saved prompt:", verify_prompt_artifact(OPTIMIZER))
                """
            )
        ),
    ]
    if spec.get("printed_program"):
        cells.append(_printed_output_cell(*spec["printed_program"]))
    return cells


DEFAULT_APPLY = (
    "Adapt the listing above to your own DSPy program, metric, and frozen "
    "train/validation split. Keep the test set untouched until the optimizer returns, "
    "then report locked-test accuracy as `correct / test examples` so every optimizer "
    "is easy to compare. Use the baseline notebook (`quickstart-ai-detector.ipynb`) "
    "when you also need uplift."
)


def _closing_cell(spec: dict[str, Any]) -> dict[str, Any]:
    return _markdown(
        "\n\n".join(
            [
                "## Apply the pattern",
                spec.get("apply", DEFAULT_APPLY),
                "All Chapter 6 results are summarized in `CHAPTER_RESULTS.md`. Scores, "
                "prompts, programs, and predictions are under `results/expanded_notebooks/` "
                "(GEPA: `results/gepa_light_standard/`). Fine-tuned model weights are "
                "generated locally and are not stored in the repository. No credentials "
                "are stored.",
            ]
        )
    )


def make_notebook(spec: dict[str, Any]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = [_intro_cell(spec), _setup_cell(spec)]
    cells.extend(_dataset_cells(spec))
    cells.extend(_model_cells(spec))
    for item in spec["listings"]:
        cells.extend(_listing_cells(item))
    cells.extend(_score_cells(spec))
    cells.extend(_saved_result_cells(spec))
    cells.extend(_reading_cells(spec))
    for item in spec.get("extras", []):
        cells.extend(_listing_cells(item))
    cells.append(_closing_cell(spec))
    return _with_cell_ids(
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


def main() -> None:
    for filename, spec in NOTEBOOKS.items():
        (CHAPTER_DIR / filename).write_text(
            json.dumps(make_notebook(spec), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        f"Generated {len(NOTEBOOKS)} Chapter 6 notebooks. Run "
        "`python -m chapter06.execute_optimizer_notebooks` to refresh their saved outputs."
    )


if __name__ == "__main__":
    main()
