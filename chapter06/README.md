# Chapter 6 notebooks

Each optimizer has its own self-contained notebook. Run the install cell and then
execute the notebook top-to-bottom. `optimizer-benchmark.ipynb` is now an index so
one optimizer failure cannot cancel every other experiment.

The default configuration targets stable DSPy 3.2.1 and uses:

- `openai/gpt-5.6-luna` for task execution
- `openai/gpt-5.6-sol` for reflection, judging, and prompt proposal

Override those defaults with `TASK_MODEL` and `REFLECTION_MODEL`. Default dataset
and optimizer budgets are intentionally small. Set `TRAIN_LIMIT=0`, `VAL_LIMIT=0`,
and `EVAL_LIMIT=0` to use the full deterministic 50/25/25 split.

## Notebook map

| Notebook | Purpose |
|---|---|
| `quickstart-ai-detector.ipynb` | shared unoptimized baseline |
| `labeled-few-shot.ipynb` | LabeledFewShot |
| `bootstrap-few-shot.ipynb` | BootstrapFewShot |
| `bootstrap-random-search.ipynb` | BootstrapRS (alias for BootstrapFewShotWithRandomSearch) |
| `bootstrap-optuna.ipynb` | BootstrapFewShotWithOptuna |
| `knn-few-shot.ipynb` | KNNFewShot |
| `infer-rules.ipynb` | InferRules |
| `copro.ipynb` | COPRO |
| `miprov2.ipynb` | MIPROv2 |
| `gepa.ipynb` | GEPA and the chapter's custom WordLimitProposer |
| `simba.ipynb` | SIMBA |
| `ensemble.ipynb` | Ensemble |
| `bootstrap-finetune.ipynb` | BootstrapFinetune (CUDA) |
| `better-together.ipynb` | BetterTogether (CUDA) |
| `grpo.ipynb` | Arbor GRPO (experimental multi-GPU) |
| `finetune-mac-m3.ipynb` | manual LoRA appendix for Apple Silicon |

## KNNFewShot on DSPy 3.x

`KNNFewShot` is different from the other few-shot optimizers. Supply `k`,
`trainset`, and `vectorizer` to the constructor, then compile with only the student:

```python
optimizer = dspy.KNNFewShot(
    k=4,
    trainset=trainset,
    vectorizer=vectorizer,
    metric=exact_match,
    max_bootstrapped_demos=2,
    max_labeled_demos=0,
)
optimized_detector = optimizer.compile(detector)
```

Do not pass `num_threads` to `KNNFewShot`. DSPy forwards unknown constructor
arguments to `BootstrapFewShot`, which does not accept it; that is the source of
`KNNFewShot.__init__() got an unexpected keyword argument 'num_threads'`-style
evaluation failures.
