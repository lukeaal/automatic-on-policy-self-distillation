# automatic-on-policy-self-distillation
will come up with better name later

(inside source)

agent; prompt optimization harness (on the evals)

data; setting up eval data

eval; running baselines (vllm and llm eval)

models; manages the models and fronteir models apis

self-distillation; runs self-distillation, https://arxiv.org/abs/2603.05433

viz; vibe coded matplot lib or other wise imaging

## Usage

Run from the repository root with `uv`:

```bash
uv run asd --help
uv run asd help
uv run asd run --model <model> --evals <evals-filepath> --trials <trials>
```

`--model` must be a Hugging Face model identifier (for example, `distilbert-base-uncased` or `meta-llama/Llama-3.2-1B`).