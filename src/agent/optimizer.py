"""Prompt optimizer utilities for generating and evaluating hypotheses."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from datasets import load_dataset

try:
    from .foundation_model import FoundationModel
except ImportError:  # allows direct script execution: uv run src/agent/optimizer.py
    from src.agent.foundation_model import FoundationModel


DATA_DIR = Path(__file__).parent / "data"
TASKS_DIR = Path(__file__).parent / "tasks"

EVAL_REGISTRY: dict[str, dict[str, str]] = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split": "test",
        "question_field": "question",
        "doc_to_text": "Question: {question}\nAnswer:",
        "example_input": (
            "Question: Janet\u2019s ducks lay 16 eggs per day. "
            "She eats three for breakfast every morning and bakes muffins "
            "for her friends every day with four. She sells the remainder "
            "at the farmers' market daily for $2 per fresh duck egg. "
            "How much in dollars does she make every day at the farmers' market?\nAnswer:"
        ),
    },
}

# fmt: off
_GSM8K_YAML_TEMPLATE = r"""task: __TASK_NAME__
dataset_path: json
dataset_kwargs:
  data_files:
    test: __DATA_PATH__
output_type: generate_until
test_split: test
doc_to_text: "{{prompt}}"
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - "\\$"
      - "(?s).*#### "
      - "\\.$"
generation_kwargs:
  until:
    - "Question:"
    - "</s>"
    - "<|im_end|>"
  do_sample: false
  temperature: 0.0
repeats: 1
num_fewshot: __NUM_FEWSHOT__
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "#### (\\-?[0-9\\.\\,]+)"
      - function: "take_first"
  - name: "flexible-extract"
    filter:
      - function: "regex"
        group_select: -1
        regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"
      - function: "take_first"
metadata:
  version: 3.0
"""
# fmt: on


BASE_PROMPT = """
I need you to come up with a new translation python function which takes a string and returns the translated string.
The translation function represents a prompt optimization hypothesis for improving performance on {eval_name}.

The input to your function is the FULL formatted prompt that will be sent to the model.
Here is an example of what the input string looks like:

{example_input}

Your function will be called on every such prompt in the eval set. It must return a modified
version of that prompt that helps a small language model answer more accurately.

I have already tried the following hypotheses with the following results:
{past_hypothesis_and_results}

Please come up with a new hypothesis that is likely to improve performance on {eval_name}.

Please return your new hypothesis as a python function that is syntactically correct and runnable.
Do not use any external libraries or packages. Assume only standard Python is available.
Do not use any comments.
Do not use any docstrings.
Do not use any type hints.
Do not use any class definitions.
Do not use any import statements.
The prompt always ends with "Answer:" — you must keep that suffix so the model knows where to respond.
You may add instructions, rephrase the question, add chain-of-thought cues, etc.
Your translation function should modify almost all non-empty strings!

define the func to be  def translate(input_text: str) -> str:
Return only the python function text with correct indentation and syntax.
""".strip()



def _format_past_hypothesis_results(past_hypothesis_and_results: Mapping[str, float]) -> str:
    if not past_hypothesis_and_results:
        return "None yet."
    lines = []
    for idx, (hypothesis, score) in enumerate(past_hypothesis_and_results.items(), start=1):
        lines.append(f"{idx}. score={score} | hypothesis={hypothesis}")
    return "\n".join(lines)


def build_hypothesis_prompt(
    past_hypothesis_and_results: Mapping[str, float], eval_name: str
) -> str:
    """Build the prompt used to request a new hypothesis from the foundation model."""
    config = EVAL_REGISTRY.get(eval_name, {})
    return BASE_PROMPT.format(
        eval_name=eval_name,
        example_input=config.get("example_input", "<no example available>"),
        past_hypothesis_and_results=_format_past_hypothesis_results(
            past_hypothesis_and_results
        ),
    )


def generate_hypothesis(
    foundation_model: FoundationModel,
    past_hypothesis_and_results: Mapping[str, float],
    eval_name: str,
    **kwargs: Any,
) -> str:
    """Generate a new hypothesis as raw python function source text."""
    prompt = build_hypothesis_prompt(past_hypothesis_and_results, eval_name)
    return foundation_model.generate(prompt, **kwargs).strip()


def load_hypothesis_function(
    hypothesis_source: str, function_name: str | None = None
) -> Callable[[str], str]:
    """
    Convert model-generated function source into a callable translation function.

    Raises:
        ValueError: If no valid function is found or loading fails.
    """
    try:
        module_ast = ast.parse(hypothesis_source)
    except SyntaxError as exc:
        raise ValueError("Generated hypothesis is not valid Python syntax.") from exc

    function_defs = [
        node for node in module_ast.body if isinstance(node, ast.FunctionDef)
    ]
    if not function_defs:
        raise ValueError("Generated hypothesis must include at least one function.")

    selected_name = function_name or function_defs[0].name
    if function_name and all(node.name != function_name for node in function_defs):
        raise ValueError(f"Function '{function_name}' not found in generated hypothesis.")

    namespace: dict[str, Any] = {}
    try:
        exec(compile(module_ast, filename="<hypothesis>", mode="exec"), namespace)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Generated hypothesis could not be executed.") from exc

    loaded = namespace.get(selected_name)
    if not callable(loaded):
        raise ValueError(
            f"Loaded object '{selected_name}' is not callable in generated hypothesis."
        )

    return loaded


def run_hypothesis_loop(
    foundation_model: FoundationModel,
    trials: int,
    eval_name: str,
    model_id: str,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    batch_size: str = "auto",
    num_fewshot: int = 0,
    limit: int | None = None,
) -> tuple[str, float, dict[str, float]]:
    """Generate hypotheses, materialize local eval data, and score with lm-eval."""
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided")

    eval_data_path = download_eval_set(eval_name)
    print(f"Eval data ready: {eval_data_path}")

    past_hypothesis_and_results: dict[str, float] = {}
    best_hypothesis = ""
    best_score = float("-inf")

    for i in range(trials):
        hypothesis_source = generate_hypothesis(
            foundation_model=foundation_model,
            past_hypothesis_and_results=past_hypothesis_and_results,
            eval_name=eval_name,
        )
        load_hypothesis_function(hypothesis_source)

        modified_path = apply_hypothesis_to_eval_set(hypothesis_source, eval_name)
        task_yaml = write_modified_task_yaml(eval_name, modified_path, num_fewshot=num_fewshot)

        # TODO: score with lm-eval against served model (placeholder for now)
        score = i
        past_hypothesis_and_results[hypothesis_source] = score
        print(f"Trial {i}: score={score}")
        print(f"  Modified data → {modified_path}")
        print(f"  Task YAML    → {task_yaml}")

        if score > best_score:
            best_score = score
            best_hypothesis = hypothesis_source

    return best_hypothesis, best_score, past_hypothesis_and_results

def download_eval_set(
    eval_name: str,
    output_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Download an eval dataset from HuggingFace and save as JSONL.

    Returns the path to the written JSONL file.  Skips the download when the
    file already exists unless *force* is ``True``.
    """
    if eval_name not in EVAL_REGISTRY:
        raise ValueError(
            f"Unsupported eval: {eval_name!r}. Supported: {list(EVAL_REGISTRY)}"
        )

    config = EVAL_REGISTRY[eval_name]
    out_dir = output_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"{eval_name}.jsonl"

    if output_file.exists() and not force:
        return output_file

    dataset = load_dataset(config["hf_path"], config["hf_name"], split=config["split"])
    with open(output_file, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    return output_file


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL eval set from disk."""
    items: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def apply_hypothesis_to_eval_set(
    hypothesis_source: str,
    eval_name: str,
    input_file: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Apply a hypothesis translate function to the full formatted prompt.

    For each item the ``doc_to_text`` template is rendered first (e.g.
    ``"Question: {question}\\nAnswer:"``), then the hypothesis ``translate``
    function is called on that string.  The result is stored in a ``prompt``
    field so the task YAML can use ``doc_to_text: "{{prompt}}"``.

    Returns the path to the modified JSONL file.
    """
    if eval_name not in EVAL_REGISTRY:
        raise ValueError(
            f"Unsupported eval: {eval_name!r}. Supported: {list(EVAL_REGISTRY)}"
        )

    config = EVAL_REGISTRY[eval_name]
    doc_to_text = config["doc_to_text"]

    in_file = input_file or (DATA_DIR / f"{eval_name}.jsonl")
    out_dir = output_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"{eval_name}_modified.jsonl"

    translate_fn = load_hypothesis_function(hypothesis_source)
    items = load_eval_set(in_file)

    for item in items:
        full_prompt = doc_to_text.format(**item)
        item["prompt"] = translate_fn(full_prompt)

    with open(output_file, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    return output_file


def write_modified_task_yaml(
    eval_name: str,
    modified_data_path: Path,
    output_dir: Path | None = None,
    num_fewshot: int = 0,
) -> Path:
    """Write an lm-eval task YAML that reads from the local modified JSONL.

    Returns the path to the written YAML file.  The parent directory can be
    passed directly to ``lm_eval.simple_evaluate`` via a ``TaskManager`` with
    ``include_path`` set to this directory.
    """
    out_dir = output_dir or TASKS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    task_name = f"{eval_name}_modified"
    output_file = out_dir / f"{task_name}.yaml"

    if eval_name == "gsm8k":
        yaml_content = (
            _GSM8K_YAML_TEMPLATE
            .replace("__TASK_NAME__", task_name)
            .replace("__DATA_PATH__", str(modified_data_path.resolve()))
            .replace("__NUM_FEWSHOT__", str(num_fewshot))
        )
    else:
        raise ValueError(
            f"No task YAML template for eval: {eval_name!r}. "
            f"Supported: {list(EVAL_REGISTRY)}"
        )

    output_file.write_text(yaml_content)
    return output_file


if __name__ == "__main__":
    frontier_model = FoundationModel(model_id="openai/gpt-5.4")

    best_hypothesis, best_score, history = run_hypothesis_loop(
        foundation_model=frontier_model,
        trials=1,
        eval_name="gsm8k",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
    )
    print(f"Best hypothesis:\n{best_hypothesis}")

    example = EVAL_REGISTRY["gsm8k"]["example_input"]
    my_fun = load_hypothesis_function(best_hypothesis)
    print(f"\nbefore:\n{example}")
    print(f"\nafter:\n{my_fun(example)}")
