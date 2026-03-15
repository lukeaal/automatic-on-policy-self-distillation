"""Prompt optimizer utilities for generating and evaluating hypotheses."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

try:
    from .foundation_model import FoundationModel
    from .vllm import LMEvalRunner
except ImportError:  # allows direct script execution: uv run src/agent/optimizer.py
    from src.agent.foundation_model import FoundationModel
    from src.agent.vllm import LMEvalRunner


BASE_PROMPT = """
I need you to come up with a new translation python function which takes a string and returns the translated string.
The translation function represents a prompt optimization hypothesis for improving performance on {eval_name}.
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
Do not add or remove sections from the original prompt template used by the eval task.
Do not remove formatting constraints that the eval task depends on.
Only make minimal text edits needed to improve performance while preserving structure.

Return only the python function text with correct indentation and syntax.
""".strip()


@dataclass(frozen=True)
class EvalSpec:
    """Configuration for constructing a local eval dataset from HF data."""

    name: str
    hf_dataset_path: str
    hf_dataset_name: str | None
    hf_split: str
    input_field: str
    target_field: str
    base_prompt_template: str


def _extract_gsm8k_target(raw_target: str) -> str:
    match = re.search(r"####\s*([^\n]+)", raw_target)
    return match.group(1).strip() if match else raw_target.strip()


def _get_eval_spec(eval_name: str) -> EvalSpec:
    normalized = eval_name.strip().lower()
    if normalized == "gsm8k":
        return EvalSpec(
            name="gsm8k",
            hf_dataset_path="openai/gsm8k",
            hf_dataset_name="main",
            hf_split="test",
            input_field="question",
            target_field="answer",
            base_prompt_template="Question: {input_text}\nAnswer:",
        )
    raise ValueError(
        f"Unsupported eval '{eval_name}'. Currently supported evals: gsm8k."
    )


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
    return BASE_PROMPT.format(
        eval_name=eval_name,
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


def _sanitize_task_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()


def _build_eval_records(
    *,
    eval_spec: EvalSpec,
    hf_split_dataset: Any,
    hypothesis_function: Callable[[str], str],
    limit: int | None,
) -> list[dict[str, str]]:
    max_examples = limit if limit is not None else len(hf_split_dataset)
    records: list[dict[str, str]] = []
    for idx, row in enumerate(hf_split_dataset):
        if idx >= max_examples:
            break
        input_text = str(row[eval_spec.input_field]).strip()
        target_raw = str(row[eval_spec.target_field]).strip()
        target = _extract_gsm8k_target(target_raw)
        base_prompt = eval_spec.base_prompt_template.format(input_text=input_text)
        transformed_prompt = hypothesis_function(base_prompt)
        if not isinstance(transformed_prompt, str):
            raise TypeError("Hypothesis function must return a string.")
        records.append({"prompt": transformed_prompt, "target": target})
    if not records:
        raise ValueError(f"No records were built for eval={eval_spec.name}.")
    return records


def _write_jsonl(records: list[dict[str, str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_lm_eval_yaml(
    *,
    task_name: str,
    dataset_jsonl_path: Path,
    output_yaml_path: Path,
) -> None:
    dataset_path = dataset_jsonl_path.resolve().as_posix().replace('"', '\\"')
    yaml_text = f"""
task: {task_name}
dataset_path: json
dataset_kwargs:
  data_files:
    test: "{dataset_path}"
test_split: test
output_type: generate_until
doc_to_text: "{{{{prompt}}}}"
doc_to_target: "{{{{target}}}}"
generation_kwargs:
  do_sample: false
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
""".lstrip()
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    output_yaml_path.write_text(yaml_text, encoding="utf-8")


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

    eval_spec = _get_eval_spec(eval_name)
    cache_root = Path(".cache/opt_hyp") / eval_spec.name
    hf_cache_dir = cache_root / "hf_cache"
    hf_dataset = load_dataset(
        path=eval_spec.hf_dataset_path,
        name=eval_spec.hf_dataset_name,
        split=eval_spec.hf_split,
        cache_dir=str(hf_cache_dir),
    )

    past_hypothesis_and_results: dict[str, float] = {}
    best_hypothesis = ""
    best_score = float("-inf")
    evaluator = LMEvalRunner(base_url=base_url, api_key=api_key)

    for trial_idx in range(1, trials + 1):
        hypothesis_source = generate_hypothesis(
            foundation_model=foundation_model,
            past_hypothesis_and_results=past_hypothesis_and_results,
            eval_name=eval_name,
        )
        hypothesis_function = load_hypothesis_function(hypothesis_source)

        trial_dir = cache_root / f"trial_{trial_idx:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        dataset_jsonl = trial_dir / "eval_set.jsonl"
        eval_records = _build_eval_records(
            eval_spec=eval_spec,
            hf_split_dataset=hf_dataset,
            hypothesis_function=hypothesis_function,
            limit=limit,
        )
        _write_jsonl(eval_records, dataset_jsonl)

        task_name = f"{_sanitize_task_name(eval_spec.name)}_hyp_trial_{trial_idx:03d}"
        task_yaml = trial_dir / f"{task_name}.yaml"
        _write_lm_eval_yaml(
            task_name=task_name,
            dataset_jsonl_path=dataset_jsonl,
            output_yaml_path=task_yaml,
        )

        result = evaluator.evaluate_task(
            task_name=task_name,
            model_id=model_id,
            include_path=trial_dir,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=limit,
        )
        score = result.score
        past_hypothesis_and_results[hypothesis_source] = score
        print(
            f"[trial {trial_idx}/{trials}] task={result.task_name} "
            f"metric={result.metric_name} score={score} yaml={task_yaml}"
        )

        if score > best_score:
            best_score = score
            best_hypothesis = hypothesis_source

    return best_hypothesis, best_score, past_hypothesis_and_results

