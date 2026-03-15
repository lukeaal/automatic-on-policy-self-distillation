"""Prompt optimizer utilities for generating and loading hypotheses."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from typing import Any

try:
    from .foundation_model import FoundationModel
    from .vllm import HypothesisLMEvalRunner
except ImportError:  # allows direct script execution: uv run src/agent/optimizer.py
    from src.agent.foundation_model import FoundationModel
    from src.agent.vllm import HypothesisLMEvalRunner


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
    """
    Generate hypotheses in a loop and track a placeholder score.

    This is a scaffold loop for optimizer development before eval integration.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")

    past_hypothesis_and_results: dict[str, float] = {}
    best_hypothesis = ""
    best_score = float("-inf")

    evaluator = HypothesisLMEvalRunner(base_url=base_url, api_key=api_key)

    for trial_idx in range(1, trials + 1):
        hypothesis_source = generate_hypothesis(
            foundation_model=foundation_model,
            past_hypothesis_and_results=past_hypothesis_and_results,
            eval_name=eval_name,
        )

        result = evaluator.evaluate_task(
            hypothesis_source=hypothesis_source,
            task_name=eval_name,
            model_id=model_id,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=limit,
        )
        score = result.score
        past_hypothesis_and_results[hypothesis_source] = score
        print(
            f"[trial {trial_idx}/{trials}] task={result.task_name} metric={result.metric_name} score={score}"
        )

        if score > best_score:
            best_score = score
            best_hypothesis = hypothesis_source

    return best_hypothesis, best_score, past_hypothesis_and_results


if __name__ == "__main__":
    sample_eval_name = "gsm8k"
    sample_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    sample_input = "Translate this sentence."
    sample_trials = 3

    print("Running optimizer hypothesis-loop smoke test...")
    model = FoundationModel()
    best_hypothesis, best_score, history = run_hypothesis_loop(
        foundation_model=model,
        trials=sample_trials,
        eval_name=sample_eval_name,
        model_id=sample_model_id,
    )
    print(f"\nGenerated {len(history)} hypotheses. Best score={best_score}\n")
    print("Best hypothesis:\n")
    print(best_hypothesis)

    hypothesis_fn = load_hypothesis_function(best_hypothesis)
    output = hypothesis_fn(sample_input)
    print("\nFunction output:\n")
    print(output)

