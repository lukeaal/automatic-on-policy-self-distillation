"""Prompt optimizer utilities for generating and loading hypotheses."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from typing import Any

try:
    from .foundation_model import FoundationModel
except ImportError:  # allows direct script execution: uv run src/agent/optimizer.py
    from foundation_model import FoundationModel


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


if __name__ == "__main__":
    sample_eval_name = "gsm8k"
    past_hypothesis_and_results: dict[str, float] = {
        "def translate(text):\n    return text.strip()": 0.42
    }
    sample_input = "Translate this sentence."

    print("Running optimizer smoke test with FoundationModel...")
    model = FoundationModel()
    hypothesis_source = generate_hypothesis(
        foundation_model=model,
        past_hypothesis_and_results=past_hypothesis_and_results,
        eval_name=sample_eval_name,
    )
    print("\nGenerated hypothesis:\n")
    print(hypothesis_source)

    hypothesis_fn = load_hypothesis_function(hypothesis_source)
    output = hypothesis_fn(sample_input)
    print("\nFunction output:\n")
    print(output)

