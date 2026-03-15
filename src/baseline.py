"""Baseline evaluation helpers powered by lm-eval."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import lm_eval

from .vllm import VLLMModel


@dataclass(frozen=True)
class BaselineResult:
    """Minimal summary for an lm-eval run."""

    eval_names: tuple[str, ...]
    task_results: dict[str, dict[str, Any]]


def _parse_eval_names(eval_name: str) -> list[str]:
    tasks = [task.strip() for task in eval_name.split(",") if task.strip()]
    if not tasks:
        raise ValueError("At least one eval task must be provided.")
    return tasks


def _effective_gpu_count(gpus: int | None) -> int:
    return VLLMModel._available_gpu_count() if gpus is None else gpus


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    return json.dumps(value, ensure_ascii=True, default=str)


def run_baseline(
    model_path: Path,
    eval_name: str,
    gpus: int | None = None,
) -> BaselineResult:
    """Run one or more lm-eval tasks against a local model path."""
    task_names = _parse_eval_names(eval_name)
    model_args = ",".join(
        [
            f"pretrained={model_path}",
            f"tensor_parallel_size={_effective_gpu_count(gpus)}",
            "gpu_memory_utilization=0.9",
            "dtype=auto",
        ]
    )
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=task_names,
        batch_size="auto",
        num_fewshot=None,
    )
    return BaselineResult(
        eval_names=tuple(task_names),
        task_results=results["results"],
    )


def write_baseline_json(
    result: BaselineResult,
    eval_name: str,
    output_path: Path = Path("baseline.json"),
) -> Path:
    """Persist final baseline metrics to a JSON artifact."""
    metrics: dict[str, Any]
    if len(result.eval_names) == 1:
        metrics = result.task_results.get(result.eval_names[0], {})
    else:
        metrics = result.task_results

    output_path.write_text(
        json.dumps(
            {
                "name": eval_name,
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def format_baseline_result(result: BaselineResult) -> str:
    """Format lm-eval output for CLI display."""
    output = ["Baseline results:"]

    for task_name, metrics in sorted(result.task_results.items()):
        output.append(f"Task: {task_name}")
        if not metrics:
            output.append("[no metrics reported]")
            continue

        for metric_name, metric_value in sorted(metrics.items()):
            output.append(f"{metric_name}={_format_metric_value(metric_value)}")

    return "\n\n".join(output)
