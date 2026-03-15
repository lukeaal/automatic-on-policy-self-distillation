"""lm-eval wrapper utilities for hypothesis-applied prompt evaluation."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LMEvalResult:
    """Structured lm-eval result for one task run."""

    score: float
    metric_name: str
    task_name: str
    raw_results: dict


class HypothesisLMEvalRunner:
    """
    Run any lm-eval task with a hypothesis function applied to doc prompts.

    This uses lm-evaluation-harness external task loading by generating a temporary
    plugin task that subclasses the target task and overrides `doc_to_text`.
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY") -> None:
        self.base_url = base_url
        self.api_key = api_key

    @staticmethod
    def _plugin_source(task_name: str, hypothesis_source: str) -> str:
        wrapped_task_name = f"{task_name}_hyp"
        return f"""
from lm_eval.tasks import get_task_dict
from src.agent.optimizer import load_hypothesis_function

TARGET_TASK = {task_name!r}
WRAPPED_TASK = {wrapped_task_name!r}
_HYPOTHESIS_SOURCE = {hypothesis_source!r}
_HYP_FN = load_hypothesis_function(_HYPOTHESIS_SOURCE)

_base_task_dict = get_task_dict([TARGET_TASK])
_base_task = _base_task_dict[TARGET_TASK]
_base_task_cls = _base_task.__class__

class WrappedHypothesisTask(_base_task_cls):
    def doc_to_text(self, doc):
        base_prompt = super().doc_to_text(doc)
        transformed = _HYP_FN(base_prompt)
        if not isinstance(transformed, str):
            raise TypeError("Hypothesis function must return a string.")
        return transformed

TASK_REGISTRY = {{WRAPPED_TASK: WrappedHypothesisTask}}
"""

    @staticmethod
    def _select_score(results_json: dict, task_name: str) -> tuple[float, str]:
        task_results = results_json.get("results", {}).get(task_name, {})
        if not task_results:
            raise ValueError(f"No results found for task '{task_name}'. Got keys: {list(results_json.get('results', {}).keys())}")

        preferred_metrics = (
            "exact_match,strict-match",
            "exact_match,none",
            "exact_match",
            "acc,none",
            "acc_norm,none",
            "f1,none",
            "bleu,none",
            "rouge1,none",
        )
        for metric_name in preferred_metrics:
            metric_value = task_results.get(metric_name)
            if isinstance(metric_value, (int, float)):
                return float(metric_value), metric_name

        for metric_name, metric_value in task_results.items():
            if isinstance(metric_value, (int, float)) and "_stderr" not in metric_name:
                return float(metric_value), metric_name

        raise ValueError(f"Unable to select numeric metric from task results: {task_results}")

    def evaluate_task(
        self,
        *,
        hypothesis_source: str,
        task_name: str,
        model_id: str,
        batch_size: str = "auto",
        num_fewshot: int = 0,
        limit: int | None = None,
    ) -> LMEvalResult:
        wrapped_task_name = f"{task_name}_hyp"
        with tempfile.TemporaryDirectory(prefix="lm_eval_hyp_") as temp_dir:
            temp_path = Path(temp_dir)
            plugin_dir = temp_path / "plugin"
            plugin_dir.mkdir(parents=True, exist_ok=True)
            plugin_file = plugin_dir / "wrapped_hypothesis_task.py"
            plugin_file.write_text(
                self._plugin_source(task_name=task_name, hypothesis_source=hypothesis_source),
                encoding="utf-8",
            )

            output_path = temp_path / "results.json"
            model_args = (
                f"model={model_id},base_url={self.base_url},api_key={self.api_key}"
            )

            command = [
                "lm_eval",
                "--include_path",
                str(plugin_dir),
                "--tasks",
                wrapped_task_name,
                "--model",
                "local-completions",
                "--model_args",
                model_args,
                "--batch_size",
                batch_size,
                "--num_fewshot",
                str(num_fewshot),
                "--output_path",
                str(output_path),
            ]
            if limit is not None:
                command.extend(["--limit", str(limit)])

            process = subprocess.run(command, check=False, capture_output=True, text=True)
            if process.returncode != 0:
                raise RuntimeError(
                    "lm_eval command failed.\n"
                    f"Command: {' '.join(command)}\n\n"
                    f"stdout:\n{process.stdout}\n\n"
                    f"stderr:\n{process.stderr}"
                )

            if not output_path.exists():
                raise FileNotFoundError(
                    f"lm_eval succeeded but did not produce output at {output_path}"
                )

            results_json = json.loads(output_path.read_text(encoding="utf-8"))
            score, metric_name = self._select_score(results_json, wrapped_task_name)
            return LMEvalResult(
                score=score,
                metric_name=metric_name,
                task_name=wrapped_task_name,
                raw_results=results_json,
            )
