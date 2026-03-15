"""Simple lm-eval wrapper utilities for local-completions evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
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


class LMEvalRunner:
    """Run lm-eval tasks against an OpenAI-compatible local-completions endpoint."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY") -> None:
        self.base_url = base_url
        self.api_key = api_key

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
        task_name: str,
        model_id: str,
        include_path: Path | None = None,
        batch_size: str = "auto",
        num_fewshot: int = 0,
        limit: int | None = None,
    ) -> LMEvalResult:
        with tempfile.TemporaryDirectory(prefix="lm_eval_hyp_") as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "results.json"
            model_args = (
                f"model={model_id},base_url={self.base_url},api_key={self.api_key}"
            )

            command = [
                sys.executable,
                "-m",
                "lm_eval",
                "--tasks",
                task_name,
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
            if include_path is not None:
                command.extend(["--include_path", str(include_path)])
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
            score, metric_name = self._select_score(results_json, task_name)
            return LMEvalResult(
                score=score,
                metric_name=metric_name,
                task_name=task_name,
                raw_results=results_json,
            )
