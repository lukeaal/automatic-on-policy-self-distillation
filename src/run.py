"""Run command handler."""

from collections.abc import Iterator
from contextlib import contextmanager
import threading
import time

import typer

from .agent.student_model import StudentModel
from .baseline import format_baseline_result, run_baseline, write_baseline_json


@contextmanager
def spinner(step: str) -> Iterator[None]:
    """Display a spinner while a CLI step is running."""
    stop_event = threading.Event()
    frames = "|/-\\"

    def _spin() -> None:
        frame_index = 0
        while not stop_event.is_set():
            frame = frames[frame_index % len(frames)]
            typer.echo(f"\r{frame} {step}", nl=False, err=True)
            frame_index += 1
            time.sleep(0.1)
        typer.echo(f"\r✓ {step}", err=True)

    spinner_thread = threading.Thread(target=_spin, daemon=True)
    spinner_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        spinner_thread.join()


def run_command(model: str, eval_name: str, gpus: int | None = None) -> None:
    """Handle the CLI run command."""

    # Validate input parameters
    if not eval_name.strip():
        raise typer.BadParameter("eval must not be empty")
    if gpus is not None and gpus < 1:
        raise typer.BadParameter("gpus must be >= 1")

    with spinner("Downloading student model"):
        student_model = StudentModel(student_model_id=model)
        local_model_path = student_model.setup()

    with spinner(f'Running baseline eval "{eval_name}"'):
        baseline_result = run_baseline(
            model_path=local_model_path,
            eval_name=eval_name,
            gpus=gpus,
        )

    baseline_json_path = write_baseline_json(baseline_result, eval_name)
    typer.echo(format_baseline_result(baseline_result))
    typer.echo(f"Wrote baseline metrics to {baseline_json_path}")

    # kickoff agent loop

    # Run self-distillation

    # Rerun evals on self-distilled model

    # Produce outputs

    typer.echo(
        "Running model="
        f"{model} local_path={local_model_path} eval={eval_name} "
        f"gpus={gpus if gpus is not None else 'auto'} "
        "with lm-eval"
    )
