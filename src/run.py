"""Run command handler."""

from pathlib import Path
from collections.abc import Iterator
from contextlib import contextmanager
import threading
import time

import typer

from .agent.foundation_model import FoundationModel
from .agent.student_model import StudentModel
from .vllm import VLLMModel


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


def run_command(model: str, evals: Path, trials: int) -> None:
    """Handle the CLI run command."""

    # Validate input parameters
    if not evals.exists():
        raise typer.BadParameter(f"Evals file not found: {evals}")
    if trials < 1:
        raise typer.BadParameter("trials must be >= 1")

    with spinner("Setting up foundation model"):
        foundation_model = FoundationModel()

    with spinner("Downloading student model"):
        student_model = StudentModel(student_model_id=model)
        local_model_path = student_model.setup()

    # TODO: We want to run evals in parallel so lets deploy multiple instances
    with spinner("Starting local vLLM model server"):
        vllm_model = VLLMModel()
        vllm_model.load_model_for_serving(model=str(local_model_path))

    with spinner('Generating vLLM health-check prompt "hello"'):
        healthcheck_responses = vllm_model.generate(prompts=["hello"], max_tokens=32)

    hello_response = healthcheck_responses[0].strip() if healthcheck_responses else ""
    typer.echo(
        'vLLM health-check prompt="hello" '
        f'response="{hello_response if hello_response else "[empty response]"}"'
    )

    # Run baselines

    # kickoff agent loop

    # Run self-distillation

    # Rerun evals on self-distilled model

    # Produce outputs

    typer.echo(
        "Running model="
        f"{model} local_path={local_model_path} evals={evals} trials={trials} "
        f"with foundation_model={foundation_model.model_id} and local vLLM GPU serving"
    )
