"""Run command handler."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import logging
import os
import threading
import time

import typer

from .agent.student_model import StudentModel
from .baseline import format_baseline_result, run_baseline, write_baseline_json


@contextmanager
def spinner(step: str) -> Iterator[Callable[[str], None]]:
    """Display a spinner while a CLI step is running."""
    stop_event = threading.Event()
    current_step = step
    frames = "|/-\\"

    def set_step(new_step: str) -> None:
        nonlocal current_step
        current_step = new_step

    def _spin() -> None:
        frame_index = 0
        while not stop_event.is_set():
            frame = frames[frame_index % len(frames)]
            typer.echo(f"\r{frame} {current_step}", nl=False, err=True)
            frame_index += 1
            time.sleep(0.1)
        typer.echo(f"\r✓ {current_step}", err=True)

    spinner_thread = threading.Thread(target=_spin, daemon=True)
    spinner_thread.start()
    try:
        yield set_step
    finally:
        stop_event.set()
        spinner_thread.join()


class _QuietEvalOutput(io.TextIOBase):
    """Suppress verbose eval logs while detecting when task execution starts."""

    _EVAL_START_MARKERS = (
        "Running generate_until requests",
        "Running loglikelihood requests",
        "Running multiple_choice requests",
        "Running rolling requests",
        "Running evals",
    )

    def __init__(self, on_eval_start: Callable[[], None]) -> None:
        self._on_eval_start = on_eval_start
        self._buffer = ""
        self.eval_started = False

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        parts = (self._buffer + text.replace("\r", "\n")).split("\n")
        self._buffer = parts.pop()
        for line in parts:
            self._handle_line(line.strip())
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._handle_line(self._buffer.strip())
            self._buffer = ""

    def _handle_line(self, line: str) -> None:
        if not line or self.eval_started:
            return
        if "%|" in line or any(marker in line for marker in self._EVAL_START_MARKERS):
            self.eval_started = True
            self._on_eval_start()


@contextmanager
def quiet_eval_output(on_eval_start: Callable[[], None]) -> Iterator[_QuietEvalOutput]:
    """Silence noisy vLLM/lm-eval output during baseline execution."""
    previous_vllm_logging = os.environ.get("VLLM_CONFIGURE_LOGGING")
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

    stream = _QuietEvalOutput(on_eval_start)
    logger_names = ("vllm", "lm_eval")
    original_levels: dict[str, int] = {}

    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.ERROR)

    try:
        with redirect_stdout(stream), redirect_stderr(stream):
            yield stream
    finally:
        stream.flush()
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)
        if previous_vllm_logging is None:
            os.environ.pop("VLLM_CONFIGURE_LOGGING", None)
        else:
            os.environ["VLLM_CONFIGURE_LOGGING"] = previous_vllm_logging


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

    with spinner("Setting up vLLM") as set_eval_step:
        with quiet_eval_output(lambda: set_eval_step("Running evals")) as eval_output:
            baseline_result = run_baseline(
                model_path=local_model_path,
                eval_name=eval_name,
                gpus=gpus,
            )
        if not eval_output.eval_started:
            set_eval_step("Running evals")

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
