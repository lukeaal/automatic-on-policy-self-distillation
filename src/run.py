"""Run command handler."""

from pathlib import Path

import typer

from .agent.foundation_model import FoundationModel
from .agent.student_model import StudentModel
from .vllm import VLLMModel


def run_command(model: str, evals: Path, trials: int) -> None:
    """Handle the CLI run command."""

    # Validate input parameters
    if not evals.exists():
        raise typer.BadParameter(f"Evals file not found: {evals}")
    if trials < 1:
        raise typer.BadParameter("trials must be >= 1")

    # Setup foundation model
    foundation_model = FoundationModel()

    # Download student model
    student_model = StudentModel(student_model_id=model)
    local_model_path = student_model.setup()

    # TODO: We want to run evals in parallel so lets deploy multiple instances
    # Run student model on vllm
    vllm_model = VLLMModel()
    vllm_model.load_model_for_serving(model=str(local_model_path))

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
