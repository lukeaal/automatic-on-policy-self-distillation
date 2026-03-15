"""CLI entrypoint using Typer."""

from pathlib import Path

import typer

from .agent.foundation_model import FoundationModel
from .agent.optimizer import run_hypothesis_loop
from .run import run_command

app = typer.Typer()


@app.command()
def help(ctx: typer.Context) -> None:
    """Show CLI help."""
    typer.echo(ctx.parent.get_help() if ctx.parent else ctx.get_help())


@app.command()
def run(
    model: str = typer.Option(..., "--model", help="Model identifier to run."),
    evals: Path = typer.Option(..., "--evals", help="Path to evals file."),
    trials: int = typer.Option(..., "--trials", help="Number of trials to run."),
) -> None:
    """Run evaluations for a model."""
    run_command(model=model, evals=evals, trials=trials)


@app.command("opt_hyp")
def opt_hyp(
    fm: str = typer.Option(..., "--fm", help="Foundation model identifier."),
    trials: int = typer.Option(
        ..., "--trials", help="Number of trials in hypothesis loop."
    ),
) -> None:
    """Run optimizer hypothesis-generation loop only."""
    if trials < 1:
        raise typer.BadParameter("trials must be >= 1")

    foundation_model = FoundationModel(model_id=fm)
    best_hypothesis, best_score, history = run_hypothesis_loop(
        foundation_model=foundation_model,
        trials=trials,
        eval_name="gsm8k",
    )

    typer.echo(f"Completed {len(history)} hypothesis trials on eval=gsm8k")
    typer.echo(f"Best score={best_score}")
    typer.echo("\nBest hypothesis:\n")
    typer.echo(best_hypothesis)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
