"""CLI entrypoint using Typer."""

from pathlib import Path

import typer

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
