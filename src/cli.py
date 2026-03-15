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
    model: str = typer.Option(
        ..., "--model", help="Student model id used by lm-eval local-completions."
    ),
    eval_name: str = typer.Option(
        "gsm8k",
        "--eval",
        help="lm-eval task name (for example: gsm8k, hellaswag).",
    ),
    base_url: str = typer.Option(
        "http://localhost:8000/v1",
        "--base-url",
        help="OpenAI-compatible endpoint for your served student model.",
    ),
    eval_api_key: str = typer.Option(
        "EMPTY",
        "--eval-api-key",
        help="API key sent to local-completions endpoint.",
    ),
    num_fewshot: int = typer.Option(
        0, "--num-fewshot", help="Few-shot count passed to lm-eval."
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Optional sample limit for fast iteration."
    ),
    output_file: Path = typer.Option(
        Path("src/agent/best_hyp.py"),
        "--output-file",
        help="Path to write the best hypothesis python file.",
    ),
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
        eval_name=eval_name,
        model_id=model,
        base_url=base_url,
        api_key=eval_api_key,
        num_fewshot=num_fewshot,
        limit=limit,
    )

    typer.echo(f"Completed {len(history)} hypothesis trials on eval={eval_name}")
    typer.echo(f"Best score={best_score}")
    if not best_hypothesis.strip():
        raise typer.BadParameter("Best hypothesis is empty; refusing to write output file.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(best_hypothesis.rstrip() + "\n", encoding="utf-8")
    typer.echo(f"Wrote best hypothesis to {output_file}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
