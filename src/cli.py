"""CLI entrypoint using Typer."""

from datetime import datetime
from pathlib import Path

import typer

from .agent.foundation_model import FoundationModel
from .agent.optimizer import run_hypothesis_loop
from .run import run_command
from .self_distill import self_distill

app = typer.Typer()


def _distilled_model_dir(model: str, output_dir: Path | None) -> Path:
    """Build an output path like `<model-name>-asd-<timestamp>`."""
    model_name = model.rstrip("/").split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parent = output_dir if output_dir is not None else Path.cwd()
    return parent / f"{model_name}-asd-{timestamp}"


@app.command()
def help(ctx: typer.Context) -> None:
    """Show CLI help."""
    typer.echo(ctx.parent.get_help() if ctx.parent else ctx.get_help())


@app.command()
def run(
    model: str = typer.Option(..., "--model", help="Model identifier to run."),
    eval_name: str = typer.Option(
        ...,
        "--eval",
        help="lm-eval task name to run, or a comma-separated list of task names.",
    ),
    gpus: int | None = typer.Option(
        None,
        "--gpus",
        help="Number of GPUs for vLLM tensor parallelism. Defaults to auto-detect.",
    ),
) -> None:
    """Run evaluations for a model."""
    run_command(model=model, eval_name=eval_name, gpus=gpus)


@app.command("self-distill")
def self_distill_command(
    model: str = typer.Option(..., "--model", help="Model identifier or local path."),
    dataset: Path = typer.Option(..., "--dataset", help="Path to json/jsonl/parquet training data."),
    batch_size: int = typer.Option(4, "--batch-size", min=1, help="Training batch size."),
    epochs: int = typer.Option(1, "--epochs", min=1, help="Number of training epochs."),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate."),
    max_new_tokens: int = typer.Option(128, "--max-new-tokens", min=1, help="Max sampled rollout length."),
    max_length: int = typer.Option(512, "--max-length", min=2, help="Max prompt+response sequence length."),
    teacher_update_steps: int = typer.Option(
        0,
        "--teacher-update-steps",
        min=0,
        help="If > 0, copy student weights into teacher every N steps.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Optional parent directory for the distilled model output.",
    ),
) -> None:
    """Run the minimal reverse-KL self-distillation loop."""
    trained_model, tokenizer = self_distill(
        model_name_or_path=model,
        dataset_source=dataset,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        teacher_update_steps=teacher_update_steps,
    )

    save_dir = _distilled_model_dir(model=model, output_dir=output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trained_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    typer.echo(f"Saved distilled model to {save_dir}")


@app.command("opt_hyp")
def opt_hyp(
    fm: str = typer.Option(..., "--fm", help="Foundation model identifier."),
    model: str = typer.Option(
        ..., "--model", help="Student model id used by lm-eval local-completions."
    ),
    eval_name: str = typer.Option(
        "gsm8k",
        "--eval",
        help="Eval name to optimize (currently supported: gsm8k).",
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
