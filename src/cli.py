"""CLI entrypoint using Typer."""

import typer

app = typer.Typer()


@app.command()
def train():
    """Run training."""
    typer.echo("train")


@app.command()
def eval():
    """Run evaluation."""
    typer.echo("eval")


@app.command()
def viz():
    """Run visualization."""
    typer.echo("viz")


def main():
    app()


if __name__ == "__main__":
    main()
