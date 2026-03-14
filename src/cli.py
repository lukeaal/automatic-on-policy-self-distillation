"""CLI entrypoint using Google Fire."""

import fire


class CLI:
    """Main CLI with subcommands."""

    def train(self, **kwargs):
        """Run training."""
        print("train", kwargs)

    def eval(self, **kwargs):
        """Run evaluation."""
        print("eval", kwargs)

    def viz(self, **kwargs):
        """Run visualization."""
        print("viz", kwargs)


def main():
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
