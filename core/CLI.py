"""CLI entry for the Varkiel agent framework."""
from __future__ import annotations

import typer

from .Runner import Runner
from .Agent import Agent
from .Context import Context
from .Tool import Tool


app = typer.Typer(add_completion=False)


@app.command()
def run() -> None:
    """Run a trivial agent pipeline."""
    tool = Tool("echo", lambda d: "ok")
    agent = Agent("agent1", tool)
    ctx = Context()
    Runner([agent]).run(ctx)
    typer.echo(ctx.data)


if __name__ == "__main__":  # pragma: no cover
    app()
