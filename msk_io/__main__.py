from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from .api import PipelineRunner
from .config import PipelineConfig

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_config(path: Optional[Path]) -> PipelineConfig:
    if path and path.exists():
        return PipelineConfig.model_validate_json(path.read_text())
    return PipelineConfig()


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    cfg = _load_config(config)
    cfg.verbose = verbose
    cfg.dry_run = dry_run
    cfg.debug = debug
    ctx.obj = cfg
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, handlers=[RichHandler(rich_tracebacks=True)])


@app.command()
def run(ctx: typer.Context, dicom_dir: Path, vault: Path) -> None:
    """Execute full pipeline."""
    runner = PipelineRunner()
    cfg: PipelineConfig = ctx.obj
    if cfg.dry_run:
        console.print(f"[yellow]Dry run with config: {cfg}")
        raise typer.Exit()
    result = runner.run(dicom_dir, cfg, vault)
    console.print_json(data=result)


@app.command()
def load(ctx: typer.Context, dicom_dir: Path) -> None:
    runner = PipelineRunner()
    volume = runner.loader.load_series(dicom_dir)
    console.print(f"Loaded volume shape {volume.shape}")


@app.command()
def segment(ctx: typer.Context, dicom_dir: Path) -> None:
    runner = PipelineRunner()
    volume = runner.loader.load_series(dicom_dir)
    mask = runner.segmentor.segment(volume)
    console.print(f"Mask shape: {mask.shape}")


if __name__ == "__main__":  # pragma: no cover
    app()
