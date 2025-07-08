from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from logging.handlers import RotatingFileHandler
import logging
import json

from .api import PipelineRunner
from .config import PipelineSettings
from .metrics_server import create_metrics_app
from .pdf.pdf_ingestor import MSKPDFIngestor
from .storage.memory_vault import MemoryVault

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_settings(path: Optional[Path]) -> PipelineSettings:
    if path and path.exists():
        return PipelineSettings.model_validate_json(path.read_text())
    return PipelineSettings()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    debug: bool = typer.Option(False, "--debug"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    settings = _load_settings(config)
    ctx.obj = settings
    log_level = logging.DEBUG if debug or verbose else getattr(logging, settings.log_level.upper(), logging.INFO)
    handlers = [RichHandler(rich_tracebacks=True)]
    file_handler = RotatingFileHandler("logs/app.log", maxBytes=10_000_000, backupCount=3)
    file_handler.setFormatter(
        logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
    )
    handlers.append(file_handler)
    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    if dry_run:
        console.print_json(data=json.loads(settings.model_dump_json()))
        raise typer.Exit()


@app.command()
def run(ctx: typer.Context, pdf: Optional[Path] = typer.Option(None, "--pdf")) -> None:
    """Execute the full pipeline."""
    settings: PipelineSettings = ctx.obj
    if pdf:
        MSKPDFIngestor().ingest(pdf)
    vault = MemoryVault(settings.vault.path)
    result = PipelineRunner().run(settings, vault)
    console.print_json(data=result)


@app.command()
def ingest(ctx: typer.Context, pdf: Path) -> None:
    settings: PipelineSettings = ctx.obj
    ingestor = MSKPDFIngestor()
    texts = ingestor.ingest(pdf)
    console.print(f"Ingested {len(texts)} pages")


@app.command()
def index(ctx: typer.Context) -> None:
    settings: PipelineSettings = ctx.obj
    runner = PipelineRunner()
    runner.indexer.index_items(["dummy"])
    console.print("Index built")


@app.command()
def serve(ctx: typer.Context) -> None:
    console.print("Serving (stub)")


@app.command("serve-metrics")
def serve_metrics(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Expose Prometheus metrics via Uvicorn."""
    import uvicorn

    uvicorn.run(create_metrics_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    app()
