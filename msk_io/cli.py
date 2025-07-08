from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import typer
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from rich.console import Console
from rich.logging import RichHandler
from logging.handlers import RotatingFileHandler

from .api import PipelineRunner, PipelineResult
from .config import PipelineSettings
from .storage.memory_vault import MemoryVault

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


def create_logger(level: int) -> None:
    logger = logging.getLogger()
    logger.handlers.clear()
    fmt = (
        '{"time":"%(asctime)s","level":"%(levelname)s","correlation":"%(correlation)s","msg":"%(message)s"}'
    )
    correlation = str(uuid.uuid4())[:8]

    class CorrelationFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.correlation = correlation
            return True

    rich_handler = RichHandler(rich_tracebacks=True)
    file_handler = RotatingFileHandler(
        "logs/app.log", maxBytes=5_000_000, backupCount=2
    )
    for h in (rich_handler, file_handler):
        h.setFormatter(logging.Formatter(fmt))
        h.addFilter(CorrelationFilter())
        logger.addHandler(h)
    logger.setLevel(level)


def _load_settings(path: Optional[Path]) -> PipelineSettings:
    if path and path.exists():
        return PipelineSettings.model_validate_json(path.read_text())
    return PipelineSettings()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config file"),
    pdf: Optional[str] = typer.Option(None, "--pdf", help="PDF to ingest"),
    data_path: Optional[str] = typer.Option(None, "--data-path"),
    vector_db_url: Optional[str] = typer.Option(None, "--vector-db-url"),
    ocr_enabled: bool = typer.Option(False, "--ocr-enabled", is_flag=True),
    log_level: str = typer.Option("INFO", "--log-level"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    settings = _load_settings(Path(config) if config else None)
    overrides = {}
    if pdf is not None:
        overrides["pdf_path"] = Path(pdf)
    if data_path is not None:
        overrides["data_path"] = Path(data_path)
    if vector_db_url is not None:
        overrides["vector_db_url"] = vector_db_url
    if ocr_enabled is not None:
        overrides["ocr_enabled"] = ocr_enabled
    if log_level:
        overrides["log_level"] = log_level
    if overrides:
        settings = settings.model_copy(update=overrides)
    ctx.obj = settings
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    create_logger(level)
    if dry_run:
        console.print_json(data=json.loads(settings.model_dump_json()))
        raise typer.Exit()


@app.command()
def run(ctx: typer.Context) -> None:
    settings: PipelineSettings = ctx.obj
    vault = MemoryVault(settings.vault.path)
    result = PipelineRunner().run(settings, vault)
    console.print_json(data=result.__dict__)


@app.command()
def serve(ctx: typer.Context, host: str = "0.0.0.0", port: int = 8000) -> None:
    settings: PipelineSettings = ctx.obj
    runner = PipelineRunner()
    api_app = FastAPI()

    @api_app.post("/rpc")
    async def rpc(payload: dict) -> JSONResponse:  # type: ignore[valid-type]
        if payload.get("method") == "run":
            res: PipelineResult = await runner.run_async(
                settings, MemoryVault(settings.vault.path)
            )
            return JSONResponse(res.__dict__)
        return JSONResponse({"error": "unknown method"}, status_code=400)

    api_app.mount("/metrics", make_asgi_app())

    import uvicorn

    uvicorn.run(api_app, host=host, port=port, log_level="info")


@app.command("serve-metrics")
def serve_metrics(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(make_asgi_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    app()
