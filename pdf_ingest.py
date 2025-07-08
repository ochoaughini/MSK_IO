"""Standalone PDF ingestion using MSK_IO components."""
from __future__ import annotations

import asyncio
from pathlib import Path
import typer

from msk_io.pdf.pdf_ingestor import MSKPDFIngestor

app = typer.Typer(add_completion=False)


@app.command()
async def ingest(path: str) -> None:
    ingestor = MSKPDFIngestor()
    loop = asyncio.get_running_loop()
    texts = await loop.run_in_executor(None, ingestor.ingest, Path(path))
    for t in texts:
        typer.echo(t[:80])


if __name__ == "__main__":  # pragma: no cover
    app()
