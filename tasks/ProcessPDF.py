"""Simple task to process a PDF using Varkiel agents."""
from __future__ import annotations

from pathlib import Path

from core.Agent import Agent
from core.Context import Context
from core.Runner import Runner
from core.Tool import Tool
from image.PDFHandler import PDFHandler


def process(pdf_path: Path) -> Context:
    handler = PDFHandler()
    tool = Tool("read", lambda _: handler.load(pdf_path))
    agent = Agent("reader", tool)
    runner = Runner([agent])
    return runner.run(Context())
