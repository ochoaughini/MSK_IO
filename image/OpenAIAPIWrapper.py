"""Placeholder API wrapper for external LLM calls."""
from __future__ import annotations

from typing import Any


class OpenAIAPIWrapper:
    def call(self, prompt: str) -> Any:
        # In real usage this would call the OpenAI API.
        return {"response": f"Echo: {prompt}"}
