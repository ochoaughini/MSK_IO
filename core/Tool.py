"""Simple callable wrapper for tools."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    func: Callable[[Any], Any]

    def __call__(self, data: Any) -> Any:
        return self.func(data)
