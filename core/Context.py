"""Context object passed between agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Context:
    data: Dict[str, Any] = field(default_factory=dict)
