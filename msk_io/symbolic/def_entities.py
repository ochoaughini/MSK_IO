from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DiagnosticEntity:
    """Representation of a segmented structure or finding."""

    name: str
    volume: int = 0
    properties: Dict[str, float] | None = None


@dataclass
class EntityRelation:
    """Directed relationship between two diagnostic entities."""

    source: str
    target: str
    relation: str
    weight: float = 1.0
