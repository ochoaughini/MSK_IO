from __future__ import annotations

from typing import Dict, List, Protocol, Type
import numpy as np


class ScoringPolicy(Protocol):
    """Strategy interface for scoring agent outputs."""

    def score(self, outputs: List["AgentOutput"]) -> np.ndarray:
        ...


POLICIES: Dict[str, Type[ScoringPolicy]] = {}


def register_policy(name: str):
    """Register a scoring policy by name."""

    def decorator(cls: Type[ScoringPolicy]) -> Type[ScoringPolicy]:
        POLICIES[name] = cls
        return cls

    return decorator
