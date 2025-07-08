from dataclasses import dataclass
from typing import List, Protocol
import numpy as np
from datetime import datetime
from ..symbolic.symbolic_state_emitter import SymbolicState


@dataclass
class AgentOutput:
    state: SymbolicState
    weight: float
    agent_id: str = "unknown"
    timestamp: datetime = datetime.utcnow()


class ScoringPolicy(Protocol):
    def score(self, outputs: List[AgentOutput]) -> np.ndarray: ...


class WeightedSumPolicy:
    def score(self, outputs: List[AgentOutput]) -> np.ndarray:
        weights = np.array([o.weight for o in outputs], dtype=float)
        weights = weights / weights.sum() if weights.sum() else weights
        scores = np.array([o.state.confidence for o in outputs], dtype=float)
        return weights * scores


class MultiAgentHarmonizer:
    """Combine multiple agent outputs via pluggable scoring."""

    def __init__(self, policy: ScoringPolicy | None = None):
        self.policy = policy or WeightedSumPolicy()

    def harmonize(self, outputs: List[AgentOutput]) -> SymbolicState:
        if not outputs:
            raise ValueError("No agent outputs provided")
        scores = self.policy.score(outputs)
        if np.all(scores == 0):
            return outputs[0].state
        idx = int(np.argmax(scores))
        return outputs[idx].state

    async def harmonize_async(self, outputs: List[AgentOutput]) -> SymbolicState:
        return self.harmonize(outputs)
