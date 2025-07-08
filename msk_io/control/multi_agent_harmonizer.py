from dataclasses import dataclass
from typing import List
import numpy as np
from ..symbolic.symbolic_state_emitter import SymbolicState

@dataclass
class AgentOutput:
    state: SymbolicState
    weight: float

class MultiAgentHarmonizer:
    """Combine multiple agent outputs via weighted softmax."""

    def harmonize(self, outputs: List[AgentOutput]) -> SymbolicState:
        weights = np.array([o.weight for o in outputs], dtype=float)
        scores = np.array([o.state.confidence for o in outputs], dtype=float)
        weighted = weights * scores
        idx = int(np.argmax(weighted))
        return outputs[idx].state
