from dataclasses import dataclass
from typing import List
import numpy as np
from datetime import datetime
from ..symbolic.symbolic_state_emitter import SymbolicState
from .policy_registry import POLICIES, ScoringPolicy, register_policy


@dataclass
class AgentOutput:
    state: SymbolicState
    weight: float
    agent_id: str = "unknown"
    timestamp: datetime = datetime.utcnow()
    model_version: str | None = None
    metadata: dict | None = None




@register_policy("weighted")
class WeightedSumPolicy:
    """Weight confidences by provided agent weights."""
    def score(self, outputs: List[AgentOutput]) -> np.ndarray:
        weights = np.array([o.weight for o in outputs], dtype=float)
        weights = weights / weights.sum() if weights.sum() else weights
        scores = np.array([o.state.confidence for o in outputs], dtype=float)
        return weights * scores


@register_policy("softmax")
class SoftmaxPolicy:
    """Apply softmax to agent confidences."""
    def score(self, outputs: List[AgentOutput]) -> np.ndarray:
        conf = np.array([o.state.confidence for o in outputs], dtype=float)
        e = np.exp(conf)
        return e / e.sum()


@register_policy("bayesian")
class BayesianFusionPolicy:
    """Normalize confidences as Bayesian fusion."""
    def score(self, outputs: List[AgentOutput]) -> np.ndarray:
        conf = np.array([o.state.confidence for o in outputs], dtype=float)
        return conf / conf.sum() if conf.sum() else conf


@register_policy("consensus")
class ConsensusVotingPolicy:
    """Vote for the most common predicate set."""
    def score(self, outputs: List[AgentOutput]) -> np.ndarray:
        preds = [tuple(o.state.predicates) for o in outputs]
        most_common = max(set(preds), key=preds.count)
        return np.array(
            [1.0 if tuple(o.state.predicates) == most_common else 0.0 for o in outputs]
        )


class MultiAgentHarmonizer:
    """Combine multiple agent outputs via pluggable scoring."""

    def __init__(self, policy: ScoringPolicy | str | None = None):
        if isinstance(policy, str):
            policy_cls = POLICIES.get(policy, WeightedSumPolicy)
            self.policy = policy_cls()
        else:
            self.policy = policy or WeightedSumPolicy()

    def harmonize(self, outputs: List[AgentOutput]) -> SymbolicState:
        if not outputs:
            raise ValueError("No agent outputs provided")
        total_weight = sum(o.weight for o in outputs)
        if not 0 < total_weight <= 1:
            raise ValueError("Sum of weights must be within (0,1]")
        scores = self.policy.score(outputs)
        if np.all(scores == 0):
            return outputs[0].state
        idx = int(np.argmax(scores))
        return outputs[idx].state

    async def harmonize_async(self, outputs: List[AgentOutput]) -> SymbolicState:
        return self.harmonize(outputs)
