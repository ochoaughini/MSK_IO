from dataclasses import dataclass
from typing import List, Protocol, Dict, Type
import numpy as np
from datetime import datetime
from ..symbolic.symbolic_state_emitter import SymbolicState


@dataclass
class AgentOutput:
    state: SymbolicState
    weight: float
    agent_id: str = "unknown"
    timestamp: datetime = datetime.utcnow()
    model_version: str | None = None
    metadata: dict | None = None


class ScoringPolicy(Protocol):
    def score(self, outputs: List[AgentOutput]) -> np.ndarray: ...


POLICIES: Dict[str, Type[ScoringPolicy]] = {}


def register_policy(name: str):
    def _decorator(cls: Type[ScoringPolicy]) -> Type[ScoringPolicy]:
        POLICIES[name] = cls
        return cls

    return _decorator


@register_policy("weighted")
class WeightedSumPolicy:
    def score(self, outputs: List[AgentOutput]) -> np.ndarray:
        weights = np.array([o.weight for o in outputs], dtype=float)
        weights = weights / weights.sum() if weights.sum() else weights
        scores = np.array([o.state.confidence for o in outputs], dtype=float)
        return weights * scores


@register_policy("softmax")
class SoftmaxPolicy:
    def score(self, outputs: List[AgentOutput]) -> np.ndarray:
        conf = np.array([o.state.confidence for o in outputs], dtype=float)
        e = np.exp(conf)
        return e / e.sum()


@register_policy("bayesian")
class BayesianFusionPolicy:
    def score(self, outputs: List[AgentOutput]) -> np.ndarray:
        conf = np.array([o.state.confidence for o in outputs], dtype=float)
        return conf / conf.sum() if conf.sum() else conf


@register_policy("consensus")
class ConsensusVotingPolicy:
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
        scores = self.policy.score(outputs)
        if np.all(scores == 0):
            return outputs[0].state
        idx = int(np.argmax(scores))
        return outputs[idx].state

    async def harmonize_async(self, outputs: List[AgentOutput]) -> SymbolicState:
        return self.harmonize(outputs)
