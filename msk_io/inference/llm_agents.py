from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..symbolic.symbolic_state_emitter import SymbolicStateEmitter, SymbolicState


@dataclass
class BaseAgent:
    """Base class for simple volume analysis agents."""

    name: str = ""
    weight: float = 1.0
    model_version: str = "0"
    emitter: SymbolicStateEmitter | None = None

    def __post_init__(self) -> None:
        if self.emitter is None:
            self.emitter = SymbolicStateEmitter()

    def analyze_volume(self, volume: np.ndarray) -> SymbolicState:
        """Return a symbolic state using the mean pixel intensity."""
        mean_val = float(volume.mean())
        emb = np.array([mean_val / (volume.max() or 1.0)])
        return self.emitter.emit_state(emb, emb)


class MiniGPTAgent(BaseAgent):
    name = "miniGPT"


class GEMAAgent(BaseAgent):
    name = "GEMA"


class PHI2Agent(BaseAgent):
    name = "PHI-2"


class TextAgent(BaseAgent):
    """Simple agent that reasons over indexed text."""

    name = "text"

    def __init__(self, indexer, **kwargs) -> None:
        super().__init__(**kwargs)
        self.indexer = indexer

    def analyze_volume(self, volume: np.ndarray) -> SymbolicState:
        text = " ".join(getattr(self.indexer, "items", []))
        lowered = text.lower()
        keywords = ["mass", "lesion", "abnormal"]
        if any(k in lowered for k in keywords):
            return SymbolicState(["positive"], 0.9)
        if text:
            return SymbolicState(["negative"], 0.9)
        return SymbolicState(["negative"], 0.5)

import asyncio
from typing import Iterable, List
from ..control.multi_agent_harmonizer import AgentOutput


def analyze_with_agents(volume: np.ndarray, agents: Iterable[BaseAgent]) -> List[AgentOutput]:
    """Run agents concurrently over the volume."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [loop.run_in_executor(None, a.analyze_volume, volume) for a in agents]
    states = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    outputs = [
        AgentOutput(state=s, weight=a.weight, agent_id=a.name, model_version=a.model_version)
        for a, s in zip(agents, states)
    ]
    return outputs
