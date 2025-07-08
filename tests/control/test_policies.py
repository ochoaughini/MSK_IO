from msk_io.control.policy_registry import POLICIES
from msk_io.control.multi_agent_harmonizer import (
    MultiAgentHarmonizer,
    AgentOutput,
    WeightedSumPolicy,
)
from msk_io.symbolic.symbolic_state_emitter import SymbolicState


def test_policy_registry():
    assert "weighted" in POLICIES
    assert issubclass(POLICIES["weighted"], WeightedSumPolicy)


def test_harmonizer_weights():
    harmonizer = MultiAgentHarmonizer()
    out1 = AgentOutput(SymbolicState(["a"], 0.8), 0.4)
    out2 = AgentOutput(SymbolicState(["b"], 0.9), 0.6)
    result = harmonizer.harmonize([out1, out2])
    assert result in (out1.state, out2.state)
