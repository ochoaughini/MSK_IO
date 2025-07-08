from msk_io.control.multi_agent_harmonizer import MultiAgentHarmonizer, AgentOutput
from msk_io.symbolic.symbolic_state_emitter import SymbolicState


def test_harmonize():
    harmonizer = MultiAgentHarmonizer()
    out1 = AgentOutput(SymbolicState(["a"], 0.9), 0.5, agent_id="A")
    out2 = AgentOutput(SymbolicState(["b"], 0.6), 1.0, agent_id="B")
    result = harmonizer.harmonize([out1, out2])
    assert result.predicates in (["a"], ["b"])
