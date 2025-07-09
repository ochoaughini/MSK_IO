from msk_io.control.multi_agent_harmonizer import MultiAgentHarmonizer, AgentOutput
from msk_io.symbolic.symbolic_state_emitter import SymbolicState


def test_harmonize_contradictory():
    harmonizer = MultiAgentHarmonizer()
    outputs = [
        AgentOutput(SymbolicState(["bone_marrow_edema"], 0.8), 0.3, agent_id="A"),
        AgentOutput(SymbolicState(["normal_marrow"], 0.7), 0.3, agent_id="B"),
        AgentOutput(SymbolicState(["synovitis"], 0.6), 0.4, agent_id="C"),
    ]
    state = harmonizer.harmonize(outputs)
    assert state.predicates in (["bone_marrow_edema"], ["normal_marrow"], ["synovitis"])
