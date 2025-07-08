"""Fuse pipeline outputs into a simple report."""
from __future__ import annotations

from msk_io.symbolic.symbolic_state_emitter import SymbolicState
from msk_io.control.multi_agent_harmonizer import AgentOutput, MultiAgentHarmonizer


def generate_report(state: SymbolicState) -> str:
    return f"Predicates: {state.predicates}, confidence: {state.confidence:.2f}"


def finalize(outputs: list[AgentOutput]) -> str:
    state = MultiAgentHarmonizer().harmonize(outputs)
    return generate_report(state)
