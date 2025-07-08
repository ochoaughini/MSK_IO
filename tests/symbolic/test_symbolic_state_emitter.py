import numpy as np
from msk_io.symbolic.symbolic_state_emitter import SymbolicStateEmitter


def test_emit_state():
    emitter = SymbolicStateEmitter()
    state = emitter.emit_state(np.array([0.6]), np.array([0.6]))
    assert state.predicates
    assert 0 <= state.confidence <= 1
