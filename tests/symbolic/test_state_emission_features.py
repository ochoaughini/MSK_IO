import numpy as np
from msk_io.symbolic.symbolic_state_emitter import SymbolicStateEmitter, SymbolicState


def test_emit_numerical_state() -> None:
    emitter = SymbolicStateEmitter()
    text_emb = np.array([0.2, 0.8])
    img_emb = np.array([0.7, 0.9])
    state = emitter.emit_state(text_emb, img_emb)
    assert isinstance(state, SymbolicState)
    assert state.predicates
    assert 0.0 <= state.confidence <= 1.0
