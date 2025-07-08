from pathlib import Path
from msk_io.storage.memory_vault import MemoryVault
from msk_io.symbolic.symbolic_state_emitter import SymbolicState


def test_checkpoint_and_retrieve(tmp_path: Path):
    vault = MemoryVault(tmp_path / 'db.sqlite')
    state = SymbolicState(['x'], 0.8)
    h = vault.checkpoint(state)
    retrieved = vault.retrieve(h)
    assert retrieved.predicates == state.predicates
