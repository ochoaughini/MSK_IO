import json
from pathlib import Path
from msk_io.inference.constraint_lattice import ConstraintLattice
from msk_io.symbolic.symbolic_state_emitter import SymbolicState


def test_validate_and_filter(tmp_path: Path):
    config = tmp_path / 'rules.json'
    config.write_text(json.dumps({'required_predicates': ['a'], 'allowed_predicates': ['a']}))
    lattice = ConstraintLattice(config)
    state_a = SymbolicState(['a'], 1.0)
    state_b = SymbolicState(['b'], 1.0)
    assert lattice.validate_chain([state_a])
    filtered = lattice.filter_chain([state_a, state_b])
    assert len(filtered) == 1
    assert filtered[0].predicates == ['a']
