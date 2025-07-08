from pathlib import Path
from typing import List
import json
from ..symbolic.symbolic_state_emitter import SymbolicState

class ConstraintLattice:
    """Validate and filter symbolic states using simple rules."""

    def __init__(self, config_path: Path):
        self.rules = json.loads(Path(config_path).read_text())

    def validate_chain(self, states: List[SymbolicState]) -> bool:
        required = set(self.rules.get('required_predicates', []))
        present = {p for s in states for p in s.predicates}
        return required.issubset(present)

    def filter_chain(self, states: List[SymbolicState]) -> List[SymbolicState]:
        allowed = set(self.rules.get('allowed_predicates', []))
        return [s for s in states if all(p in allowed for p in s.predicates)]
