from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..symbolic.def_entities import DiagnosticEntity

@dataclass
class Predicate:
    label: int
    name: str

class ConstraintMapper:
    """Map segmentation labels to high-level predicates."""

    def __init__(self, mapping: Dict[int, str]):
        self.mapping = mapping

    def map(self, mask: np.ndarray) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for label, name in self.mapping.items():
            result[name] = int((mask == label).sum())
        return result

    def map_entities(self, mask: np.ndarray) -> List[DiagnosticEntity]:
        """Return a list of ``DiagnosticEntity`` objects for each label."""
        entities: List[DiagnosticEntity] = []
        for label, name in self.mapping.items():
            volume = int((mask == label).sum())
            entities.append(DiagnosticEntity(name=name, volume=volume))
        return entities
