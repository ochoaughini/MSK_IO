from dataclasses import dataclass
from typing import Dict
import numpy as np

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
