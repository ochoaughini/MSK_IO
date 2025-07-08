from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class SymbolicState:
    predicates: List[str] = field(default_factory=list)
    confidence: float = 1.0

class SymbolicStateEmitter:
    """Emit symbolic state from text and image embeddings."""

    def emit_state(self, text_embeddings: np.ndarray, image_embeddings: np.ndarray) -> SymbolicState:
        score = float(text_embeddings.mean() + image_embeddings.mean()) / 2.0
        if score > 0.5:
            preds = ['positive']
        else:
            preds = ['negative']
        return SymbolicState(predicates=preds, confidence=score)
