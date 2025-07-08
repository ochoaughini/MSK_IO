from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .segmentor import Segmentor

try:  # pragma: no cover - optional dependency
    import onnxruntime as ort
except Exception:  # pragma: no cover - fallback
    ort = None


class DLSegmentor(Segmentor):
    """Deep learning segmentor with optional ONNX runtime."""

    def __init__(self, model_path: Optional[Path] = None, threshold: float = 0.5):
        super().__init__(threshold=threshold)
        self.model_path = model_path
        if model_path and ort:
            self.session = ort.InferenceSession(str(model_path))
        else:  # pragma: no cover - fallback
            self.session = None

    def segment(self, volume: np.ndarray) -> np.ndarray:  # type: ignore[override]
        if self.session:
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: volume[None].astype(np.float32)})[0]
            mask = output.argmax(axis=0).astype(np.uint8)
        else:
            mask = super().segment(volume)
        # simple small object removal
        labels, counts = np.unique(mask, return_counts=True)
        for label, count in zip(labels, counts):
            if label != 0 and count < 10:
                mask[mask == label] = 0
        return mask
