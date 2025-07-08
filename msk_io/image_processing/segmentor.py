import numpy as np

class Segmentor:
    """Dummy segmentation that thresholds the volume."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def segment(self, volume: np.ndarray) -> np.ndarray:
        volume = volume.astype(np.float32)
        max_val = volume.max() or 1.0
        return (volume / max_val > self.threshold).astype(np.uint8)
