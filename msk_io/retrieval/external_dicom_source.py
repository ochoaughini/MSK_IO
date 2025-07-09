from __future__ import annotations

"""Interface for external DICOM data sources."""

import numpy as np
from abc import ABC, abstractmethod


class ExternalDICOMSource(ABC):
    """Retrieve DICOM images from an external service."""

    @abstractmethod
    async def retrieve(self) -> np.ndarray:
        """Return a volume as a numpy array."""
        raise NotImplementedError
