from __future__ import annotations

"""High level loader that combines canvas capture and network sniffing."""

import asyncio
import logging
from typing import Optional

import numpy as np

from .ohif_canvas_extractor import OHIFCanvasExtractor
from .dicom_stream_sniffer import DICOMStreamSniffer

logger = logging.getLogger(__name__)


class RemoteDICOMLoader:
    """Attempt to fetch DICOM volume from a remote OHIF viewer."""

    def __init__(self, slices: int = 1) -> None:
        self.slices = slices

    async def _load_async(self, url: str, token: Optional[str]) -> np.ndarray:
        try:
            canvas = OHIFCanvasExtractor(url, token, slices=self.slices)
            volume = await canvas.retrieve()
            if volume.ndim == 3:
                return volume
            logger.warning("Canvas capture returned unexpected shape")
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Canvas capture failed: %s", exc)
        sniffer = DICOMStreamSniffer(url, token)
        volume = await sniffer.retrieve()
        return volume

    def load(self, url: str, token: Optional[str] = None) -> np.ndarray:
        """Synchronous wrapper for :meth:`_load_async`."""
        return asyncio.run(self._load_async(url, token))
