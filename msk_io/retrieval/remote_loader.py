from __future__ import annotations

"""High level loader that combines canvas capture and network sniffing."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict

import numpy as np

from .ohif_canvas_extractor import OHIFCanvasExtractor
from .dicom_stream_sniffer import DICOMStreamSniffer

logger = logging.getLogger(__name__)


class RemoteDICOMLoader:
    """Attempt to fetch DICOM volume from a remote OHIF viewer."""

    def __init__(self, slices: int = 1) -> None:
        self.slices = slices

    def _dump_state(self, method: str, start: float, errors: Dict[str, str]) -> None:
        data = {
            "method": method,
            "start": start,
            "end": time.time(),
            "errors": errors,
        }
        try:
            Path("volume_retrieval.json").write_text(json.dumps(data))
        except Exception:  # pragma: no cover - diagnostics only
            pass

    async def _load_async(self, url: str, token: Optional[str]) -> np.ndarray:
        errors: Dict[str, str] = {}
        start = time.time()
        method = "canvas"
        try:
            canvas = OHIFCanvasExtractor(url, token, slices=self.slices)
            volume = await canvas.retrieve()
            if volume.ndim == 3:
                self._dump_state(method, start, errors)
                return volume
            errors[method] = "unexpected_shape"
            logger.warning("Canvas capture returned unexpected shape")
        except Exception as exc:  # pragma: no cover - best effort
            errors[method] = str(exc)
            logger.warning("Canvas capture failed: %s", exc)

        method = "sniffer"
        try:
            sniffer = DICOMStreamSniffer(url, token)
            volume = await sniffer.retrieve()
            self._dump_state(method, start, errors)
            return volume
        except Exception as exc:
            errors[method] = str(exc)
            self._dump_state(method, start, errors)
            raise

    def load(self, url: str, token: Optional[str] = None) -> np.ndarray:
        """Synchronous wrapper for :meth:`_load_async`."""
        return asyncio.run(self._load_async(url, token))
