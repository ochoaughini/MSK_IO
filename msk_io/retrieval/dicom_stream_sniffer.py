from __future__ import annotations

"""Retrieve DICOM files by monitoring network traffic of an OHIF session."""

import asyncio
import io
from typing import Optional, List

import numpy as np
import httpx
import pydicom

from .external_dicom_source import ExternalDICOMSource
from .nbia_authenticator import NBIAAuthenticator


class DICOMStreamSniffer(ExternalDICOMSource):
    """Network-level DICOM extractor using ``mitmproxy`` logs."""

    def __init__(self, url: str, token: Optional[str] = None) -> None:
        self.url = url
        self.token = token
        self.auth = NBIAAuthenticator(token) if token else None

    async def _discover_endpoints(self) -> List[str]:
        """Launch mitmproxy in passive mode and capture DICOM URLs."""
        # In this simplified implementation we just assume the OHIF viewer
        # exposes a `/dicom/` endpoint returning files.
        return [f"{self.url}/dicom/{i}.dcm" for i in range(1, 2)]

    async def retrieve(self) -> np.ndarray:
        endpoints = await self._discover_endpoints()
        headers = self.auth.headers() if self.auth else None
        volume = []
        async with httpx.AsyncClient() as client:
            for ep in endpoints:
                resp = await client.get(ep, headers=headers)
                resp.raise_for_status()
                ds = pydicom.dcmread(io.BytesIO(resp.content))
                volume.append(ds.pixel_array)
                await asyncio.sleep(0)
        return np.stack(volume)
