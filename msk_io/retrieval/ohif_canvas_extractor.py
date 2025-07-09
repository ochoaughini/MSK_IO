from __future__ import annotations

"""Capture rendered frames from an OHIF viewer canvas."""

import asyncio
import base64
import io
from typing import Optional, List

import numpy as np
from PIL import Image

from .external_dicom_source import ExternalDICOMSource


class OHIFCanvasExtractor(ExternalDICOMSource):
    """Headless browser canvas scraper using ``pyppeteer``."""

    def __init__(self, url: str, token: Optional[str] = None, slices: int = 1) -> None:
        self.url = url
        self.token = token
        self.slices = slices

    async def _capture_slice(self, page) -> np.ndarray:
        elem = await page.querySelector("canvas")
        data_url = await page.evaluate("(e) => e.toDataURL()", elem)
        _, b64 = data_url.split(",", 1)
        buf = base64.b64decode(b64)
        img = Image.open(io.BytesIO(buf))
        return np.array(img)

    async def retrieve(self) -> np.ndarray:
        from pyppeteer import launch  # type: ignore

        browser = await launch(headless=True, args=["--no-sandbox"])
        page = await browser.newPage()
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        if headers:
            await page.setExtraHTTPHeaders(headers)
        await page.goto(self.url)
        await page.waitForSelector("canvas")
        frames: List[np.ndarray] = []
        for _ in range(self.slices):
            arr = await self._capture_slice(page)
            frames.append(arr)
            await page.keyboard.press("ArrowDown")
            await asyncio.sleep(0.1)
        await browser.close()
        return np.stack(frames)
