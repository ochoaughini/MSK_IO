"""Fetch web pages for prompt enrichment."""
from __future__ import annotations

import urllib.request


class WebFetcher:
    def fetch(self, url: str) -> str:
        with urllib.request.urlopen(url) as resp:
            return resp.read().decode()
