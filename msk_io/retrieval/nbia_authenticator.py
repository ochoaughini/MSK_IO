from __future__ import annotations

"""Utility for NBIA/OHIF token-based authentication."""

from typing import Optional, Dict


class NBIAAuthenticator:
    """Simple bearer token header generator."""

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token

    def headers(self) -> Dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}
