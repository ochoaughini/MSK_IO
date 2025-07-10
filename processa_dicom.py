#!/usr/bin/env python
"""Download a DICOM series from an OHIF viewer and save as NIfTI."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse
import base64
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import typer


@dataclass
class TokenStatus:
    """Record how the authentication token was sourced and validated."""

    source: str
    expired: bool | None = None


def _check_expiration(token: str) -> bool | None:
    """Return True if expired, False if valid, None if undecodable."""
    try:
        payload_b64 = token.split(".")[1]
        # Pad base64 if necessary
        pad = len(payload_b64) % 4
        if pad:
            payload_b64 += "=" * (4 - pad)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode())
        exp = payload.get("exp")
        if exp is None:
            return None
        return datetime.now(timezone.utc).timestamp() > exp
    except Exception:
        return None

from msk_io.preprocessing.nifti_converter import NiftiConverter
from msk_io.retrieval.remote_loader import RemoteDICOMLoader

app = typer.Typer(add_completion=False)


@app.command()
def main(
    url: Optional[str] = None,
    out: str = "volume.nii.gz",
    token: Optional[str] = None,
    slices: int = 1,
) -> None:
    """Fetch ``url`` and save the volume to ``out``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    if url is None:
        url = os.environ.get("MSK_REMOTE_URL")
        if not url:
            raise typer.BadParameter("URL must be provided via argument or MSK_REMOTE_URL")
        logging.warning("Using MSK_REMOTE_URL environment variable")
    status = TokenStatus(source="url")
    if token is None and url:
        qs = parse_qs(urlparse(url).query)
        token = qs.get("token", [None])[0]
    if not token:
        token = os.environ.get("MSK_AUTH_TOKEN")
        status.source = "env"
        if not token:
            logging.warning("No authentication token provided")
    status.expired = _check_expiration(token) if token else None

    loader = RemoteDICOMLoader(slices=slices)
    volume = loader.load(url, token)
    path = NiftiConverter().to_nifti(volume, Path(out))
    with open("token_status.json", "w") as f:
        json.dump(status.__dict__, f)
    typer.echo(f"Saved {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
