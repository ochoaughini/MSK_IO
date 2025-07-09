#!/usr/bin/env python
"""Download a DICOM series from an OHIF viewer and save as NIfTI."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import typer

from msk_io.preprocessing.nifti_converter import NiftiConverter
from msk_io.retrieval.remote_loader import RemoteDICOMLoader

app = typer.Typer(add_completion=False)


@app.command()
def main(
    url: str, out: str = "volume.nii.gz", token: str | None = None, slices: int = 1
) -> None:
    """Fetch ``url`` and save the volume to ``out``."""
    if token is None:
        qs = parse_qs(urlparse(url).query)
        token = qs.get("token", [None])[0]
    loader = RemoteDICOMLoader(slices=slices)
    volume = loader.load(url, token)
    path = NiftiConverter().to_nifti(volume, Path(out))
    typer.echo(f"Saved {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
