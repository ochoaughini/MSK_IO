#!/usr/bin/env python
"""CLI for cropping a volume to its body mask using TotalSegmentatorV2."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import typer

from totalsegmentatorv2.cropping import crop_to_mask

app = typer.Typer(add_completion=False)


@app.command()
def main(volume: str, mask: str, out: str) -> None:
    """Crop ``volume`` with ``mask`` and save to ``out``."""
    vol = nib.load(volume).get_fdata()
    msk = nib.load(mask).get_fdata()
    cropped = crop_to_mask(vol, msk)
    nib.save(nib.Nifti1Image(cropped, np.eye(4)), out)


if __name__ == "__main__":  # pragma: no cover
    app()
