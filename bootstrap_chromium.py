#!/usr/bin/env python3
"""Fallback helper to unpack a local Chromium archive.

Run by bootstrap_chromium.sh when a system Chromium is absent.
"""
import os
import shutil
import sys
import tarfile
from pathlib import Path

CHROMIUM_CMDS = ["chromium-browser", "chromium", "google-chrome"]

def chromium_exists() -> bool:
    for cmd in CHROMIUM_CMDS:
        path = shutil.which(cmd)
        if path:
            os.environ.setdefault("PYPPETEER_EXECUTABLE_PATH", path)
            print(f"Chromium available via {cmd}")
            return True
    return False


def main(res_dir: str) -> None:
    if chromium_exists():
        return

    archive = Path(res_dir) / "chromium.tar.xz"
    out_dir = Path(res_dir) / "bin"

    if archive.is_file():
        print(f"Extracting {archive} to {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            tf.extractall(out_dir)
        os.environ["PATH"] = str(out_dir) + os.pathsep + os.environ.get("PATH", "")
        cand = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
        if cand:
            os.environ.setdefault("PYPPETEER_EXECUTABLE_PATH", cand)
            print("Chromium extracted and available")
            return
        print("Extraction finished but chromium command still missing", file=sys.stderr)
    else:
        print(f"Chromium archive {archive} not found", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    resources = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "resources", "chromium")
    main(resources)
