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
        if shutil.which(cmd):
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
        with tarfile.open(archive, "r:xz") as tf:
            members = tf.getmembers()
            root_prefix = Path(members[0].name).parts[0] if members else ""
            for m in members:
                parts = Path(m.name).parts
                if parts[0] == root_prefix:
                    m.name = Path(*parts[1:]).as_posix()
            tf.extractall(out_dir, members)
        os.environ["PATH"] = str(out_dir) + os.pathsep + os.environ.get("PATH", "")
        if chromium_exists():
            print("Chromium extracted and available")
            return
        print("Extraction finished but chromium command still missing", file=sys.stderr)
    else:
        print(f"Chromium archive {archive} not found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    resources = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent / "resources" / "chromium"
    )
    main(str(resources))
