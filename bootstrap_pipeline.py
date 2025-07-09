#!/usr/bin/env python3
"""Bootstrap MSK_IO under constrained environments.

This helper validates the bundled Chromium archive, extracts it via
``bootstrap_chromium.sh`` when no system binary is found, and launches the
pipeline with environment variables for remote OHIF access.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

CHROMIUM_SCRIPT = Path(__file__).with_name("bootstrap_chromium.sh")
CHROMIUM_ARCHIVE = Path(__file__).parent / "resources" / "chromium" / "chromium.tar.xz"
CHROMIUM_CMDS = ["chromium-browser", "chromium", "google-chrome"]


def chromium_available() -> bool:
    for cmd in CHROMIUM_CMDS:
        if shutil.which(cmd):
            print(f"Chromium available via {cmd}")
            return True
    return False


def ensure_chromium() -> None:
    if chromium_available():
        return
    if not CHROMIUM_ARCHIVE.is_file():
        raise FileNotFoundError(f"{CHROMIUM_ARCHIVE} not found")
    print("Extracting bundled Chromium")
    subprocess.run([str(CHROMIUM_SCRIPT)], check=True)
    if not chromium_available():
        raise RuntimeError("Chromium extraction failed")


def run_pipeline(remote_url: str, auth_token: str) -> None:
    env = os.environ.copy()
    env["MSK_REMOTE_URL"] = remote_url
    env["MSK_AUTH_TOKEN"] = auth_token
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "msk_io.cli", "run"], check=True, env=env)


def main() -> None:
    remote_url = os.environ.get("MSK_REMOTE_URL")
    auth_token = os.environ.get("MSK_AUTH_TOKEN")
    if not remote_url or not auth_token:
        print("MSK_REMOTE_URL and MSK_AUTH_TOKEN must be set", file=sys.stderr)
        sys.exit(1)
    ensure_chromium()
    run_pipeline(remote_url, auth_token)


if __name__ == "__main__":
    main()
