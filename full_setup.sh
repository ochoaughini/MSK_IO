#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
VENV_DIR="./venv"
CHROMIUM_BOOTSTRAP="./bootstrap_chromium.sh"
PYTHON_BOOTSTRAP="./bootstrap_pipeline.py"
CHROMIUM_ARCHIVE="./resources/chromium/chromium.tar.xz"

# --- Ensure virtualenv ---
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- Install requirements ---
pip install --upgrade pip
if ! pip install -e .[dev]; then
    pip install pydantic-settings prometheus-client httpx fastapi numpy pydicom lmdb pillow nibabel pdfminer.six pyppeteer
fi

# --- Run tests ---
./run_tests.sh

# --- Validate Chromium setup ---
if ! command -v chromium-browser &> /dev/null && [ ! -f "$CHROMIUM_ARCHIVE" ]; then
    echo "❌ Chromium not found and no local archive present: $CHROMIUM_ARCHIVE"
    echo "Please download a compatible Chromium binary and place it there."
    exit 1
fi
bash "$CHROMIUM_BOOTSTRAP"

# --- Launch pipeline ---
if [ -z "${MSK_REMOTE_URL:-}" ] || [ -z "${MSK_AUTH_TOKEN:-}" ]; then
    echo "❌ MSK_REMOTE_URL and MSK_AUTH_TOKEN must be set as environment variables."
    exit 1
fi
python "$PYTHON_BOOTSTRAP"

