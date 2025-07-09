#!/usr/bin/env bash
# Bootstrap Chromium by extracting a local archive if a system install is missing.
set -e

CHROMIUM_CMDS=("chromium-browser" "chromium" "google-chrome")

for cmd in "${CHROMIUM_CMDS[@]}"; do
    if command -v "$cmd" >/dev/null 2>&1; then
        echo "Chromium already available via $cmd"
        exit 0
    fi
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_DIR="$SCRIPT_DIR/resources/chromium"
ARCHIVE="$RES_DIR/chromium.tar.xz"

if [ -f "$ARCHIVE" ]; then
    echo "Unpacking Chromium from $ARCHIVE"
    mkdir -p "$RES_DIR/bin"
    tar -xf "$ARCHIVE" -C "$RES_DIR/bin" --strip-components=1
    export PATH="$RES_DIR/bin:$PATH"
    echo "Chromium unpacked to $RES_DIR/bin"
    exit 0
fi

if command -v python3 >/dev/null 2>&1; then
    python3 "$SCRIPT_DIR/bootstrap_chromium.py" "$RES_DIR"
else
    echo "chromium not found and no Python fallback available" >&2
    exit 1
fi
