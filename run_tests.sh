#!/bin/bash
# Run the full test suite with the proper environment.
# Usage: ./run_tests.sh [pytest-args]

set -e

# Ensure we're at repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

pip install -q -e ".[cli,dev]" pydantic-settings prometheus-client >/dev/null
PYTHONPATH=$(pwd) pytest -v --tb=short tests/ "$@"

