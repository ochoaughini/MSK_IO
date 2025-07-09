#!/bin/bash
# Simple script to run the full test suite with correct PYTHONPATH
# Usage: ./run_tests.sh

set -e

# Ensure we're at repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH=$(pwd) pytest -v --tb=short tests/
