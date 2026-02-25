#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$PROJECT_DIR/.venv/pyside/bin/activate"

exec python "$PROJECT_DIR/src/App.py"
