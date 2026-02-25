#!/usr/bin/env bash
set -euo pipefail
source ".venv/pyside/bin/activate"

exec python "src/App.py"
