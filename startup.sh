#!/bin/bash
set -euo pipefail

source ".venv/pyside/bin/activate"

exec uvicorn App:app --reload
