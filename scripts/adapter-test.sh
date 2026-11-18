#!/usr/bin/env bash
# Lint and test the adapter. Tests marked `model` are skipped unless ADAPTER_MODEL_TESTS=1.
set -euo pipefail
cd "$(dirname "$0")/../adapter"
[[ -x .venv/bin/python ]] || ../scripts/setup-adapter.sh
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
if [[ "${ADAPTER_MODEL_TESTS:-0}" == "1" ]]; then
    .venv/bin/python -m pytest -q
else
    .venv/bin/python -m pytest -q -m "not model"
fi
