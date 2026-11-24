#!/usr/bin/env bash
# Test the X_eTaL libraries in xetal/ (pytest driving scripts/xt). Needs bin/xetal
# (just xetal) and work/xetal-ml (scripts/xetal-ml.sh); without bin/xetal they skip.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
[[ -x adapter/.venv/bin/python ]] || scripts/setup-adapter.sh
adapter/.venv/bin/python -m ruff format --config adapter/pyproject.toml --check xetal
adapter/.venv/bin/python -m ruff check --config adapter/pyproject.toml xetal
# Tests marked `model` (TransformerLens agreement) run only with ADAPTER_MODEL_TESTS=1.
if [[ "${ADAPTER_MODEL_TESTS:-0}" == "1" ]]; then
    adapter/.venv/bin/python -m pytest -q -p no:cacheprovider --rootdir=. xetal
else
    adapter/.venv/bin/python -m pytest -q -p no:cacheprovider --rootdir=. -m "not model" xetal
fi
