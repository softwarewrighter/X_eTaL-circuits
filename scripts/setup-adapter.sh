#!/usr/bin/env bash
# Create adapter/.venv with uv and install the adapter.
# Usage: scripts/setup-adapter.sh [--model]   (--model adds torch + TransformerLens)
set -euo pipefail
cd "$(dirname "$0")/../adapter"
extras="dev"
if [[ "${1:-}" == "--model" ]]; then extras="dev,model"; fi
[[ -d .venv ]] || uv venv
uv pip install --python .venv/bin/python -e ".[${extras}]"
