#!/usr/bin/env bash
# Build the static site into work/site (or OUT_DIR) from the committed captures:
# every head's heatmap, the score grids and the model map, drawn by X_eTaL.
#   scripts/pages.sh [OUT_DIR]
# Needs bin/xetal (just xetal) and adapter/.venv (just setup). GitHub Actions runs
# this and deploys the result (.github/workflows/pages.yml).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/work/site}"
cd "$root"
[[ -x adapter/.venv/bin/python ]] || scripts/setup-adapter.sh
adapter/.venv/bin/python -m pages.build "$out"
echo "$out"
