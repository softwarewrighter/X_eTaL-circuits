#!/usr/bin/env bash
# Run the adapter's patch command: scripts/patch.sh --experiment FILE [--seeds finding|heldout|N,M] --out runs/NAME
# (needs the model extra: just setup --model)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
[[ -x "$root/adapter/.venv/bin/python" ]] || "$root/scripts/setup-adapter.sh" --model
exec "$root/adapter/.venv/bin/python" -m circuits_adapter patch "$@"
