#!/usr/bin/env bash
# Run the adapter's capture command: scripts/capture.sh --out runs/NAME [options]
# (needs the model extra: just setup --model)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
[[ -x "$root/adapter/.venv/bin/python" ]] || "$root/scripts/setup-adapter.sh" --model
exec "$root/adapter/.venv/bin/python" -m circuits_adapter capture "$@"
