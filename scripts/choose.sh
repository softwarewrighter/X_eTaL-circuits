#!/usr/bin/env bash
# The edges to path-patch, one "sender receiver" per line: scripts/choose.sh --edges FILE --runs DIR
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
[[ -x "$root/adapter/.venv/bin/python" ]] || "$root/scripts/setup-adapter.sh"
exec "$root/adapter/.venv/bin/python" -m circuits_adapter choose "$@"
