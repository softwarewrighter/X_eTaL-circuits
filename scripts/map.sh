#!/usr/bin/env bash
# Draw the model map (xetal/Map): a row per layer, a cell per head colored by its
# attention-pattern scores and an MLP cell, with a circuit drawn over it:
#   scripts/map.sh CAPTURE_DIR CIRCUIT.toml [OUT.svg]
# CAPTURE_DIR holds every layer of a repeated random-token prompt; CIRCUIT.toml is
# a file like experiments/circuits/gpt2-induction.toml. Without OUT the SVG goes
# to standard output.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 2 ] && [ $# -le 3 ] || { sed -n 2,7p "$0" >&2; exit 2; }
dir="$(cd "$1" && pwd)"
circuit="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
[ -f "$circuit" ] || { echo "map: no circuit file $2" >&2; exit 2; }
svg="$("$root/scripts/xt" eval -e "\"mp:\" u_se< \"Map\"
\"$dir\" mp:s_vg \"$circuit\"")"
if [ $# -eq 3 ]; then printf '%s\n' "$svg" > "$3"; else printf '%s\n' "$svg"; fi
