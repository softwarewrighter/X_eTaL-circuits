#!/usr/bin/env bash
# Draw one attention head of a capture as an SVG heatmap (xetal/Heatmap):
#   scripts/heatmap.sh CAPTURE_DIR LAYER HEAD [OUT.svg]
# LAYER and HEAD are the model's numbers, from 0 (head 5.1: LAYER 5, HEAD 1).
# Without OUT the SVG goes to standard output. An observation, not causal evidence.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 3 ] && [ $# -le 4 ] || { sed -n 2,5p "$0" >&2; exit 2; }
dir="$(cd "$1" && pwd)"
for n in "$2" "$3"; do
    [[ "$n" =~ ^[0-9]+$ ]] || { echo "heatmap: layer and head are numbers from 0, got $n" >&2; exit 2; }
done
svg="$("$root/scripts/xt" eval -e "\"hm:\" u_se< \"Heatmap\"
\"$dir\" hm:s_vg ($2, $3)")"
if [ $# -eq 4 ]; then printf '%s\n' "$svg" > "$4"; else printf '%s\n' "$svg"; fi
