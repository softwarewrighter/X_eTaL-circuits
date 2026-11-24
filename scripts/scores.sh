#!/usr/bin/env bash
# Previous-token and induction scores of every head (xetal/Scores), from a capture
# of a repeated random-token prompt (just capture --length N, all layers):
#   scripts/scores.sh CAPTURE_DIR [OUT_DIR]
# Prints the leading heads of each score; with OUT_DIR also writes induction.svg
# and previous.svg there. Attention-pattern scores: candidates, not causal evidence.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 1 ] && [ $# -le 2 ] || { sed -n 2,6p "$0" >&2; exit 2; }
dir="$(cd "$1" && pwd)"
xt() { "$root/scripts/xt" eval -e "\"sc:\" u_se< \"Scores\"
$1"; }
printf '%s\n' "$(xt "sc:r_eport \"$dir\"")"
if [ $# -eq 2 ]; then
    mkdir -p "$2"
    for name in induction previous; do
        printf '%s\n' "$(xt "\"$dir\" sc:s_vg \"$name\"")" > "$2/$name.svg"
    done
fi
