#!/usr/bin/env bash
# Which edges beat their matched random controls (xetal/EdgeEffects), from a
# directory of edgepatch runs (seed-N each, made by just edgepatch):
#   scripts/edge-verdict.sh RUNS_DIR [OUT_DIR]
# Prints the verdicts; with OUT_DIR also writes edges-<kind>.svg there for each
# input kind (q, k, v, logits) that has a passing edge.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 1 ] && [ $# -le 2 ] || { sed -n 2,6p "$0" >&2; exit 2; }
dir="$(cd "$1" && pwd)"
runs="$(find "$dir" -maxdepth 1 -type d -name 'seed-*' | sed 's/.*seed-//' | sort -n | sed "s|^|$dir/seed-|" | tr '\n' ' ' | sed 's/ $//')"
[ -n "$runs" ] || { echo "edge-verdict: no seed-N runs in $1" >&2; exit 2; }
xt() { "$root/scripts/xt" eval -e "\"ee:\" u_se< \"EdgeEffects\"
$1"; }
printf '%s\n' "$(xt "ee:v_erdicts \"$runs\"")"
if [ $# -eq 2 ]; then
    mkdir -p "$2"
    for kind in q k v logits; do
        if [ "$(xt "\"$runs\" ee:c_ountInto \"$kind\"")" != 0 ]; then
            printf '%s\n' "$(xt "\"$runs\" ee:s_vgPassing \"$kind\"")" > "$2/edges-$kind.svg"
        fi
    done
fi
