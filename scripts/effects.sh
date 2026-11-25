#!/usr/bin/env bash
# The effects of interventions on every head (xetal/Effects), from a directory of
# runs, one per seed (seed-N), made by just ablate or just patch:
#   scripts/effects.sh RUNS_DIR [OUT_DIR]
# Prints each method's leading heads (ablation runs: zero, mean; patching runs:
# denoise, noise); with OUT_DIR also writes <method>.svg there.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 1 ] && [ $# -le 2 ] || { sed -n 2,6p "$0" >&2; exit 2; }
dir="$(cd "$1" && pwd)"
runs="$(find "$dir" -maxdepth 1 -type d -name 'seed-*' | sed 's/.*seed-//' | sort -n | sed "s|^|$dir/seed-|" | tr '\n' ' ' | sed 's/ $//')"
[ -n "$runs" ] || { echo "effects: no seed-N runs in $1" >&2; exit 2; }
first="${runs%% *}"
if [ -f "$first/zero_effect.txt" ]; then methods="zero mean"; else methods="denoise noise"; fi
xt() { "$root/scripts/xt" eval -e "\"ef:\" u_se< \"Effects\"
$1"; }
sep=""
for m in $methods; do
    printf '%s' "$sep"; sep=$'\n'
    printf '%s\n' "$(xt "\"$runs\" ef:r_eport \"$m\"")"
    if [ $# -eq 2 ]; then
        mkdir -p "$2"
        printf '%s\n' "$(xt "\"$runs\" ef:s_vg \"$m\"")" > "$2/$m.svg"
    fi
done
