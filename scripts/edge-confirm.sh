#!/usr/bin/env bash
# Which edges that beat their controls on the finding seeds do so again on the
# held-out seeds (xetal/EdgeEffects), from two directories of edgepatch runs of
# the same edges (seed-N each, made by just edgepatch, --seeds heldout for the
# second):
#   scripts/edge-confirm.sh FINDING_DIR HELDOUT_DIR [OUT_DIR]
# Prints the verdicts; with OUT_DIR also writes confirmed-<kind>.svg there for
# each input kind (q, k, v, logits) that has a confirmed edge.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 2 ] && [ $# -le 3 ] || { sed -n 2,8p "$0" >&2; exit 2; }
runs() {
    local dir; dir="$(cd "$1" && pwd)"
    find "$dir" -maxdepth 1 -type d -name 'seed-*' | sed 's/.*seed-//' | sort -n \
        | sed "s|^|$dir/seed-|" | tr '\n' ' ' | sed 's/ $//'
}
found="$(runs "$1")"; held="$(runs "$2")"
[ -n "$found" ] && [ -n "$held" ] || { echo "edge-confirm: no seed-N runs" >&2; exit 2; }
xt() { "$root/scripts/xt" eval -e "\"ee:\" u_se< \"EdgeEffects\"
$1"; }
printf '%s\n' "$(xt "ee:c_onfirmations (\"$found\", \"$held\")")"
if [ $# -eq 3 ]; then
    mkdir -p "$3"
    for kind in q k v logits; do
        if [ "$(xt "(\"$found\", \"$held\") ee:c_onfirmedInto \"$kind\"")" != 0 ]; then
            printf '%s\n' "$(xt "(\"$found\", \"$held\") ee:s_vgConfirmed \"$kind\"")" > "$3/confirmed-$kind.svg"
        fi
    done
fi
