#!/usr/bin/env bash
# Which tested heads pass again on the held-out seeds (xetal/Effects): mean
# ablation and denoising, from the finding seeds' runs (ablation, patching,
# controls) and the held-out seeds' runs of the same three:
#   scripts/confirm.sh RUNS_PREFIX HELDOUT_PREFIX [OUT_DIR]
# where PREFIX-ablation, PREFIX-patch and PREFIX-controls are directories of runs
# (seed-N), e.g. fixtures/gpt2-induction and fixtures/gpt2-induction-heldout.
# Prints each head's verdict; with OUT_DIR also writes confirmed-<method>.svg.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 2 ] && [ $# -le 3 ] || { sed -n 2,9p "$0" >&2; exit 2; }
runs() {
    local dir; dir="$(cd "$1" && pwd)"
    find "$dir" -maxdepth 1 -type d -name 'seed-*' | sed 's/.*seed-//' | sort -n \
        | sed "s|^|$dir/seed-|" | tr '\n' ' ' | sed 's/ $//'
}
xt() { "$root/scripts/xt" eval -e "\"ef:\" u_se< \"Effects\"
$1"; }
sep=""
for m in mean denoise; do
    if [ "$m" = mean ]; then kind=ablation; else kind=patch; fi
    sets="(\"$(runs "$1-$kind")\", \"$(runs "$1-controls")\", \"$(runs "$2-$kind")\", \"$(runs "$2-controls")\")"
    printf '%s' "$sep"; sep=$'\n'
    printf '%s\n' "$(xt "$sets ef:c_onfirmations \"$m\"")"
    if [ $# -eq 3 ]; then
        mkdir -p "$3"
        printf '%s\n' "$(xt "$sets ef:s_vgConfirmed \"$m\"")" > "$3/confirmed-$m.svg"
    fi
done
