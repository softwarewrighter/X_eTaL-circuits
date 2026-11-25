#!/usr/bin/env bash
# Which heads beat their matched random controls (xetal/Effects), for mean
# ablation and denoising, from three directories of runs (seed-N each) of the
# same seeds, made by just ablate, just patch and just controls:
#   scripts/verdict.sh ABLATION_DIR PATCH_DIR CONTROLS_DIR [OUT_DIR]
# Prints the passing heads; with OUT_DIR also writes passing-<method>.svg there.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ $# -ge 3 ] && [ $# -le 4 ] || { sed -n 2,6p "$0" >&2; exit 2; }
runs() {
    local dir; dir="$(cd "$1" && pwd)"
    find "$dir" -maxdepth 1 -type d -name 'seed-*' | sed 's/.*seed-//' | sort -n \
        | sed "s|^|$dir/seed-|" | tr '\n' ' ' | sed 's/ $//'
}
ablation="$(runs "$1")"; patch="$(runs "$2")"; controls="$(runs "$3")"
xt() { "$root/scripts/xt" eval -e "\"ef:\" u_se< \"Effects\"
$1"; }
sep=""
for m in mean denoise; do
    if [ "$m" = mean ]; then real="$ablation"; else real="$patch"; fi
    printf '%s' "$sep"; sep=$'\n'
    printf '%s\n' "$(xt "(\"$real\", \"$controls\") ef:v_erdicts \"$m\"")"
    if [ $# -eq 4 ]; then
        mkdir -p "$4"
        printf '%s\n' "$(xt "(\"$real\", \"$controls\") ef:s_vgPassing \"$m\"")" > "$4/passing-$m.svg"
    fi
done
