#!/usr/bin/env bash
# Get X_eTaL-ML's libraries at the known-good commit in XETAL_ML_COMMIT: clone
# X_eTaL-ML into work/xetal-ml (gitignored, never edited) and check the commit
# out. Only its libraries are used (Attention, NN), through XETAL_PATH (scripts/xt).
#   scripts/xetal-ml.sh                            # from GitHub
#   XETAL_ML_SOURCE=../X_eTaL-ML scripts/xetal-ml.sh   # first clone from a local checkout
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit="$(tr -d '[:space:]' < "$root/XETAL_ML_COMMIT")"
source="${XETAL_ML_SOURCE:-https://github.com/softwarewrighter/X_eTaL-ML.git}"
clone="$root/work/xetal-ml"

if [ ! -d "$clone/.git" ]; then
    mkdir -p "$root/work"
    echo "xetal-ml: cloning $source into work/xetal-ml" >&2
    git clone --quiet "$source" "$clone"
fi
if ! git -C "$clone" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$clone" fetch --quiet origin
fi
if [ "$(git -C "$clone" rev-parse HEAD)" != "$commit" ]; then
    git -C "$clone" checkout --quiet --detach "$commit"
fi
[ -z "$(git -C "$clone" status --porcelain)" ] || { echo "xetal-ml: work/xetal-ml has local changes; it must match XETAL_ML_COMMIT" >&2; exit 1; }
echo "$clone"
