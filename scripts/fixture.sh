#!/usr/bin/env bash
# Regenerate the committed fixtures, GPT-2 Small at its pinned revision, seed 0:
#   fixtures/gpt2-induction       BOS + 8 random tokens repeated (17 positions),
#                                 layers 0 and 5, both file formats
#   fixtures/gpt2-induction-full  BOS + 16 random tokens repeated (33 positions),
#                                 all 12 layers (the head scores and the model map)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/fixtures/gpt2-induction"
rm -rf "$out"
"$root/scripts/capture.sh" --out "$out" --seed 0 --length 8 --layers 0,5 --safetensors
out="$root/fixtures/gpt2-induction-full"
rm -rf "$out"
"$root/scripts/capture.sh" --out "$out" --seed 0 --length 16
# fixtures/gpt2-induction-ablation  each head zero- and mean-ablated, the finding
#                                   seeds of experiments/induction.toml (seed-N)
out="$root/fixtures/gpt2-induction-ablation"
rm -rf "$out"
"$root/scripts/ablate.sh" --experiment "$root/experiments/induction.toml" --out "$out"
# fixtures/gpt2-induction-patch     activation patching, the finding seeds (seed-N)
out="$root/fixtures/gpt2-induction-patch"
rm -rf "$out"
"$root/scripts/patch.sh" --experiment "$root/experiments/induction.toml" --out "$out"
# fixtures/gpt2-induction-controls  matched random controls, the finding seeds (seed-N)
out="$root/fixtures/gpt2-induction-controls"
rm -rf "$out"
"$root/scripts/controls.sh" --experiment "$root/experiments/induction.toml" --out "$out"
# fixtures/gpt2-induction-heldout-*  the same three on the held-out seeds
for kind in ablate patch controls; do
    name="$kind"; [ "$kind" = ablate ] && name=ablation
    out="$root/fixtures/gpt2-induction-heldout-$name"
    rm -rf "$out"
    "$root/scripts/$kind.sh" --experiment "$root/experiments/induction.toml" --seeds heldout --out "$out"
done
