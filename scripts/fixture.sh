#!/usr/bin/env bash
# Regenerate fixtures/gpt2-induction: GPT-2 Small at its pinned revision, seed 0,
# BOS + 8 random tokens repeated (17 positions), layers 0 and 5, both file formats.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/fixtures/gpt2-induction"
rm -rf "$out"
"$root/scripts/capture.sh" --out "$out" --seed 0 --length 8 --layers 0,5 --safetensors
