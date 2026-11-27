# X_eTaL-circuits tasks; each recipe calls a script in scripts/.

default:
    @just --list

# Create adapter/.venv (uv) and install the adapter; pass --model for torch + TransformerLens
setup *ARGS:
    scripts/setup-adapter.sh {{ARGS}}

# Lint and test the adapter (model tests only with ADAPTER_MODEL_TESTS=1)
adapter-test:
    scripts/adapter-test.sh

# Every repository check: adapter lint + tests, markdown
check:
    scripts/check.sh

# Capture a model run: just capture --out runs/NAME [--text T | --length N] [--layers 0,5] [--safetensors]
capture *ARGS:
    scripts/capture.sh {{ARGS}}

# Regenerate the committed fixture capture (fixtures/gpt2-induction)
fixture:
    scripts/fixture.sh

# Get and build xetal at the known-good commit in XETAL_COMMIT (clone in work/xetal, binary bin/xetal)
xetal:
    @scripts/xetal.sh

# Test the X_eTaL libraries in xetal/ (needs just xetal)
xetal-test:
    scripts/xetal-test.sh

# Draw one attention head as SVG: just heatmap CAPTURE_DIR LAYER HEAD [OUT.svg] (model numbers, from 0)
heatmap *ARGS:
    scripts/heatmap.sh {{ARGS}}

# Previous-token and induction scores of every head: just scores CAPTURE_DIR [OUT_DIR]
scores *ARGS:
    scripts/scores.sh {{ARGS}}

# Draw the model map with a circuit: just map CAPTURE_DIR CIRCUIT.toml [OUT.svg]
map *ARGS:
    scripts/map.sh {{ARGS}}

# Build the static site into work/site (or OUT_DIR): every picture, drawn by X_eTaL
pages *ARGS:
    scripts/pages.sh {{ARGS}}

# Retake the README's screenshot of the live demo (images/demo.png; needs Chrome)
screenshot:
    scripts/screenshot.sh

# An experiment's metric on a clean prompt and its twin: just baseline --experiment F --seed N --out runs/NAME
baseline *ARGS:
    scripts/baseline.sh {{ARGS}}

# Each head zero- and mean-ablated: just ablate --experiment F [--seeds finding|heldout|N,M] --out runs/NAME
ablate *ARGS:
    scripts/ablate.sh {{ARGS}}

# Activation patching between the twins: just patch --experiment F [--seeds finding|heldout|N,M] --out runs/NAME
patch *ARGS:
    scripts/patch.sh {{ARGS}}

# Effects of interventions on every head, from ablation or patching runs: just effects RUNS_DIR [OUT_DIR]
effects *ARGS:
    scripts/effects.sh {{ARGS}}

# Matched random controls of each head: just controls --experiment F [--seeds finding|heldout|N,M] --out runs/NAME
controls *ARGS:
    scripts/controls.sh {{ARGS}}

# Heads that beat their random controls: just verdict ABLATION_DIR PATCH_DIR CONTROLS_DIR [OUT_DIR]
verdict *ARGS:
    scripts/verdict.sh {{ARGS}}

# Tested heads checked on held-out seeds: just confirm RUNS_PREFIX HELDOUT_PREFIX [OUT_DIR]
confirm *ARGS:
    scripts/confirm.sh {{ARGS}}

# Path patching of every edge: just pathpatch --experiment F [--seeds finding|heldout|N,M] --out runs/NAME
pathpatch *ARGS:
    scripts/pathpatch.sh {{ARGS}}

# Edge attribution of every edge: just attribute --experiment F [--seeds finding|heldout|N,M] --out runs/NAME
attribute *ARGS:
    scripts/attribute.sh {{ARGS}}

# The chosen edges path-patched with controls: just edgepatch --edges F --attribution DIR [--seeds ...] --out runs/NAME
edgepatch *ARGS:
    scripts/edgepatch.sh {{ARGS}}

# Which edges beat their random controls (X_eTaL): just edge-verdict RUNS_DIR [OUT_DIR]
edge-verdict *ARGS:
    scripts/edge-verdict.sh {{ARGS}}

# Tested edges checked on held-out seeds (X_eTaL): just edge-confirm FINDING_DIR HELDOUT_DIR [OUT_DIR]
edge-confirm *ARGS:
    scripts/edge-confirm.sh {{ARGS}}

# The edges to path-patch from attribution runs: just choose --edges F --runs DIR
choose *ARGS:
    scripts/choose.sh {{ARGS}}
