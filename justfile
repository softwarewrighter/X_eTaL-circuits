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
