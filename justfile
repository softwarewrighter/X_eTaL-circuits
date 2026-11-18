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
