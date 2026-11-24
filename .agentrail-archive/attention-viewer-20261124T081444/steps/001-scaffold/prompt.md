Scaffold the repository for Saga 1 (see docs/plan.md, "Saga 1").

- Layout: `adapter/` (Python, uv project: pyproject.toml, src/, tests/;
  TransformerLens, torch, safetensors, pytest), `xetal/` (X_eTaL programs
  and libraries for the analysis), `experiments/` (declarative experiment
  specs), `runs/` (outputs; gitignored), `scripts/`.
- Python: always `uv venv` + `uv pip install`; run via `.venv/bin/python`
  or `uv run`; never the system python3.
- `justfile` with recipes that call scripts: `just check` (ruff + pytest
  for the adapter, sw-markdown-checker on docs/*.md), `just adapter-test`.
- `.gitignore`: .venv, runs/, caches, model weights, *.safetensors outside
  test fixtures.
- Add a repo-specific section to CLAUDE.md OUTSIDE the agentrail markers
  (project overview, layout, build/test commands, rules: American
  spellings, ASCII-only markdown, observation-is-not-causation, record
  model revision/seed/dtype on every run, user-facing docs say what/how
  never when/plans), modeled on ../X_eTaL/CLAUDE.md.
- A trivial pytest that imports the adapter package, so `just check` is
  green. Do not download models in this step.
