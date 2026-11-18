Attention heatmap in X_eTaL, test-first.

- Render one layer/head's T x T attention as a heatmap with token labels
  on both axes, plus a layer/head selector (in whatever front end the
  xetal-ingest decision and docs/design.md settle: CLI/TUI or the live
  web demo).
- Numeric agreement: for a real GPT-2 Small induction capture, values
  read in X_eTaL equal TransformerLens's cache values (a test comparing a
  handful of (layer, head, q, k) entries, and the per-head row sums).
- Document in README (what it is, how to run it), every example
  executed by a test. Label it clearly as observation, not causal
  evidence.
