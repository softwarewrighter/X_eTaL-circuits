Attention heatmap in X_eTaL, test-first.

- Draw one layer/head's T x T attention with the built-in `[]N_GRID`
  (QD8: color scales heat, gray, diverging) with token labels on both
  axes, plus a way to choose the layer and head. Command line first
  (xetal-x); browser pages stay open (docs/design.md, Open).
- Numeric agreement: for a real GPT-2 Small induction capture, values
  read in X_eTaL equal TransformerLens's cache values within 1e-5
  relative (a test comparing a handful of (layer, head, q, k) entries,
  and the per-head row sums).
- Document in README (what it is, how to run it), every example
  executed by a test. Label it clearly as observation, not causal
  evidence.
