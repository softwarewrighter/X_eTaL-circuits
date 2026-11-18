Implement the adapter's capture path (Python, adapter/), test-first.

- `capture` CLI: model (default gpt2-small), prompt text or a generated
  repeated-random-token induction prompt (seeded), output dir.
- Writes attention patterns (L x H x T x T), final logits for the last
  position, token ids and strings, plus a metadata JSON per docs/design.md
  (model id + revision, TransformerLens version, dtype, device, seed,
  prompt). Output format as decided in the xetal-ingest step.
- Deterministic: same seed -> byte-identical outputs (test it).
- Tests: shapes, every attention row sums to 1 within tolerance, causal
  mask (upper triangle zero), metadata fields present, induction prompt
  generator reproducible. Keep model-download tests marked so `just
  check` can skip them offline.
- A small committed fixture capture (a few layers/tokens, or a tiny
  model) for the X_eTaL steps to test against.
