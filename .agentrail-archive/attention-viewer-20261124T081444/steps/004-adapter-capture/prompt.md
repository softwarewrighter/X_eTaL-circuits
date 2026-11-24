Implement the adapter's capture path (Python, adapter/), test-first.
Format and metadata: docs/design.md (D1-D4, D6, "Capture format").

- Pin `transformer-lens==4.4.0` in the `model` extra (it needs
  transformers>=5.9, torch>=2.6); load GPT-2 with
  `TransformerBridge.boot_transformers("gpt2", revision=...)`, revision
  pinned (docs/reference.md).
- `capture` CLI: model, prompt text or a generated repeated-random-token
  induction prompt (seeded), output directory.
- Writes a capture directory: `attn.xta` (f32, layer head query key),
  `logits.xta` (f32, the last position's logits) and `capture.toml`
  (every value an array of strings, the keys in docs/design.md).
- An XTLARR01 writer and reader in Python (numpy), tested against the
  spec in ../X_eTaL-gpu/docs/binary-arrays.md: header bytes, rank 0-9,
  row-major, refuses truncation and extra bytes; round trip.
- Deterministic: same seed -> byte-identical outputs (test it).
- Tests: shapes, every attention row sums to 1 within 1e-4, causal
  mask (upper triangle zero), TOML keys present and shapes equal to the
  .xta headers, induction prompt generator reproducible. Tests that load
  a model are marked `model`, so `just check` skips them offline.
- A small committed fixture capture under a `fixtures/` directory
  (for example the first two layers and a short prompt) for the X_eTaL
  steps to test against.
