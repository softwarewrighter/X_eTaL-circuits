X_eTaL reads a capture (xetal/), test-first.

- Load a capture produced by the adapter (the fixture from the previous
  step) into an X_eTaL array with axes L x H x T x T, plus the token
  labels and metadata.
- Validate in X_eTaL: shape matches metadata, each attention row sums to
  1 within tolerance, upper triangle is zero; clear error otherwise.
- Selection helpers: one layer, one head, one query row.
- Tests run through the xetal CLI (spec/golden style, following how
  X_eTaL-libraries tests its libraries); wire them into `just check`.
