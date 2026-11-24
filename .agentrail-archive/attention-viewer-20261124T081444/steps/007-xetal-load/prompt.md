X_eTaL reads a capture (xetal/), test-first, with the pinned X_eTaL
(bin/xetal, `just xetal`). Arrays come through the text route for now
(D10: `n_umbers []N_GET "<name>.txt"` reshaped by `<name>_shape`) and
metadata through `[]L_IST` on capture.toml (D2).

- A library (for example xetal/Capture.xtl) that loads a capture
  directory: the attention array (layer head query key), token labels,
  token ids, and the metadata keys of docs/design.md.
- Arrays are read by ONE function of the library, so moving to the
  array-file extension (ask X1) later changes that function only.
- Validate in X_eTaL: format key, shapes in the TOML, each attention
  row sums to 1 within 1e-4, upper triangle zero; a clear error
  otherwise (try broken copies of the fixture too, made under the
  test's own temporary directory).
- Selection helpers: one layer, one head, one query row.
- Reuse X_eTaL-ML's libraries (Attention's c_ausal mask, NN) rather
  than redefining them where they fit (D5, ask M2), pinned by an
  XETAL_ML_COMMIT, as the sibling repositories pin theirs; never edit
  X_eTaL-ML (D9).
- Tests run through bin/xetal with reg-rs goldens or pytest, as
  X_eTaL-libraries tests its libraries; wire them into `just check`.
