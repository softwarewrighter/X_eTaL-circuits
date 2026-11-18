X_eTaL reads a capture (xetal/), test-first, through the array-file
extension (D1) and `[]L_IST` for capture.toml (D2).

- A library (for example xetal/Capture.xtl) that loads a capture
  directory: the attention array (layer head query key), token labels,
  token ids, and the metadata keys of docs/design.md.
- Validate in X_eTaL: format key, shapes in the TOML equal the .xta
  shapes, each attention row sums to 1 within 1e-4, upper triangle zero;
  a clear error otherwise (try the broken fixtures too).
- Selection helpers: one layer, one head, one query row; one layer read
  with the extension's slice function.
- Reuse X_eTaL-ML's libraries (Attention's c_ausal mask, NN) rather
  than redefining them (D5); pin the commits used, as X_eTaL-ML pins
  its own dependencies.
- Tests run through the X_eTaL CLI with reg-rs goldens, as
  X_eTaL-libraries tests its libraries; wire them into `just check`.
