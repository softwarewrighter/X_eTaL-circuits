Text-numbers route for now (the user's choice, D10), test-first: the
adapter also writes each array as a text file X_eTaL reads today with
`n_umbers []N_GET`, until ask X1 lands.

- `<name>.txt` beside each `<name>.xta`: the same array, row-major,
  each f32 as the exact decimal of its f64 widening (Python's repr of
  float(f32)), so X_eTaL's values equal what X1 will give bit for bit;
  one line per last-axis row (one query row of attention).
- capture.toml gains `text_arrays` naming the arrays written as text.
- Tests: text parses back to the binary array exactly; line structure;
  determinism; the fixture regenerated with the text files and still
  byte-identical on regeneration.
- Check with the real X_eTaL that `n_umbers []N_GET "attn.txt"`
  reshaped by `attn_shape` gives the fixture's values (an X_eTaL
  pinned for this repository: XETAL_COMMIT and a script that builds it
  into a gitignored work/, as the sibling repositories do).
- Record D10 in docs/design.md and the route as X1's workaround in
  docs/xetal-extensions-asks.md.
