Decide, WITH THE USER, how X_eTaL ingests model tensors, and start
docs/design.md (rules + decisions register).

- Survey ../X_eTaL (docs/lang-choices.md, the system component's file
  built-ins), ../X_eTaL-libraries and ../X_eTaL-extensions for what
  exists today: reading files, numeric arrays from text/binary, JSON,
  plotting/heatmaps, graph drawing.
- Present the options with trade-offs (e.g. a safetensors reader as an
  X_eTaL-extensions native library; a plain text/CSV array format the
  adapter also writes; something else) and ask the user to choose. Do not
  decide alone, and do not change the X_eTaL language repo in this step.
- Record the decision (D1...) in docs/design.md, along with the capture
  metadata schema (axes names and order, dtype, token strings and ids,
  model id + revision, prompt, seed).
- If the decision needs work in another repo, record it in docs/plan.md
  and insert an agentrail step for it rather than doing it here.
