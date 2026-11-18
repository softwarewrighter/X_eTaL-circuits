The array-file extension (XTLARR01 and safetensors) and the shared
crate, done IN ../X_eTaL-extensions by that repository's own plan and
saga. This repository never edits another (D9): the requirements are
asks X1 and X2 in docs/xetal-extensions-asks.md (and G1 in
docs/xetal-gpu-asks.md).

This step is reserved for X_eTaL-extensions. When the asks land:

- Record the landing commits in the ask ledgers (status landed) and
  in docs/design.md where the extension is named.
- Here: how to run X_eTaL with the extension (`xetal-x --ext ...`,
  the pinned X_eTaL-extensions commit, for example an
  XETAL_EXTENSIONS_COMMIT file and a script that clones and builds it
  into a gitignored work/, as X_eTaL-gpu does), and a `just` recipe.
- Then `agentrail close 5 --elsewhere X_eTaL-extensions@<commit>`.
