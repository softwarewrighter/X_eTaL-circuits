Model map in X_eTaL, test-first: one picture of the whole model that
highlights heads, MLPs, layers and a candidate circuit (asked for by the
user).

- Layout: one row per layer (embedding at the bottom, unembedding at the
  top); in each row one cell per attention head plus one MLP cell; the
  residual stream drawn as the spine connecting rows.
- Color any per-component score the earlier steps produce (induction,
  previous-token, later ablation and patching effects) onto the cells,
  with a scale; select a layer, head or MLP to see its numbers.
- Overlay a circuit: a named set of components and edges (for example
  previous-token head 4.11 feeding induction heads 5.1, 5.5, 6.9, 7.2,
  7.10).
- One evidence key, the same in every picture this repository draws
  (the user's choice): each component and edge carries the level it
  has reached, and its style shows it. Dashed: **observed** (an
  attention pattern or attribution score only). Solid: **tested** (an
  intervention moves the margin, against matched random controls).
  Thick: **confirmed** (also on held-out prompts). Until Saga 2 every
  edge is observed and is drawn and labeled as such. The map is then a
  running record of discoveries: the same picture, edges moving from
  dashed to solid as evidence comes in. The key is drawn on the map.
- The circuit is data (a file in experiments/), not code, so later sagas
  draw their own circuits with the same view.
- Generic pieces (grid layout, directed-graph overlay) are written so
  they could later move to X_eTaL-libraries; do not move them yet.
- Check against the Python preview in docs/reference.md's numbers.
- The pages step after this one publishes the map.
