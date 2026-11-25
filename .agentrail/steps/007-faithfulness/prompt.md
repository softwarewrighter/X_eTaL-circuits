Faithfulness, completeness and minimality, test-first.

- Faithfulness: keep only the circuit's heads and edges (mean-ablate
  every other head's output, and every other path into the circuit's
  receivers); the fraction of the clean metric the circuit alone keeps.
- Completeness: removing the circuit from the full model vs removing it
  from the circuit; minimality: each head's contribution within the
  circuit.
- Against random subgraphs of the same size (heads and edges), on the
  finding and held-out seeds. Decisions and thresholds in design.md.
- X_eTaL report and grids; README and pages.
