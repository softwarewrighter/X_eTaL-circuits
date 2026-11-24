# experiments/

Declarative experiment specifications: prompts, prompt pairs,
interventions and metrics, shared by the adapter and the X_eTaL
analysis. Outputs go to `runs/`, which is not committed.

## circuits/

Circuits to draw on the model map (`just map`), one TOML file each,
every value a list of strings: `name`, `model`, `source` (where the
circuit comes from), `heads` (layer.head by the model's numbers, from
0) with their `roles`, and `edges` ("a b": a feeds b) with one
`evidence` level each:

| Level | Drawn | Means |
| ----- | ----- | ----- |
| observed | dashed | an attention pattern or attribution score only |
| tested | solid | an intervention moves the margin, against matched random controls |
| confirmed | thick | also on held-out prompts |

`circuits/gpt2-induction.toml` is the previous-token head 4.11
feeding the induction heads 5.1, 5.5, 6.9, 7.2 and 7.10; every edge
is observed.
