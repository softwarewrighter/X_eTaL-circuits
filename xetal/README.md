# xetal/

X_eTaL libraries for the analysis. Each is `xetal/<Name>/src/<Name>.xtl`,
with its tests in `xetal/<Name>/tests/`. `scripts/xt` runs the pinned
X_eTaL (`bin/xetal`, from `just xetal`) with these libraries, and with
X_eTaL-ML's, on `XETAL_PATH`:

```bash
just xetal                                   # build bin/xetal at XETAL_COMMIT
scripts/xt eval -e '"cp:" u_se< "Capture"
s_hape cp:a_ttention "'"$PWD"'/fixtures/gpt2-induction"'
just xetal-test                              # the libraries' tests
```

## Capture

Reads a capture written by the adapter (`just capture`) and checks it.
A capture directory `d` is given as an absolute path: X_eTaL finds a
relative data path beside the library that names it, not in the
working directory, so a relative one is refused (`error[capture]`).

| Function | Gives |
| -------- | ----- |
| `d cp:t_exts "key"` | a key of `capture.toml`: its strings, boxed |
| `d cp:i_nts "key"` | a key of `capture.toml` as integers (`"attn_shape"` gives `2 12 17 17`) |
| `cp:t_okens d` | the prompt's tokens as text, one per position |
| `d cp:r_ead "name"` | the array `attn` or `logits`, shaped by its `<name>_shape` |
| `cp:a_ttention d` | the attention array (layer x head x query x key), after checking the format, that nothing attends to a later key, and that each row sums to 1 within 0.0001 |
| `d cp:p_osition n` | where model layer n (as `attn_layers` counts, from 0) is in the capture, counted from 1 |
| `i cp:l_ayer A` | layer i of the capture: head x query x key |
| `(i, h) cp:h_ead A` | one head: query x key, a row per query |
| `(i, h, q) cp:q_uery A` | where the query at position q looks: one row over the keys |

Layers, heads and positions are counted from 1, as X_eTaL counts.
Every refusal is `error[capture]` and names the directory. The examples
in the library's `##` comments are run by `xetal doc --test`.

These are observations of where the model attends. They are not
evidence that a head matters to the output.

## Heatmap

Draws one attention head of a capture as an SVG heatmap: a row per
query token, a column per key token, each cell the weight in whole
percent on `[]N_GRID`'s heat scale, the tokens on both axes, and a
title that says it is an observation. `just heatmap` (scripts/heatmap.sh)
is its command line and makes the directory absolute.

| Function | Gives |
| -------- | ----- |
| `d hm:p_attern (layer, head)` | one head, query x key, by the model's numbers from 0 (head 5.1 is `(5, 1)`); a layer not captured or a head outside the model is an error |
| `hm:p_ercent A` | weights as whole percents, rounded half up |
| `hm:x_ml s` | text made safe inside SVG (`&`, `<`, `>`, `"` escaped) |
| `d hm:s_vg (layer, head)` | the head's heatmap as SVG text |

Layers and heads here are the model's numbers, from 0, as
TransformerLens and the papers name heads; `Capture`'s selection
functions count from 1, as X_eTaL counts. A test reruns GPT-2 Small
and checks the values drawn against TransformerLens's cache within
1e-5 relative (`ADAPTER_MODEL_TESTS=1 just xetal-test`).

## Scores

Previous-token and induction scores of every head, from a capture of
a repeated random-token prompt (a beginning token, then t random
tokens, then the same t again) with all layers. `just scores`
(scripts/scores.sh) prints the leading heads and draws both grids.

| Function | Gives |
| -------- | ----- |
| `sc:p_revious d` | layer x head: each head's mean weight from a position to the one before it |
| `sc:i_nduction d` | layer x head: each head's mean weight from a token of the second copy to the token after its earlier occurrence; a prompt that does not repeat is an error |
| `d sc:r_anked S` | every head, best score first, a row (layer, head) each by the model's numbers |
| `sc:r_eport d` | the leading heads of each score as text, in percent |
| `d sc:s_vg "induction"` | a score's layer x head grid as SVG (`"induction"` or `"previous"`) |

These are attention-pattern scores: candidates, not evidence that a
head matters. Tests check them against numpy on the binary capture,
against TransformerLens's cache (within 1e-5 relative), and that the
known GPT-2 Small heads lead (induction 5.1, 5.5, 6.9, 7.2, 7.10;
previous token 4.11).

`Heatmap` also exports `hm:g_rid (title, note, rows, cols, P)`: any
labeled Int matrix drawn as such a grid.

## Map

The model map: a row per layer (the first at the bottom), a cell per
head colored by its scores (blue induction, orange previous token,
deeper for higher, gray under 20%; the numbers on hover), an MLP cell
per layer, the residual stream, and a circuit from a TOML file in
`experiments/circuits/` drawn over it, each edge dashed (observed),
solid (tested) or thick (confirmed). `just map` (scripts/map.sh) is
its command line and makes both paths absolute.

| Function | Gives |
| -------- | ----- |
| `d mp:s_vg c` | the map of capture d with circuit file c, as SVG |
| `mp:f_ill (previous, induction)` | a head's color from its two scores |
| `mp:h_ex v` | three channels 0-255 as `#rrggbb` |

Each circuit head is outlined by its `head_evidence` (dashed observed,
solid tested, thick confirmed), with its `head_basis` on hover.

A circuit is refused (`error[map]`) when it is for another model, has
a head evidence level other than the three or not one per head,
an evidence level other than the three or not one per edge, names a
layer not captured or a head outside the model, or has an edge that
does not go to a later layer.

## Effects

The effects of interventions on every head, from the adapter's runs
(`just ablate`, `just patch`; one run directory per seed). A set of
runs is one text: the seed directories' absolute paths separated by
spaces (`just effects` makes it). Methods: `"zero"` and `"mean"`
(ablation: percent of the clean logit difference lost), `"denoise"` and
`"noise"` (patching: percent of the clean-twin gap restored or
destroyed).

| Function | Gives |
| -------- | ----- |
| `D ef:e_ffects "method"` | seeds x layer x head, in percent |
| `ef:m_ean S`, `ef:s_pread S` | the mean and standard deviation over the seeds |
| `ef:r_anked M` | every head, largest effect first, a row (layer, head) each |
| `D ef:r_eport "method"` | the leading heads with mean and spread, as text |
| `D ef:s_vg "method"` | the mean effects as a layer x head grid on the diverging scale |
| `d ef:r_ead "name"`, `d ef:k_ey "key"` | one run's array or run.toml key |

`Heatmap` also exports `hm:g_ridOn (scale, title, note, rows, cols, P)`:
`g_rid` on any `[]N_GRID` scale. These effects are not yet compared
with random controls.
