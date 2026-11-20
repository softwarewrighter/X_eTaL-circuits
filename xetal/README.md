# xetal/

X_eTaL libraries for the analysis. Each is `xetal/<Name>/src/<Name>.xtl`,
with its tests in `xetal/<Name>/tests/`. `scripts/xt` runs the pinned
X_eTaL (`bin/xetal`, from `just xetal`) with these libraries, and with
X_eTaL-ML's, on `XETAL_PATH`:

```bash
just xetal                                   # build bin/xetal at XETAL_COMMIT
scripts/xt eval -e '"cp:" u_se< "Capture"
s_hape cp:a_ttention "fixtures/gpt2-induction"'
just xetal-test                              # the libraries' tests
```

## Capture

Reads a capture written by the adapter (`just capture`) and checks it.

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
