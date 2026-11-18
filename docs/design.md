# X_eTaL-circuits -- Design

The rules the code follows and the decisions behind them. A decision
is made with the user, recorded here with its reason, and changed
only by a later decision that says what it replaces.

## Rules

1. A capture is a directory: one XTLARR01 file per array, plus
   `capture.toml` (D2, D3).
2. Arrays are stored as f32. X_eTaL reads them as Float (f64).
   Comparisons use tolerances an f32 computation can meet (D4).
3. Every capture carries the metadata of D3. A capture without it is
   refused, by the adapter's writer and by the X_eTaL reader.
4. Axis order is fixed and written in the metadata. Attention
   patterns are `layer head query key`, row-major, so a query row is
   contiguous.
5. X_eTaL reads arrays only through the extension (D1). It never
   parses binary data itself and never reads the Python environment.

## Capture format

```
runs/<name>/
  capture.toml      metadata, all values arrays of strings (D2)
  attn.xta          f32, layer head query key
  logits.xta        f32, vocab (the last position) or position vocab
```

XTLARR01, as specified in X_eTaL-gpu's `docs/binary-arrays.md`: the
8-byte magic `XTLARR01`, a scalar tag (1 f32, 2 f64, 3 i64), the rank
(0-9), 6 reserved zero bytes, one little-endian u64 per dimension, then
the little-endian payload, row-major. Truncation and extra bytes are
errors.

`capture.toml` (every value an array of strings, so X_eTaL's built-in
`"file" []L_IST "key"` reads it; numbers go through `n_umbers`):

```toml
format         = ["xetal-circuits-capture-1"]
model          = ["gpt2"]
revision       = ["607a30d783dfa663caf39e06633721c8d4cfcd7e"]
library        = ["transformer-lens 4.4.0", "torch 2.15.0", "transformers 5.23.0"]
device         = ["cpu"]
dtype          = ["float32"]
seed           = ["0"]
prompt_kind    = ["repeated-random"]     # or "text"
prompt         = ["<the exact prompt text, or empty for token prompts>"]
token_ids      = ["50256", "4023", ...]
tokens         = ["<|endoftext|>", " foo", ...]
arrays         = ["attn", "logits"]
attn_axes      = ["layer", "head", "query", "key"]
attn_shape     = ["12", "12", "101", "101"]
attn_layers    = ["0", "1", ..., "11"]       # the model layer of each attn row
logits_axes    = ["vocab"]
logits_shape   = ["50257"]
logits_position = ["100"]
```

The shapes in the TOML must equal the shapes in the `.xta` headers.
Both readers check this.

## Decisions register

| #  | Decision | Reason |
| -- | -------- | ------ |
| D1 | X_eTaL reads model tensors through a native extension in X_eTaL-extensions. It is built like the `image` extension: a Rust shared library (ABI V1) behind a typed facade written with `lib/Ffi.xtlm`. Run with `xetal-x`. Its functions: `r_ead : text -> floats` (the whole array, any rank), `a_t : text int -> floats` (one slice along the first axis, such as one layer), `s_hape : text -> ints` (the header only). | User's choice. Keeps binary parsing out of the language core. ABI V1 already passes dense row-major f64 arrays up to rank 9. One-layer slices keep a Qwen3-4B capture (about 830 MB as f64) workable. |
| D2 | Metadata lives in a TOML file next to the arrays, read with the built-ins `[]L_IST` and `[]T_ABLE`. Every value is an array of strings. | User's choice. No extension code is needed for metadata. Any tool can read it, and test fixtures diff in git. `[]L_IST` reads arrays of strings only, so scalars are one-element arrays. |
| D3 | The array file format is XTLARR01, from X_eTaL-gpu, not safetensors. The adapter writes it with a small numpy writer. The extension reuses X_eTaL-gpu's reader (`xetal_gpu_xir::binary`), moved into a crate both repositories share rather than copied. | User's choice. One binary format across the X_eTaL repositories, and GPU sessions bind the same files (`gp:b_indFile`). It is simple enough to write and check in a few lines. Python tools convert from safetensors when needed. |
| D4 | Captures are f32. Checks use f32-sized tolerances: attention rows sum to 1 within 1e-4, and agreement with TransformerLens within 1e-5 relative. | f32 is the GPU's native width (the M1 Max has no f64). TransformerLens runs GPT-2 in f32. X_eTaL-gpu found tighter tolerances unmeetable on f32 hardware. |
| D5 | In-context learning work is split with X_eTaL-ML. X_eTaL-ML keeps the teaching demos: a hand-built ICL circuit on its Attention library, a toy trained in X_eTaL, and browser pages. X_eTaL-circuits does the experiments on real models (GPT-2, Qwen) and the causal tooling, reusing X_eTaL-ML's `Attention`, `Norm` and `NN` libraries rather than redefining them. | User's choice. Avoids two copies of the same study and keeps each repository's character. |
| D7 | The extension reads **both** XTLARR01 and safetensors files, telling them apart by their first bytes (`XTLARR01`, or safetensors' 8-byte header length followed by `{`). For safetensors: list the tensor names, read a tensor by name (whole, or one first-axis slice), and its shape. f32, f64, f16 and bf16 are read as Float. The adapter still writes XTLARR01 captures (D3), and can also write the same arrays as one `capture.safetensors`. This replaces the "not safetensors" part of D3. | User's choice. Captures stay in the format GPU sessions bind, while X_eTaL can open tensors from the wider ML world (model weights, the paper authors' outputs, other tools) without a conversion step. |
| D6 | GPT-2 is loaded with TransformerLens 4.4 (`TransformerBridge.boot_transformers`), without compatibility mode unless a comparison with old `HookedTransformer` results needs it. The revision is pinned. | The default matches Hugging Face's numbers, and so the paper authors' code. The probe in `docs/reference.md` found CPU, MPS and compatibility mode in agreement on the attention-pattern scores. |

## Open

- Browser pages. Native extensions do not load in the browser
  (WebAssembly). A browser explorer would carry its data with the page,
  as X_eTaL-ML's pages do. The first views run from the command line
  (`xetal-x`).
- Ownership of the shared XTLARR01 crate (X_eTaL-extensions or
  X_eTaL-gpu). This is decided with the owners when the extension step
  runs.
