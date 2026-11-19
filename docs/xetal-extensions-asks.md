# Asks for X_eTaL-extensions

Native code this repository needs, asked of
[X_eTaL-extensions](https://github.com/softwarewrighter/X_eTaL-extensions),
where Rust extensions of X_eTaL live (`../X_eTaL-extensions`, read
only from here: this repository never edits another, and that
repository's own plan and saga take an ask). The decisions behind
these asks are in [`design.md`](design.md) (D1, D3, D7, D8).

Each entry: status, kind, the ask, what here needs it, and the
workaround in use. Status: **not yet taken** (filed here, to be taken
to that repository's plan), **planned** or **in progress** (in its
plan), **landed** (in the commit this repository uses), **declined**
(with its reason), **blocker** (no workaround: the steps that need it
wait, and the user is told).

| #  | Status | Kind | Ask | Needed by | Workaround |
| -- | ------ | ---- | --- | --------- | ---------- |
| X1 | not yet taken | extension | An extension that reads array files into X_eTaL: XTLARR01 and safetensors (spec below) | Saga 1 steps xetal-load, attention-heatmap, induction-score, model-map | the text-numbers route (below), adopted by the user (D10) |
| X2 | not yet taken | shared crate | `crates/xetal-arrayfile`: the XTLARR01 and safetensors reading and writing, with no ABI or device dependency, for X1 and for X_eTaL-gpu (spec below) | X1; X_eTaL-gpu (see [`xetal-gpu-asks.md`](xetal-gpu-asks.md)) | none |

## X1: the array-file extension

Modeled on `extensions/image`: an `extensions/<name>/` directory made
with `just new-ext`, a facade written with `lib/Ffi.xtlm`, reached
through `xetal-x`. Names are proposals, to be settled there (for
example the extension `arrays`, facade `Arrays.xtl`, alias `"ar:"`).

**Functions**

| Facade signature | Meaning |
| ---------------- | ------- |
| `r_ead : text -> floats` | the whole array of an f32 or f64 file, any rank (0-9), as Float |
| `a_t : text int -> floats` | one slice along the first axis, counted from 1 as X_eTaL counts (A5): `"attn.xta" ar:a_t 6` is layer 6 of 12. Only that slice is read from disk |
| `s_hape : text -> ints` | the shape, from the header alone (namespaced, so the built-in `s_hape` is unaffected) |
| `i_nts : text -> ints` | the whole array of an i64 file |
| `n_ames : text -> text` | a safetensors file's tensor names, one per line, in file order |

**Addressing.** A path names an XTLARR01 file, or a tensor inside a
safetensors file as `file.safetensors#name` (ABI V1 allows two
arguments at most, and `a_t` uses both). Paths are relative, under the
working directory or an environment root (as `XETAL_IMAGE_ROOT` is
for images), with no `..`.

**Formats, told apart by their first bytes.**

- XTLARR01, as X_eTaL-gpu's `docs/binary-arrays.md` specifies: magic,
  scalar tag (1 f32, 2 f64, 3 i64), rank 0-9, six zero bytes, u64
  dimensions, a little-endian row-major payload. Every refusal that
  document lists: version, tag, rank, reserved bytes, overflow, and
  exact payload length (truncation and extra bytes).
- safetensors: an 8-byte little-endian header length, then a JSON
  header (`dtype`, `shape`, `data_offsets` per tensor, optional
  `__metadata__`), then the data. F32 and F64 are read as Float, F16
  and BF16 widened to Float, I64 through `i_nts`. Refused, each by
  name: a header length beyond the file, header JSON that does not
  parse, offsets outside the data or not matching dtype and shape,
  overlapping tensors, an unknown dtype, and a missing tensor name.

**Errors** name the file (and the tensor), as the image extension's
do. No panic crosses the ABI.

**Tests there.** A round trip of every scalar type and of ranks 0-9;
each refusal above; slices of both formats; f16 and bf16 widening
against known values. A demo and golden read this repository's
fixture `fixtures/gpt2-induction` in both formats (copied there with
a provenance note: GPT-2 Small at revision 607a30d, made by
`just fixture` here). It checks that `attn.xta` and
`capture.safetensors#attn` give the same array, of shape
`2 12 17 17`.

**Size.** Until X_eTaL's native hook (X_eTaL-extensions' ask E1), a
call's result crosses `xetal-x`'s text channel. X_eTaL-gpu measured
that at about 0.4 s per million Floats. A whole GPT-2 Small capture
(12 x 12 x 101 x 101, about 1.5 million) is about 0.6 s, which is
acceptable. A Qwen3-4B capture (about 100 million) is not, so `a_t`
reads one layer at a time.

**Why here.** X_eTaL analyzes captures written by this repository's
Python adapter (`docs/design.md`, "Capture format"). The metadata
(`capture.toml`) is read with X_eTaL's own `[]L_IST`, so this
extension handles numbers only.

**The text-numbers route (adopted, D10).** X_eTaL already reads
numbers from text: `n_umbers []N_GET "attn.txt"`, then `r_eshape`. The
adapter writes such files beside the binary ones, each value the exact
decimal of its f64 widening, and a test with the pinned X_eTaL checks
that every value equals the binary file's. This serves GPT-2 sizes.
Qwen-sized captures still need X1. When X1 lands, X_eTaL reads the
binary files instead and the text copies stop.

## X2: the shared array-file crate

The user chose X_eTaL-extensions as its owner (D8): X_eTaL-gpu
already depends on X_eTaL-extensions (its `XETAL_EXTENSIONS_COMMIT`
pin), not the other way round.

- Standalone: no ABI, SDK or device dependency. An array type of a
  shape (`Vec<usize>`) and its data (f32, f64 or i64), row-major.
- XTLARR01: `encode`, `decode`, `read`, `write` (refusing an existing
  file, as the Python writer here does), `shape` (header only), and a
  slice along the first axis read without the rest of the payload. Its
  behavior matches X_eTaL-gpu's `components/xir/src/binary.rs` (its
  semantics, its refusals) so that repository can switch to it
  without a change in behavior.
- safetensors: list tensors, read one (whole, or a first-axis slice),
  with the dtype widening of X1. The `safetensors` crate may be used
  or the header read directly. Whichever is chosen, the refusals of X1
  are tested.
- This repository's Python implementation (`adapter/src/circuits_adapter/xta.py`,
  tests in `adapter/tests/test_xta.py`) gives byte-level cases to test
  against: the header of an f32 2 x 3 matrix, and each refusal.
