Build the array-file extension (XTLARR01 and safetensors, D7) in ../X_eTaL-extensions (D1, D3
in docs/design.md). The work lands in that repository and follows ITS
CLAUDE.md, conventions and saga (add a step to its saga for it); this
step here records it and its commits.

- Extension (name to agree there, for example `arrays` with facade
  `Arrays.xtl`, alias "ar:"), modeled on extensions/image:
  `r_ead : text -> floats` (whole array, any rank),
  `a_t : text int -> floats` (one slice along the first axis),
  `s_hape : text -> ints` (header only). f32 and f64 files read as
  Float; safetensors (D7): the same functions on a named tensor, a
  way to list a file's tensor names, f32/f64/f16/bf16 read as Float;
  the format told apart by its first bytes. Paths relative, no `..`, as the image extension does.
- Reuse X_eTaL-gpu's reader (`xetal_gpu_xir::binary`): propose moving
  it into a crate both repositories share, rather than copying it.
  Agree ownership with the user, record the result in docs/design.md
  (Open) here, and file an ask in the owning repository's asks ledger
  if the move has to happen there.
- Tests there: round trip, every refusal the XTLARR01 spec lists,
  malformed safetensors headers, bf16/f16 widening, slices of both, a
  demo reading this repository's fixture capture in both formats.
- Here: note in docs/design.md how to run X_eTaL with the extension
  (`xetal-x --ext ...`), and add a `just` recipe or script that finds it.
