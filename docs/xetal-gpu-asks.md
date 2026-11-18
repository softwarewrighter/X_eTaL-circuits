# Asks for X_eTaL-gpu

What this repository needs from
[X_eTaL-gpu](https://github.com/softwarewrighter/X_eTaL-gpu)
(`../X_eTaL-gpu`, read only from here: its own plan takes an ask).
Statuses as in [`xetal-extensions-asks.md`](xetal-extensions-asks.md).

| #  | Status | Kind | Ask | Needed by | Workaround |
| -- | ------ | ---- | --- | --------- | ---------- |
| G1 | not yet taken (waits for X2) | refactor | When X_eTaL-extensions' shared array-file crate (ask X2 there) lands, use it in place of `components/xir/src/binary.rs`, at that repository's next move of `XETAL_EXTENSIONS_COMMIT`, so the XTLARR01 format has one implementation (D3, D8 in [`design.md`](design.md)). `pack-array`, `--bind NAME=@FILE` and `gp:b_indFile` keep their behavior | one format and one reader across the X_eTaL repositories; captures written here bound on the GPU unchanged | none needed: binary.rs works as it is; until then there are two readers of one format |
| G2 | not yet taken | note | Captures written here are f32, axes `layer head query key`, row-major, rank 4, so they can be bound as they are (`gp:b_indFile`, f32 widening to a declared f64 input). Large sweeps over them (Qwen3-4B, activation-patching grids) are a later use of the Accel operations, with results returned as reductions rather than whole T x T maps, as that repository advises while outputs cross as text | later sagas here (icl-mechanistic, circuit-discovery) | none needed yet |
