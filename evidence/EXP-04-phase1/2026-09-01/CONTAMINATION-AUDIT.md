# Phase 1 — P-ARCH-20 contamination audit (faulty half2 store)

Date: 2026-09-01. Audit of the inherited FP16-accumulator store bug and which
experiments/builds compiled or executed it.

## The bug

In `ggml/src/ggml-cuda/escha-moe.cu`, the FP16 accumulator store read
`acc16[i][j].x[l].x` inside a loop over `l < tile_c::ne` (= 4). The fp16
fragment `tile_ah` (`tile<16,4,half2>`) has `ne = 2` (two half2 registers).
For `l = 2,3` this is an out-of-bounds read, and for every `l` the `.y` lane
of each half2 is dropped. Introduced at `f61c9cc2f`; fixed at `7b1880f41`
(EXP-04 Stage 2 implementation, Sol-reviewed).

```cpp
// faulty (pre-7b1880f41):
partial[...] = FP16_ACC ? __half2float(acc16[i][j].x[l].x) : acc[i][j].x[l];
// loop bound: l < tile_c::ne (4), but acc16.ne == 2
```

## Table

| Experiment/build | Commit (escha-moe blob) | Compiled faulty store? | Executed faulty kernel? | Conclusion validity |
| --- | --- | --- | --- | --- |
| P-ARCH-20 fp16acc-001 / fp16acc-async-001 | `0b035b3a2` tree; escha-moe sha `f24b86fa…` = `f61c9cc2f` blob | **YES** (`-DESCHA_MMA_FP16ACC_EXPERIMENT=1`; SASS in `build-cuda-p20-fp16acc` shows `HMMA.16816.F16`) | **YES** (K2 5120→17408 at rows=2048 matched the single-shape predicate) | **INVALID.** −44% (1513.5 vs 2167.4 tok/s, sync) and −2.4% (2250.6 vs 2306.7, async) were measured on the faulty store. Correctness never established: both parity runs errored (`--prompt-tokens-file`) or produced no compare report. |
| P-ARCH-20 control-001 / control-async-002 | same tree; fp32 path | No (FP16_ACC=false; fp32 store `acc[i][j].x[l]` correct) | No | **VALID as controls** (2167.4 / 2306.7 tok/s) |
| EXP-02 (direct-fragment K2) | `215aa4ac3` blob `bfe0e43d…` (fp32-acc; no FP16ACC flag) | No | No (route proof: `mma-directfrag-fp32` 128 + `mma-fp16` 672; no `mma-fp16acc`) | **VALID** (−3.92% rejected; tile/fragment layout experiment, accumulator untouched) |
| EXP-03 (shared-B 256x64) | `215aa4ac3` blob `bfe0e43d…` | No | No (route proof: all `mma-fp16`) | **VALID** (−0.42% neutral; tile-aspect experiment, accumulator untouched) |
| EXP-04 Stage 1 (attribution) | `4501b3ee1` blob `bfe0e43d…` | No (no FP16ACC/MIXEDACC flag compiled) | No (all `mma-fp16` fp32-acc) | **VALID** (rotate 4.6 / matmul 88.6 / epilogue 6.7) |
| EXP-04 Stage 2 control | `build-cuda-exp04-stage2-control` (gate off, `7b1880f41` source) | No | No (fp32-acc dispatch) | **VALID** (control median 2251–2281 tok/s) |
| EXP-04 Stage 2 candidate | `build-cuda-exp04-stage2-mixedacc` (gate on, `7b1880f41` source) | **FIXED** (correct 2-half2 store) | Yes — fp16-acc families, but with the FIXED store | **VALID** (+10.0/+10.9/+10.5% median; P2/P7 16/16; SASS `.F16` proof) |

## Consequences for prior conclusions

1. **P-ARCH-20's "FP16 toggle alone loses 44.22%" is CONTAMINATED.** The
   measured fp16 route executed the faulty store. Both the magnitude of the
   slowdown and the absence of any correctness gate make the number
   unusable. It must NOT be cited as evidence that an fp16 accumulator is
   intrinsically slow in BeeLlama.
2. **ARCH-01 audit + ledger + current-state cite −44.22% to rule out an
   accumulator-only toggle.** That reasoning chain is broken by this
   contamination: the toggle's negative result came from a buggy kernel, not
   from a faithful fp16-accumulator port. (Note: ARCH-01's P-ARCH-19 official
   mixed-vs-fp32 1.888× evidence is from the official runtime, unaffected.)
3. **EXP-02 / EXP-03 negatives remain valid** (never executed the faulty
   path; their questions were tile/fragment layout, not accumulator type).
4. **EXP-04 Stage 2 stands on clean evidence** (fixed store; fp32 control
   identical; SASS and parity independently verified). No earlier decisive
   result in this phase is invalidated beyond P-ARCH-20's fp16 side.
5. This also explains why P-ARCH-20 showed a *worse* regression than any
   plausible accumulator-policy effect: the dropped `.y` lanes make each
   fp16 MMA write only half its outputs into the wrong partial positions.

## Correction to append

The ARCH-01 audit and ledger row P-ARCH-20 (and any document citing the
−44.22%) must be amended with: "The P-ARCH-20 fp16-accumulator result is
contaminated by an fp16 fragment store bug (OOB read of `tile_ah` beyond its
2 half2 elements and dropped `.y` lanes; fixed in EXP-04 Stage 2,
`7b1880f41`). It is not evidence against fp16 MMA accumulation in BeeLlama
and must not be used to rule out an accumulator-only toggle."
