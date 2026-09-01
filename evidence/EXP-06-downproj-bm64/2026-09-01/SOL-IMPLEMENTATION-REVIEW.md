# EXP-06 Gate 2 implementation review

Commit reviewed: `cf53d803c` against parent `9a11417a1` (parent identity verified). Review was source-only and read-only; no build, benchmark, GPU, or network operation was performed.

## PLAN MATCH

- Scope: PASS. The only runtime experiment is the guarded K3 BM64 launch in `escha-moe.cu`; the other committed files are the requested static proof and result. No global BM, BN, WN, thread count, accumulator, decode, payload, shared-B, fragment-store, partial-layout, split-K, finalize, or fusion change appears.
- Candidate: PASS. It is exactly `escha_matmul_dense_tiled_mma<3,64,ESCHA_MMA_BN,false>`; `ESCHA_MMA_BN` remains 128, WN remains 2, and the block remains 256 threads.
- Mechanical fallout: PASS. The only shared kernel refactor derives CPB from BM, admits only 8 or 16 bytes, and selects `uint2` only for the 8-byte synchronous path.
- Required corrections before build/benchmark: none found.

## ROUTE/PREDICATE

The guarded predicate is exactly `use_mma && K==3 && IC==17408 && OC==5120 && n_rows==2048`. Since `use_mma` already requires non-generation, an eligible architecture, 128-aligned OC, no WMMA route, and no `ESCHA_NO_MMA`, the candidate cannot leak into generation, K2, other projections, other M values, ragged/FMA, WMMA, or unsupported architectures. CUBLAS still wins the earlier dispatch branch when requested. The profile tag is uniquely `mma-down-bm64-exp`; all non-target MMA calls retain the Stage-2 mixed-acc tags and launches. Escha-op metadata selection remains outside and unchanged, so standard Qwen routing is unaffected.

## A-STAGE + OUTPUT-STORE SAFETY

For BM64, CPB is 8. Across `tid=0..255`, `cp_m=tid/4` covers rows 0..63 and `cp_h=(tid%4)*4` covers half offsets 0,4,8,12: every `[64][16]` half is copied exactly once, with no overlap or OOB. Target global addresses are 8-byte aligned (`u` allocation, even IC stride, 32-byte K-tile stride, and 8-byte suboffset); shared `s_u` begins after a 1,536-byte payload region and has 32-byte row stride. The async path uses legal 8-byte copies and preserves commit/wait/barrier/double-buffer ordering. The SM120 synchronous fallback uses `uint2` for both initial and next tiles; it neither crosses rows nor overlaps. BM128 folds to the prior CPB=16 mapping and `uint4` path.

The FP32 candidate has MT=1 and NTT=8. The proof mirrors the actual NVIDIA `tile<16,8,float>` formulas and enumerates `(warp,lane,i,j,l)`: all 64x128 outputs occur once, with no missing, duplicate, or OOB coordinate. `l < tile_c::ne` (`ne=4`) keeps every `acc[i][j].x[l]` access in bounds, so P-ARCH-20 does not recur. The FP16 `tile_ah` store is unchanged and is not instantiated for the candidate.

## SPLIT-K/SMEM/GRID

Existing prefill slicing uses the control BM128 accounting: at M=2048, `n_tb*n_cb=16*40=640`, so integer `512/640` clamps to `n_slices=1`; the candidate also asserts 1. Partial indexing remains `((sl*n_rows+row)*OC+col)`, and the unchanged finalize consumes the same one-slice layout.

Dynamic smem is `8*24*8 + 2*64*16*2 + 128*16*2 = 9,728` bytes: payload 1,536, double-buffered A 4,096, shared B 4,096, with non-overlapping boundaries. Grid is `dim3(ceil(2048/64),5120/128,1)=dim3(32,40,1)` and block is `dim3(32,8,1)`. Exact M and OC leave no candidate row/column tails; unchanged guards remain safe for existing BM128 tail launches.

## EXPERIMENT-OFF EQUIVALENCE

Macro-off routing and launch selection remain Stage 2. The generic CPB refactor is source-visible even with the experiment off, but every existing normal-kernel BM128 instantiation resolves to CPB=16 and discards the `uint2` branch via `if constexpr`; source semantics therefore match the parent. Binary/SASS identity is not established by this review and must not be claimed until compilation comparison. The separately duplicated fused-finalize kernel was untouched.

## REQUIRED PRE-BENCH PROOFS

- Fresh macro-off and macro-on compilation; per-symbol SASS/resource evidence, including macro-off BM128 comparison, candidate CPB=8 copy instructions, <=128 registers/thread, STACK=0, LOCAL=0, zero spills, and 9,728-byte dynamic smem.
- Route census proving 128/800 candidate calls and 672 unchanged, with zero predicate/tag mismatches; boundary M=17/2047/2049/4096 and non-target families must stay control.
- Identical-input BM64/BM128 target equality and non-target/control byte equality, finite-output checks, then P2/P7 16/16 and decode-r5 regression <=2%, with no CUDA errors or hangs.
- Only after those gates: the preregistered target-family, cross-family, and nine-pair full-2K performance protocol from `SOL-PLAN.md`.

## EVIDENCE GAPS

The static JSON proves coordinate coverage, not compilation, emitted instruction width, register/local/stack/spill usage, binary equivalence, runtime routing counts, numerical equality, model quality, stability, or performance. Async 8-byte lowering and macro-off SASS equivalence remain explicit compile/SASS gates. No source defect requires correction before obtaining those proofs.

VERDICT=CONFIRM
