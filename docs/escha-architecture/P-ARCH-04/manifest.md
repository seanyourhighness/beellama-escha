# P-ARCH-04 — SM120 MMA production qualification

**Status:** `PASS — SM120 MMA QUALIFIED AND DEFAULTED`  
**Closed prerequisites:** `P-ARCH-01`, `P-ARCH-02`, `P-ARCH-03`  
**Qualified implementation:** `escha_matmul_dense_tiled_mma<K,128,128>`  
**New official Bee SM120 controlled baseline:** **1230.03 prompt tok/s**

## Decision

Bee's existing MMA path is qualified on NVIDIA SM120 for this model's production Escha prefill envelope when:

```text
n_rows > 16
OC % ESCHA_MMA_BN(128) == 0
cc >= GGML_CUDA_CC_TURING
cc <= GGML_CUDA_CC_BLACKWELL
ESCHA_NO_MMA is unset
```

Generation (`n_rows <= 16`) retains the established FP32 generation route. Explicit `ESCHA_NO_MMA=1` retains tiled-FMA for qualified prefill shapes. Unqualified future architectures above SM120 remain opt-in through `ESCHA_FORCE_MMA=1`. No kernel optimization was performed.

## Production model envelope

Direct safetensors inspection found 400 packed Escha projections, 272 K2 and 128 K3, across these real roles/shapes:

| Format / role | IC → OC | Count |
| --- | ---: | ---: |
| K2 FFN gate | 5120 → 17408 | 64 |
| K3 FFN up | 5120 → 17408 | 64 |
| K3 FFN down | 17408 → 5120 | 64 |
| K2 linear-attention QKV | 5120 → 10240 | 48 |
| K2 linear-attention gate/Z | 5120 → 6144 | 48 |
| K2 linear/full-attention output | 6144 → 5120 | 64 |
| K2 full-attention Q | 5120 → 12288 | 16 |
| K2 full-attention K/V | 5120 → 1024 | 32 |

All production OC values are divisible by 128. The runtime collapses these to seven unique IC/OC pairs because K2/K3 share `5120→17408` and output projections share `6144→5120`.

## Pre-change forced-MMA qualification

### Route envelope and boundaries

- Real prefill rows `17, 128, 511, 512` selected `mma-fp16` for every one of the seven runtime shape pairs.
- Prompt 513 exercised the real scheduler tail as `512 + 1`, not an invented 513-row operation.
- Rows `1, 8, 16` selected `gen-splitk-fp32`.
- With `ESCHA_FORCE_MMA=1` and `ESCHA_NO_MMA=1`, rows 17 and 512 selected `tiled-fma-fp32` for all seven shape pairs.
- All route/fallback runs exited 0; no CUDA errors or residual benchmark processes.

Evidence:

- `evidence/P-ARCH-04/2026-08-29/route-matrix-forced-001/`
- `evidence/P-ARCH-04/2026-08-29/fallback-matrix-001/`

### Stability

Three independent graph-disabled profiled runs covered prompt lengths `128, 512, 1544, 2048, 4096`:

- **40,800** `mma-fp16` calls completed.
- 2,400 legitimate generation-tail calls selected `gen-splitk-fp32`.
- 3/3 runs exited 0.
- No CUDA errors, illegal accesses, NaNs, failed calls, or residual processes were detected.

Evidence: `evidence/P-ARCH-04/2026-08-29/stability-forced-001/`.

### Correctness

The established Bee-vs-Escha/SGLang e3 reference harness ran three independent times with forced MMA:

- P1/P2/P5/P6/P7 each matched 16/16 generated-prefix tokens in every run.
- P5 exercised 1,544 prompt tokens.
- 3/3 harnesses exited 0 with empty runner stderr.

This qualifies the tested 16-token reference prefix; it is not claimed as full-continuation equality. Evidence: `evidence/P-ARCH-04/2026-08-29/parity-repeated-forced-001/`.

## Minimal source correction

File: `ggml/src/ggml-cuda/escha-moe.cu`.

Old:

```cpp
const bool mma_arch_ok = cc >= GGML_CUDA_CC_TURING
                      && (cc < GGML_CUDA_CC_BLACKWELL || getenv("ESCHA_FORCE_MMA") != nullptr);
```

New:

```cpp
const bool mma_arch_ok = cc >= GGML_CUDA_CC_TURING
                      && (cc <= GGML_CUDA_CC_BLACKWELL || getenv("ESCHA_FORCE_MMA") != nullptr);
```

This adds exactly qualified SM120 to the existing Turing–Ada default range and does not qualify architectures above SM120. `ESCHA_NO_MMA`, generation routing, ragged-output fallback, and experimental WMMA/cuBLAS opt-ins remain intact.

Final source SHA-256: `147d3690aa83987856ad16288b6c0bca24d64001212fe63f79338b940b9ba4ee`.  
Final `libggml-cuda.so` SHA-256: `f40de5db8af0a2ce5c7fac5bb417efa439ba66271dbb01e76483d2b75f68231b`.

## Post-change validation without ESCHA_FORCE_MMA

- Automatic route matrix exited 0.
- Rows `1,8,16`: `gen-splitk-fp32`.
- Rows `17,128,511,512`: `mma-fp16` across all real runtime shapes.
- `ESCHA_NO_MMA=1`: `tiled-fma-fp32` at rows 17 and 512 across all shapes.
- Three automatic-route P1/P2/P5/P6/P7 runs: 16/16 each, empty stderr.
- No detected CUDA errors or residual processes.
- Final exact-predicate smoke: rows 16 generation; rows 17/512 automatic MMA; exit 0.

Evidence:

- `evidence/P-ARCH-04/2026-08-29/postchange-qualification-001/`
- `evidence/P-ARCH-04/2026-08-29/final-predicate-smoke-001/`

## Controlled baseline

Same production benchmark contract: graph mode enabled, profiling disabled, exactly 2,048 prompt tokens, `-b 2048 -ub 512`, F16 KV, FA on, three repetitions.

| Route | Samples (tok/s) | Median |
| --- | --- | ---: |
| Old normal tiled-FMA | `666.312 / 653.131 / 655.468` | **655.468** |
| Qualified automatic SM120 MMA | `1246.82 / 1230.03 / 1217.32` | **1230.03** |

The automatic MMA baseline is approximately 1.8766x the old normal baseline. P-ARCH-03's forced-MMA median `1243.72` remains supporting evidence, not the official post-change baseline.

## Final qualification report

```text
Qualification: PASS
Qualified formats: K2 and K3 across all 400 model projections
Qualified shapes: seven runtime IC/OC pairs listed above, all OC%128==0
Qualified row/prompt envelope: prefill rows 17/128/511/512; prompts through 4096 with real 512+tail scheduling; generation <=16 remains fallback
Correctness: PASS for repeated P1/P2/P5/P6/P7 16-token reference-prefix gate
Stability: PASS — 40,800 MMA calls, 3/3 runs, no detected CUDA errors
Fallback conditions: generation <=16; ESCHA_NO_MMA; OC%128!=0; architecture >SM120 unless forced
Old dispatch: SM120 tiled-FMA unless ESCHA_FORCE_MMA
New dispatch: Turing through SM120 MMA by default when prefill/aligned/not disabled
Controlled baseline before: 655.468 tok/s tiled-FMA
Qualified MMA baseline: 1230.03 tok/s
Source change: BLACKWELL strict-less-than changed to inclusive upper bound
```

P-ARCH-04 is complete. P-ARCH-05 may begin.