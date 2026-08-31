# P-ARCH-06 — matched prefill execution attribution

**Status:** `COMPLETE — W2 LINEAR/GEMM DOMINATES`

## Answer

**W2 dense linear/GEMM is the dominant material divergence.** Bee executes 1,600 timed 512-row dense projections, while Escha executes 400 fused `escham_code_gemm` launches. Their direct GPU durations differ by **801.749 ms**, or **82.08%** of P-ARCH-05's 976.775 ms matched-prefill deficit. No optimization was applied.

## Reconciled attribution

P-ARCH-05's uninstrumented matched wall medians remain the denominator. Direct GPU timings establish the linear row. The balance is intentionally residual; WSL Nsight Systems produced CUDA API but no GPU-kernel records for Bee, so a finer Bee split would be invented rather than measured.

| Stage | Bee total ms | Escha total ms | Delta ms | Bee/Escha | % of total deficit |
|---|---:|---:|---:|---:|---:|
| W2 dense linear / GEMM | 1318.314 | 516.565 | **801.749** | 2.552× | **82.08%** |
| Non-W2-linear prefill residual (attention, norm/activation, conversion, LM-head, launches/gaps) | 328.015 | 152.989 | 175.026 | 2.144× | 17.92% |
| **Matched full prefill** | **1646.329** | **669.554** | **976.775** | **2.459×** | **100.00%** |

`1318.314 - 516.565 = 801.749`; `328.015 - 152.989 = 175.026`; the two deltas reconcile exactly to 976.775 ms at reported precision.

## Direct linear evidence

### Bee

`ESCHA_PROFILE=1 LOWGPU_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1` profiled one exact-ID `llama-bench` run. It emits a warm-up and measured pass; only the latter 1,600 CUDA-event records were aggregated.

| Item | Invocation count | Representative shapes (`IC→OC`, rows) | Selected path | Measured total ms |
|---|---:|---|---|---:|
| W2 dense projections | 1,600 | 5120→17408, 17408→5120, 5120→12288, 5120→10240, 5120→6144, 6144→5120, 5120→1024; rows=512 | `mma-fp16`, `escha_matmul_dense_tiled_mma<K,128,128>` | 1318.314 |
| └ activation rotation | 1,600 | same | `escha_rotate_in_dense` profile stage | 57.630 |
| └ MMA body | 1,600 | same | tiled MMA profile stage | 1221.119 |
| └ epilogue/finalize | 1,600 | same | finalize profile stage | 39.582 |

Measured-pass shape totals: 5120→17408: 512 calls / 633.065 ms; 17408→5120: 256 / 279.118; 5120→10240: 192 / 126.183; 6144→5120: 256 / 109.955; 5120→6144: 192 / 92.449; 5120→12288: 64 / 59.012; 5120→1024: 128 / 18.533.

### Escha

`bench_one_batch --profile --profile-stage prefill --profile-activities GPU --profile-record-shapes` captured one prefill-only Torch Chrome trace. The fused W2 code-GEMM subset is 516.565 ms across 400 launches.

| Item | Invocation count | Representative path / geometry | Measured total ms |
|---|---:|---|---:|
| Fused W2 code-GEMM | 400 | `escham_code_gemm_kernel`; K=3/K=2, BM=128, BN=64 variants | 516.565 |
| └ K=3 `<1,3,128,64,2,false,true>` | 64 | covered prefill projection | 251.810 |
| └ K=2 `<1,2,128,64,2,true,true>` | 240 | covered prefill projection | 181.455 |
| └ K=3 `<1,3,128,64,2,true,true>` | 64 | covered prefill projection | 79.030 |
| └ K=2 `<1,2,32,32,3,true,true>` | 32 | smaller covered projection | 4.270 |

The decisive contrast is not merely a different MMA instruction: Bee has four times as many separately timed dense projection launches and 2.552× the direct W2 linear GPU time.

## Escha trace-only supporting buckets

These facts are not used to fabricate a Bee sub-bucket delta.

| Escha bucket | Invocation count | Representative kernel/path | GPU ms | % of 599.770 ms trace |
|---|---:|---|---:|---:|
| W2 linear/GEMM | 400 | `escham_code_gemm_kernel` | 516.565 | 86.13% |
| attention / hybrid GDN | 592 | Triton `_fwd_kernel` (16), GDN chunk/conv/scan helpers | 21.085 | 3.52% |
| normalization / activation | 273 | FlashInfer RMSNorm/fused-add RMSNorm, layer norm, SwiGLU | 12.606 | 2.10% |
| tensor conversion / reconstruction | 624 | CatArray copies and direct casts/copies | 36.230 | 6.04% |
| LM-head inside profiled prefill | 1 | cuBLAS `gemvx` | 1.593 | 0.27% |
| other trace kernels | 747 | RoPE, KV store, small elementwise/index/reduction helpers | 11.691 | 1.95% |
| **Trace GPU total** | **2,637** | all kernels | **599.770** | **100.00%** |

The Escha wall-minus-trace-GPU remainder is 69.784 ms. Together with its traced non-GEMM work it is retained in the 152.989 ms reconciled Escha residual. Bee's 328.015 ms residual is likewise its P-ARCH-05 wall time minus direct W2 CUDA-event time.

## Evidence and limits

Artifacts: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-06/2026-08-29/trace-001/`.

- `bee-events-stderr.log`: 3,200 records; lines 1,601–3,200 are the measured pass.
- `escha-prefill_batch1_input2048_output1_prefill.trace.json.gz`: prefill-only Torch GPU trace; matching logs and result JSONL record the configuration.
- `bee-shared-2048.nsys-rep`: retained negative evidence; WSL emitted no CUDA kernel data.
- `bee-ncu-stdout.csv`: retained negative evidence; Nsight Compute reports `ERR_NVGPUCTRPERM`.

Profilers may perturb wall time (Bee disables graphs for safe per-call CUDA events; Torch records CPU/GPU activity), so they establish operation shares and paths only. P-ARCH-05 uninstrumented medians remain authoritative for totals.

## Gate

`P-ARCH-06 CLOSED.` The W2 dense linear/GEMM stage is proven largest (801.749 ms; 82.08%).

**NEXT_GATE: P-ARCH-07 — root-cause the dominant W2 linear/GEMM divergence and identify the smallest isolated correction. Do not implement it in this gate.**
