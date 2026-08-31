# P-ARCH-11 — SM120 async-overlap broad validation and residual attribution

**Status:** `COMPLETE — K2/K3 PASS; ASYNC ROUTE IS A PRODUCTION-CORRECTION CANDIDATE`  
**NEXT_GATE:** `P-ARCH-12 — bound the remaining K2 W2 code-GEMM residual`

## Scope and invariant

P-ARCH-01 through P-ARCH-10 remain closed.  This gate changes no kernel math or
production default.  The only diagnostic source addition is `k=%d` in the
pre-existing `ESCHA_PROFILE` line, allowing K2/K3 aggregate attribution where
they share the same IC/OC dimensions.  The experimental binary is selected
only by `CMAKE_CUDA_FLAGS=-DESCHA_MMA_SM120_ASYNC_EXPERIMENT=1`; the control
has empty CUDA flags.

The experimental route retains decoded B → shared → CTA barrier → `ldmatrix` →
HMMA.  It only restores A-stage double-buffered `cp.async` overlap.

## K2/K3 measured profile matrix

Matched persisted 2,048-ID prompt; M=512 per W2 call; `-b 2048 -ub 512`, F16
KV, FA on, graphs disabled, one warm-up then one 1,600-record measured pass.
All rows selected `mma-fp16` and completed with exit 0.

| K | IC | OC | Calls | Sync ms | Async ms | Speedup | Correct | Stable | Regs |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| 2 | 5120 | 1024 | 128 | 12.442 | 7.731 | 1.609x | model gate | yes | 128 |
| 2 | 5120 | 6144 | 192 | 84.047 | 51.941 | 1.618x | model gate | yes | 128 |
| 2 | 5120 | 10240 | 192 | 112.401 | 65.286 | 1.722x | model gate | yes | 128 |
| 2 | 5120 | 12288 | 64 | 54.631 | 33.950 | 1.609x | model gate | yes | 128 |
| 2 | 5120 | 17408 | 256 | 292.268 | 165.519 | 1.766x | model gate | yes | 128 |
| 2 | 6144 | 5120 | 256 | 94.422 | 60.460 | 1.562x | model gate | yes | 128 |
| **K2 aggregate** | — | — | **1088** | **650.211** | **384.887** | **1.689x** | pass | yes | 128 |
| 3 | 5120 | 17408 | 256 | 299.179 | 166.105 | 1.801x | model gate | yes | 128 |
| 3 | 17408 | 5120 | 256 | 256.810 | 155.144 | 1.655x | model gate | yes | 128 |
| **K3 aggregate** | — | — | **512** | **555.989** | **321.249** | **1.731x** | pass | yes | 128 |
| **all W2** | — | — | **1600** | **1206.201** | **706.136** | **1.708x** | pass | yes | 128 |

The K2/K3 mapping itself is the already-closed P-ARCH-01 result; this gate
measures execution only.  The async cubin reports 128 registers/thread for
both K2/K3 MMA symbols.  Dynamic shared memory remains 13,824 B for the
representative K2 launch.  Barrier count is unchanged: 960/CTA for IC=5120,
1,152/CTA for IC=6144, and 3,264/CTA for IC=17408.

P-ARCH-12 independently re-summed the retained `ESCHA_PROFILE` measured pass
and corrected a clerical aggregate-call-count error only: the six K2 rows sum
to 1,088 calls and all W2 rows sum to 1,600.  No timing, route, correctness, or
P-ARCH-11 decision changed.

## Stability, safety, and regression envelope

- Cold process profile control and async runs exited 0; three P-ARCH-10 async
  profiles plus the P-ARCH-11 K-aware async profile completed.  No process was
  left pending and GPU usage returned to idle after each run.
- The deterministic e3 gate at temperature 0/seed 42 passed P1, P2, P5
  (1,544-token prompt), P6, and P7, each 16/16 prefix tokens.  This is the
  established logits/output proxy, not a claim of full-continuation equality.
- Async prompt matrix (`ubatch=512`) completed at M=128/512/1544/4096:
  763.38 / 1768.40 / 1604.93 / 1783.55 tok/s.
- Exact shared-ID M=2048 ubatch matrix completed twice per configuration:
  ubatch 1024: 2126.22/2106.96 tok/s (mean 2116.59); ubatch 2048:
  2179.23/2291.29 (mean 2235.26).
- No CUDA error, illegal access, NaN, launch failure, fallback, or SM120/WSL
  pending-kernel condition was found in the retained stderr/log evidence.
- Async binary SHA-256: `32911ef90000dfc31d7149d5cf9897e7b087547f7ac87ff348d8ebbb97865edc`;
  sync control SHA-256: `0283fb9a56b5544e14ac5b9bc052f8d42b27584a42477ad4963119763efe6114`.

## Fresh post-correction attribution

The Bee old/async values below are the paired P-ARCH-11 CUDA-event passes above.
Escha is the retained matched static-prefill GPU trace from P-ARCH-06 (no Escha
code/configuration changed in this gate); its trace remains the comparable
external reference.  Graph-enabled wall time is kept separate from the
graph-disabled profiling pass.

| Stage | Bee old ms | Bee async ms | Escha ms | Remaining async delta |
|---|---:|---:|---:|---:|
| K2 W2 (rotate + GEMM + epilogue) | 711.140 | 442.862 | 185.725 | **257.137** |
| K3 W2 (rotate + GEMM + epilogue) | 591.374 | 356.771 | 330.840 | 25.931 |
| **all W2** | **1302.513** | **799.633** | **516.565** | **283.068** |
| Non-W2 prefill residual (wall minus W2) | 328.015 | 363.146 | 152.989 | **210.157** |
| **full matched prefill** | **1646.329** | **1162.779** | **669.554** | **493.225** |

The paired P-ARCH-11 graph-disabled CUDA-event passes are authoritative for the
K2/K3/all-W2 rows.  Full-prefill and non-W2 rows use the graph-enabled P-ARCH-05
Bee baseline, P-ARCH-10 async full-prefill result, and P-ARCH-06 Escha trace
accounting.  The post-async residual is 1162.779 ms minus the direct P-ARCH-11
async W2 CUDA-event sum; the same accounting for Escha is P-ARCH-06.

K2 W2 is therefore the new largest directly measured residual (257.137 ms
against the matched Escha K2 subset), narrowly ahead of the 210.157 ms non-W2
wall residual.  This is measured aggregate GPU time, not extrapolation from
the former single representative call.  K3 is near the matched Escha aggregate
after overlap; no K3-specific optimization is warranted.

## Decision

**CASE A — K2 and K3 both pass and scale.**  The SM120 async-overlap path is
ready to proceed as a candidate production correction, but this gate does not
flip the production default.  P-ARCH-12 is limited to diagnosing the remaining
aggregate K2 W2 code-GEMM divergence; it must not implement an optimization.

Evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-11/2026-08-29/validation-001/`.
