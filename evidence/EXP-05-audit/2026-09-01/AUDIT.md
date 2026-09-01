# EXP-05 Phase 2 — Reference escham_code_gemm mainloop audit

Date: 2026-09-01. Read-only attribution. No BeeLlama kernel changes.
Wheel: `escha-1.2.0+qwen3dense-cp312…whl` SHA-256
`735f4b7ae5c03ac11c754b2a6a4e44326a1d175139f4e87fd82b2e0f4e27c22e` (verified).
Extracted `escha/_C.cpython-312-x86_64-linux-gnu.so` sha `2c55cf7c…` to ext4.

## 1. Reproduced official runtime at M=2048

Official escha-sglang (venv /home/sean/escha-venv, sglang 0.5.15.post1,
ESCHA_PREFILL=fused, ESCHA_PREFILL_ACC=mixed) serving the canonical
`escha-w2-lowgpu-mono` safetensors dir. 2048-token prefill:
- steady TTFT **623.8 ms / 3029 tok/s** (P-ARCH-19 mixed baseline 623.380 ms —
  reproduced); warmup 877–1012 ms (CUDA-graph capture).

## 2. Template space (56 sm_120 symbols)

`escham_code_gemm_kernel<A, K, BM, BN, BK, FP16ACC, FEPI>`
- A∈{0,1,2}, K∈{2,3}, BM∈{32,64,128}, BN∈{32,64}, BK∈{2,3}, two bools.
- **Param 6 = FP16-accumulate** (proven: same shape, acc_mode 0↔1 flips bool).
- **Param 7 = FEPI** (proven: ESCHAM_GEMM_FEPI=0 flips it).
- ESCHAM_GEMM_BM→BM, ESCHAM_GEMM_BK→BK, ESCHAM_GEMM_WIDE_HAD→separate
  `escham_had_in_wide_kernel` (not code_gemm). Env is read fresh per process
  (each sweep ran in a fresh `python`).

## 3. Default kernel per family + grid/block (direct op, torch.profiler)

| family | K IC→OC | acc | kernel | grid | block | regs | smem | per-call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate | 2 5120→17408 | fp16 | `<1,2,128,64,2,T,T>` | [136,16,1] | [256] | 80 | 45,056 | 1.31 ms |
| up | 3 5120→17408 | fp16 | `<1,3,128,64,2,T,T>` | [136,16,1] | [256] | 88 | 45,056 | 1.57 ms |
| down | 3 17408→5120 | fp32 | `<1,3,128,64,2,F,T>` | [40,16,1] | [256] | 122 | 45,056 | 3.96 ms |
| qkv | 2 5120→10240 | fp16 | `<1,2,128,64,2,T,T>` | [80,16,1] | [256] | 80 | 45,056 | 0.82 ms |
| k | 2 5120→1024 | fp16 | `<1,2,32,32,3,T,T>` | [8,64,1] | [256] | 45 | 15,872 | 0.15 ms |

- CTA coverage: BM=128 rows × 2×64 = 128 output cols per CTA (grid.x = OC/128,
  grid.y = M/128) → **two independently owned 64-column bands per CTA**,
  256 threads. Confirms the goal's two-band hypothesis.
- sm_120 resources: fp16 kernels REG 63–96, fp32 REG 122–127, STACK ≤256,
  LOCAL 0 (no spills), SHARED 45,056 B (main) / 8–40 KiB (small tiles).

## 4. One-factor sweeps (fresh processes; dominant families)

gate K2 5120→17408 (fp16):
| config | kernel | per-call |
| --- | --- | --- |
| default | `<1,2,128,64,2,T,T>` | **1.34 ms** |
| BM=64 | `<1,2,64,32,3,T,T>` | 1.53 |
| BM=32 | `<1,2,32,32,3,T,T>` | 1.89 |
| BK=64 | `<1,2,128,64,2,T,T>` | 1.36 |
| BK=32 | `<1,2,128,32,3,T,T>` | 1.38 |
| FEPI=0 | `<1,2,128,64,2,T,F>` | 1.50 |
| WIDE_HAD=0 | unchanged | 1.37 |

down K3 17408→5120 (fp32):
| config | kernel | per-call |
| --- | --- | --- |
| default | `<1,3,128,64,2,F,T>` | **3.96 ms** |
| BM=64 | `<1,3,64,32,3,F,T>` | **1.99 ms** |
| BM=32 | `<1,3,32,32,3,F,T>` | 2.28 |
| BK=32 | `<1,3,128,32,3,F,T>` | 2.26 |
| FEPI=0 | `<1,3,128,64,2,F,F>` | 2.06 |
| WIDE_HAD=0 | unchanged | 4.18 |

**Finding:** the official default BM=128/BK=2 is ~2× slower on the long-IC
down_proj family than BM=64/BK=32/FEPI=0 alternatives — a real, measurable
official-runtime inefficiency on that family.

## 5. SASS comparison (sm_120, per-symbol)

| metric | Official gate K2 fp16 | Official down K3 fp32 | BeeLlama K2 fp32 | BeeLlama K3 fp32 |
| --- | ---: | ---: | ---: | ---: |
| HMMA.16816.F16 | 64 | 0 | 0 | 0 |
| HMMA.16816.F32 | 0 | 64 | 32 | 32 |
| LDS (total) | **136** | **136** | (LDSM 16) | (LDSM 16) |
| LDSM.16.M88.2 | 0 | 0 | 16 | 16 |
| STS.U16 (decoded-B store) | 0 | 0 | 16 | 16 |
| STL.128 | 16 | 32 | 0 | 0 |
| SHFL.BFLY (Hadamard) | 80 | 80 | 0 | 0 |
| WARPSYNC.COLLECTIVE | 40 | 40 | 0 | 0 |
| ENDCOLLECTIVE | 40 | 40 | 0 | 0 |
| FFMA / decode (SHF/LOP3) | 72/135 | 72/118 | — | — |

Note: official `LDS*` totals are 136 per symbol (134 unsuffixed `LDS` + 2
`LDS.128`); the audit text originally said 128 (selected subset). Corrected.
Raw one-factor sweep traces are retained under `raw-sweeps/` (fresh process
per env value, gate/up/down × BM/BK/FEPI/WIDE_HAD). No Bee K3-FP16 SASS file
was retained (Bee K3 uses fp32-acc for down_proj; the fp16 K3 kernel is
instantiated but not selected for any family — noted as a coverage gap).

- **Official:** decoded-B is consumed warp-locally immediately before MMA
  (no complete shared-B write/reload — 0 STS.U16, uses LDS/STL staging +
  warp-collective syncs); in-kernel Hadamard epilogue (80 SHFL.BFLY per
  invocation); collective schedule (40+40 WARPSYNC/ENDCOLLECTIVE). 128 LDS =
  A-fragment + payload staging. 45 KiB shared = deep A/B staging, 80 regs.
- **BeeLlama:** decodes B into shared (16 STS.U16), reloads via ldmatrix
  (16 LDSM), 32× HMMA, no in-kernel butterflies (separate finalize), no
  warp-collective syncs; 13,824 B shared, 97–128 regs.

## 6. Attribution table (aggregate matmul at 2k, direct per-call × call count)

| family | official ms | Bee-Stage2 ms | Bee-fp32 ms | off/Stage2 |
| --- | ---: | ---: | ---: | ---: |
| gate | 167.8 | 216.1 | 268.2 | **0.78×** |
| up | 200.4 | 222.8 | 270.5 | 0.90× |
| down | 506.5 | 283.9 | 285.2 | **1.78×** |
| qkv | 79.0 | 99.4 | 126.1 | 0.80× |
| k | 9.3 | 8.9 | 12.0 | 1.04× |
| **total** | **963** | **831** | **962** | **1.16×** |

Caveat: direct-call per-op times use random code tensors and no graph/overlap;
Bee profile numbers include per-op sync. Wall is the clean apples-to-apples:
official 623.8 ms vs Bee-Stage2 ~817–870 ms → official **~25% faster wall**.

**Attribution (measured):**
- Short-IC families (gate/up/qkv): official mainloop is 0.78–0.90× of Bee —
  the two-band warp-collective schedule + fp16-acc + fused epilogue is faster.
- Long-IC down_proj: official default is **1.78× slower** than Bee — its
  BM=128/BK=2 config is suboptimal there (BM=64 → 1.99 ms, near Bee's 2.22 ms).
  So the official wall win does NOT come from down_proj; it comes from the
  short-IC mainloop + fused epilogue + graphs.

**Hypotheses (unmeasured):** the two-band direct-fragment schedule may carry
the short-IC win; the in-kernel epilogue removes the fp32 partial round trip
(Stage 1 epilogue = 6.7% of projection time); graph capture accounts for
launch overhead.

## 7. History corrections (append-only)

1. **P-ARCH-09** characterized the short-IC reference symbol as FP32
   accumulating. **CORRECTION: current SASS + direct-op evidence shows the
   short-IC `escham_code_gemm_kernel<1,K,128,64,2,true,true>` uses
   HMMA.16816.F16 (fp16 accumulate)**; the fp32 twin is only selected for
   IC>6144 (down_proj) or acc_mode=0. Original entry preserved, marked
   superseded. (No P-ARCH-09 evidence dir/doc exists in the current tree; the
   claim was carried in prior audit prose. This correction supersedes it.)
2. **-b 2048 -ub 2048 supersedes the older -b 2048 -ub 512 wording** for the
   official 2048-row apples-to-apples comparison (effective GEMM M=2048,
   single 2048-token batch). Preserved, marked superseded.

## 8. Phase 3 decision — Sol review result (2026-09-01)

**Sol VERDICT=REVISE, Phase 3 gate NO-GO.**
- Audit checks: template mapping FLAG (raw fresh-process traces now retained
  under `raw-sweeps/`), grid/block PASS (inference caveat on "independently
  owned"), SASS structural PASS (LDS count corrected to 136), attribution
  PASS (gate 0.776×, up 0.899×, qkv 0.795×, down 1.784×, total 1.159×),
  down-default FLAG (raw traces now retained). Corrections (P-ARCH-09,
  -ub 2048): PASS.
- **Credible budget: NO.** The measured short-IC winning coverage
  (gate/up/qkv) saves 48.3+22.4+20.4 = **91.1 ms = 10.96%** of Stage 2's
  831.1 ms aggregate matmul — below the ≥15% bar even as an upper bound
  (official timings include fused-epilogue advantages that Phase 3 would not
  carry). Even mixing the separately tuned down_proj BM=64 result gives
  14.47%, still <15% and no longer single-variable.
- **Per the goal: do not implement speculatively. EXP-05 closes as negative
  evidence.**
- Top risks recorded (register/lifetime EXP-02 repeat, 40+40 collective
  amortization across 82–128 SMs, short-IC advantage conflating mainloop +
  fused epilogue + graphs, official down_proj default trap).

## 9. Next targets ranked by measured wall-time coverage

From Stage 1 profile (rotate 4.6% / matmul 88.6% / epilogue 6.7% of
projection time) + this audit:

1. **Fuse output rotation/scale into the GEMM epilogue** (finalize = 6.7%,
   largest single fuseable bound) — Sol PLAN READY
   (`evidence/EXP-04-nextvar/2026-09-01/NEXTVAR-PLAN.md`, warp-owned
   `_fusedfin` for n_slices==1, +8 KiB staging). Credible window 5.0–6.7%.
2. **Official-style down_proj config correction** — the official default
   BM=128/BK=2 is ~2× slower on 17408→5120 than BM=64/BK=32/FEPI=0. BeeLlama
   is already faster there (2.22 ms vs 3.96 ms official default), so the
   actionable item is a BM=64-style tile sweep for the long-IC family in
   BeeLlama, isolated as a separate single-variable experiment (note:
   EXP-03 tested 256x64; a 128x64 K3-only sweep is a different variable).
3. **Fuse input rotation into the GEMM** (rotate = 4.6%, second-largest
   fuseable bound).
4. **MLP up+gate transformed-activation sharing** only if a fresh measured
   upper bound ≥5% (currently unbudgeted).

Do not revisit EXP-02/EXP-03 fragment/tile-layout work without new profiler
evidence that overturns their negative findings. Do not begin implementation
of any next target without a fresh Sol PLAN gate.
