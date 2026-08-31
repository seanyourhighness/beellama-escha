# P-ARCH-20 — Bee one-shape FP16 accumulator prototype

**Status: CLOSED — REJECTED.**

## Hypothesis

Changing only Bee's MMA accumulator for `M=2048, IC=5120, OC=17408, K=2`
from FP32 to FP16 will capture a material portion of native Escha's mixed-policy
advantage while preserving the existing packed W2 representation and kernel
structure.

## Native arithmetic mapping

- Wrapper: `_acc_mode_for(IC)` returns `1` for `IC <= 6144` under the default
  mixed policy and `0` above that threshold.
- Call site: the result is passed as the final argument to
  `torch.ops.escha.escham_code_gemm`.
- Shipped wheel SASS, matched K2 `128×64` family:
  - FP16 specialization: `64 × HMMA.16816.F16`, zero FP32 HMMA.
  - FP32 specialization: `64 × HMMA.16816.F32`, zero FP16 HMMA.
- Therefore native mixed uses true FP16 MMA accumulation, not output-only
  narrowing or a host/global cuBLAS mode.

## Bee change

Rollback-safe source specialization:

- Template: `escha_matmul_dense_tiled_mma<K,BM,BN,FP16_ACC>`.
- Compile guard: `ESCHA_MMA_FP16ACC_EXPERIMENT`.
- Runtime predicate: exactly `n_rows=2048`, `IC=5120`, `OC=17408`, `K=2`.
- Unchanged: packed codes, scales, weight layout, activation staging, async
  policy, `128×128` tile geometry, shared-B path, barriers, split/finalize,
  output layout, launch topology, and model artifact.
- Production defaults remain unchanged.

## Build and SASS verification

- Control build: `build-cuda-p20-control`.
- Experimental build: `build-cuda-p20-fp16acc`.
- Experimental symbol: 32 `HMMA.16816.F16`, 0 `.F32`.
- Existing FP32 twin: 0 `.F16`, 32 `.F32`.
- Full-library counts: experimental 147424 FP16 HMMA vs control 147392,
  exactly +32; both contain 158392 FP32 HMMA because both template variants
  remain compiled.

The first compile attempt failed due to an incorrect overload; the operand type
was corrected, the retry completed with `P20_FP16_BUILD2_OK`, and only that
successful binary was benchmarked.

## Matched 2k result

RTX 5090, identical 8.619 GB GGUF, exactly 2048 prompt tokens,
`-b 2048 -ub 2048`, F16 KV, flash attention on, n=3.

- FP32 control samples: `987.781 / 929.039 / 920.661 ms`.
  Median: **929.039 ms / 2204.43 tok/s**.
- FP16 accumulator samples: `1397.949 / 1323.860 / 1339.906 ms`.
  Median: **1339.906 ms / 1528.47 tok/s**.
- Delta: **+410.867 ms / 44.22% slower**; throughput ratio `0.6934×`.

## Decision

**REJECT.** The result is far below the `<10%` pivot threshold and strongly in
the wrong direction. Bee's current shared-B `128×128` kernel does not benefit
from an arithmetic-only FP16 accumulator conversion. Native Escha's mixed gain
therefore depends on surrounding architecture not reproduced here, plausibly
its `128×64` ownership/schedule, dependency structure, or dequant/MMA
interleaving.

The performance hard gate failed before parity, quality, and detailed
register/occupancy gates; no correctness claim is made for this rejected
candidate.

## Evidence

- Raw runs:
  `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-20/2026-08-30/{control-001,fp16acc-001}/`
- Source rollback snapshot:
  `/home/sean/beellama-escha-pre-p20-20260830.{patch,status,head}`
- Build logs: `/home/sean/p20-{control,fp16}-*.log`
- Build directories: `build-cuda-p20-{control,fp16acc}`

**NEXT_GATE: P-ARCH-21 — one-shape direct-fragment/shared-B-bypass prototype
for `M=2048, IC=5120, OC=17408, K=2`, retaining FP32 accumulation initially.**
