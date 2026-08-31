# Overnight autonomous run — HANDOFF (2026-08-30)

## Current active gate

**D-ARCH-08 is CLOSED** (combined production regression passed). The decode
program reaches its clean stopping point: the remaining W2 decode GEMM gap
requires a dedicated Blackwell batched-GEMV decode kernel, which is beyond a
bounded experiment. No gate is left in progress.

## What was proven this run

- **P-ARCH-13 (CLOSED, CASE C):** K2 output geometry 128x128 -> 128x64 is
  neutral-to-regressive (symmetric 4-run CUDA-event medians: K2 stage 465.683
  vs 442.713 ms; -8.93% of the 257.137 ms residual; uninstrumented prefill
  -0.78%). Geometry is not the K2 residual driver.
- **P-ARCH-14 (CLOSED):** the smallest boundary fusion (single-slice finalize
  fused into the MMA kernel, flag-gated `ESCHA_MMA_FUSED_FINALIZE_EXPERIMENT`)
  is neutral (-0.52% of the residual); parity 16/16 and memcheck 0 errors.
  Rejected for production.
- **P-ARCH-15 (CLOSED):** proven-candidate regression (MMA default + async
  overlap + ub2048) passes the prompt x ubatch matrix; 2k/ub2048 =
  2228.75 tok/s (3.40x old baseline, 72.9% of Escha).
- **P-ARCH-16 (CLOSED):** prefill investigation closed at 919.41 ms /
  2228.75 tok/s (72.8% of Escha). Remaining delta fragmented: K2 W2 66.2%,
  non-W2 46.0%, K3 -12.2%.
- **D-ARCH-01 (CLOSED):** matched decode baseline: Bee 38.17 / 50.42 / 58.28 /
  59.77 tok/s vs Escha 60.88 / 105.28 / 247.19 / 496.36 at c=1/2/4/8 (gap
  1.6x at c=1, 8.3x at c=8). Bee decode W2 path = `gen-splitk-fp32`.
- **D-ARCH-02/03 (CLOSED):** W2 decode GEMM dominates both runtimes (~83%);
  Bee deficit concentrated in the W2 kernel family/launch structure.
- **D-ARCH-04 (CLOSED):** Bee CUDA graphs are active (1.62x at c=1); the gap
  is inside the replayed graph.
- **D-ARCH-05/06/07 (CLOSED):** the smallest decode correction
  (`ESCHA_WARP_GEMV=1`) passes parity 16/16 and improves per-step latency
  11-21% but hurts wall throughput at c=1-4 (TTFT penalty); not promoted.
  Final decode attribution: step 19.6 ms vs Escha 11.6 ms (1.69x), aggregate
  c=8 62 vs 496 tok/s (7.95x).

## What remains unknown

- The K2 W2 residual (prefill, 165 ms of the 249.9 ms remaining delta) and the
  decode W2 step deficit (1.7x) both trace to the Escha-style
  decode/staging/GEMV kernel structure; a direct-fragment decode (prefill) and
  a Blackwell batched-GEMV decode kernel (decode) were not implemented.
- Bee's non-W2 prefill residual (114.8 ms) and decode non-W2 (~2 ms/step) are
  bounded but unsplit (WSL profiling limits).

## Exact next commands / experiments

1. Prefill: implement the direct-fragment decode experiment (decode packed
   payload straight into HMMA B-fragment registers, skipping shared-B +
   ldmatrix) for the K2 prefill MMA path, then repeat the P-ARCH-13-style
   symmetric CUDA-event comparison.
2. Decode: implement a Bee `gemv_bw`-style batched decode kernel for M=1..8,
   then A/B at c=1..8 and parity.
3. Promotion: flip production defaults only after both are proven; run the
   P-ARCH-15 matrix + D-ARCH-01 baseline again on the promoted binary.

## P-ARCH-18 added (2026-08-30) — original LowGPU control CLOSED, CASE D

The original LowGPU quant (TheWegemann `Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS`,
9,570,663,040 B, SHA-256
`ad85e40a28aafd907eebb6ff6b21786b897dd750b0918427f1243d6d84ebcc72`, 851
tensors, no escha/lowgpu sidecars) runs directly in the P-ARCH-15 candidate
binary at a same-session matched 2k-prefill median of **594.037 ms /
3447.60 tok/s** (n=8) — faster than the same-session hybrid (880.076 ms /
2327.07 tok/s) and original Escha W2 (848.847 ms / 2412.69 tok/s), and faster
than the retained Escha-runtime reference (669.554 ms / 3058.75 tok/s). The
LowGPU model bypasses the escha W2 path entirely (standard GGML quant kernels,
all projections stock `MUL_MAT`); the operator path is the first responsible
difference, not file size. No kernel, loader, or model change was made.

This supersedes the earlier "reproduce the Escha K2 kernel" path as the
cheapest prefill lever: selective standard-quant substitution for W2
projections (e.g., FFN or K2 layers) is now the next bounded experiment, with
a matched quality + prefill gate. See
`docs/escha-architecture/P-ARCH-18/manifest.md` and
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-18/2026-08-30/`.

## P-ARCH-19 added (2026-08-30) — Escha mixed accumulator policy CLOSED

On the original Escha runtime at the matched 2,048-token TTFT boundary,
forcing FP32 accumulation for every prefill shard costs **553.502 ms (+88.79%
wall; 1.888x slower)**: mixed **623.380 ms / 3285.31 tok/s** vs forced FP32
**1176.882 ms / 1740.19 tok/s** (n=3 per arm, all HTTP 200 with server-verified
2048 prompt tokens; identical model/runtime/env/input; only
`ESCHA_PREFILL_ACC` changed). `ESCHA_PREFILL_ACC` has a single consumer
(`_acc_mode_for(IC)`) and a single call site (the per-call accumulator argument
of `escham_code_gemm`); `mixed` = fp16 accumulate for `IC <= 6144` (the
short-IC 5120→17408 K2/K3 population) and fp32 above, so the knob cannot
change dispatch, geometry, graphs, or workspace. The retained Escha reference
(669.554 ms / 3058.75 tok/s) was measured with the same mixed default — a
forced-FP32 Escha runtime is ~1.76-1.89x slower than that family, so the
reference's speed is largely arithmetic-policy-carried. Next: P-ARCH-20
(one-shape Bee FP16 accumulator on 5120→17408 with a quality gate) or an
explicit fp32-arithmetic parity re-target. Evidence:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-19/2026-08-30/`;
manifest: `docs/escha-architecture/P-ARCH-19/manifest.md`.

## Binaries / hashes / source state

- Control/candidate: `beellama-escha/build-cuda-parch10-async` (async overlap);
  llama-bench `32911ef90000dfc31d7149d5cf9897e7b087547f7ac87ff348d8ebbb97865edc`;
  libggml-cuda `3178e5ad91bc58594212a29a19cbee081624cf86e3201face523944bd928a614`.
- K2 128x64 experimental: `build-cuda-parch13-k2bn64` (rejected; keep for
  reference). Fused-finalize experimental: `build-cuda-parch14-fusedfin`
  (rejected; keep for reference).
- Commit: `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d` (detached, uncommitted
  worktree). `escha-moe.cu` SHA-256
  `f956cadc00e7f957d9b334230c790bed88b1bbc51fc929ee4707760768f39401` plus the
  flag-gated P-ARCH-14 kernel block and the profiler event-reuse mitigation.
- Source changes this run: `escha-moe.cu` (flag-gated fused-finalize kernel +
  host plumbing, inert without the flag), scripts (`escha-parch13-measure.sh`,
  `escha-profile-aggregate.py`, `escha-parch13-table.py`,
  `escha-compare/escha_trace_k2_audit.py`, `escha-compare/escha_decode_trace_audit.py`,
  `escha-compare/escha_decode_kernel_map.py`, `escha-decode-w2-floor.py`,
  `escha-decode-bench.py`), docs/manifests/ledger, evidence dirs.

## Rollback

- Rejected experiments are compile-flag gated (`ESCHA_MMA_SM120_K2_BN64_EXPERIMENT`,
  `ESCHA_MMA_FUSED_FINALIZE_EXPERIMENT`) or runtime opt-in
  (`ESCHA_WARP_GEMV`); removing the flag/env restores the default path.
- Profiler-only event-reuse (`escha-moe.cu:1564-1601`) is measurement-only and
  optional.

## Evidence paths

- P-ARCH-13..16: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-{13,14,15,16}/2026-08-29/`
- D-ARCH-01..08: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/D-ARCH-{01..08}/2026-08-29/`
- Manifests: `beellama-escha/docs/escha-architecture/P-ARCH-{13..16}/` and
  `D-ARCH-{01..08}/`
- Ledger: `beellama-escha/docs/escha-architecture-diff-ledger.md` (mirrored to
  GBrain `projects/beellama-escha-architecture-diff-ledger`); dated mirrors in
  `D:\CODEX WORKSPACE\gbrain-updates\`.
