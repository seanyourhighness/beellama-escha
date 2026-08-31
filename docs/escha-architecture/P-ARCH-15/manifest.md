# P-ARCH-15 — production-candidate prefill regression matrix

**Status:** `COMPLETE — CANDIDATE PASSES THE REGRESSION ENVELOPE; DEFAULT UNCHANGED`
**NEXT_GATE:** `P-ARCH-16 — final prefill attribution with the best proven configuration`

## Candidate configuration

Only individually proven corrections are combined:

- SM120 MMA dispatch default (P-ARCH-04, source default);
- SM120 A-stage async overlap (`-DESCHA_MMA_SM120_ASYNC_EXPERIMENT=1`,
  P-ARCH-10/11);
- `-ub 2048` for the matched 2k prefill (P-ARCH-11 ubatch matrix).

K2 128x64 geometry (P-ARCH-13) and fused finalize (P-ARCH-14) are rejected and
excluded. Binary: `build-cuda-parch10-async/bin/llama-bench`
(`32911ef90000dfc31d7149d5cf9897e7b087547f7ac87ff348d8ebbb97865edc`),
libggml-cuda `3178e5ad91bc58594212a29a19cbee081624cf86e3201face523944bd928a614`.

## Regression matrix (exact shared-ID prefixes, graphs ON, profiler OFF, -r 2)

| prompt | ubatch 512 | ubatch 1024 | ubatch 2048 |
|---|---:|---:|---:|
| 512 | 1480.86 | 1686.44 | 1723.18 |
| 1024 | 1691.81 | 1984.52 | 2068.16 |
| 2048 | 1720.57 | 2046.08 | **2228.75** |
| 4096 | 1717.31 | 2068.94 | **2237.62** |

All twelve cells exit 0 with zero CUDA-error/abort lines in stderr; GPU returns
idle (~1,580 MiB) after every process. The 2k/ub2048 cell reproduces the
P-ARCH-11 value (2235.26 vs 2228.75, -0.3%).

## Baselines at the matched 2k gate

| configuration | tok/s | ratio |
|---|---:|---:|
| old Bee tiled-FMA (P-ARCH-02/03) | 655.468 | 1.00x |
| Bee production default MMA-sync (P-ARCH-04) | 1230.03 | 1.88x |
| **candidate async + ub2048 (this gate)** | **2228.75** | **3.40x** |
| Escha reference (P-ARCH-05) | 3058.75 | 4.67x |

The candidate reaches 72.9% of the Escha matched 2k prefill throughput at
ub2048 (56.2% at ub512).

## Safety / route / repeatability evidence

- Route: profiled 2k/ub2048 capture (this gate) selects `mma-fp16` for
  800/800 records at rows=2048, exit 0, no fallback; ub512 route evidence is
  retained from P-ARCH-11/13 (3,200/3,200).
- Deterministic correctness: P-ARCH-13 `parity-reuse-001/base` P1/P2/P5/P6/P7
  = 16/16 on this exact binary; P-ARCH-11 parity-full retained.
- Repeatability: two samples per matrix cell plus the P-ARCH-11 cross-check
  above; no NaN/launch/illegal-access evidence in any retained stderr.
- VRAM: model 8.6 GB + context fits comfortably in the 32 GB RTX 5090; no
  workspace pressure observed (post-state ~1,580 MiB GPU memory).

Evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-15/2026-08-29/regression-matrix-001/`
(derived `shared-{512,1024,2048,4096}.ids`, per-prompt stdout/stderr,
`route-ub2048` profiled capture, run manifest).

## Decision

The proven-candidate assembly passes the prompt x ubatch regression envelope.
Production defaults remain unchanged; the candidate is the experimental
configuration for P-ARCH-16's final attribution.
