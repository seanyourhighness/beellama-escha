# D-ARCH-05 — smallest decode correction (ESCHA_WARP_GEMV)

**Status:** `COMPLETE — CORRECT BUT NOT A WALL WIN; NOT PROMOTED`
**NEXT_GATE:** `D-ARCH-06 — decode scaling of the corrected path`

## Answer

`ESCHA_WARP_GEMV=1` (existing opt-in `escha_matmul_dense_warp<K>`, 33/37
regs) passes deterministic parity P1/P2/P5/P6/P7 16/16 and improves per-step
decode latency by 11-21% at c=1-8, but the client wall aggregate is worse at
c=1-4 (TTFT 1.40 vs 0.54 s at c=1) and only +4% at c=8. The correction is
therefore not promoted; the W2 decode kernel family remains the dominant
stage, and closing the gap needs a dedicated Blackwell GEMV decode kernel.

Evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/D-ARCH-05/2026-08-29/`
(`summary.md`, `bee-warpgemv.json`, `bee-baseline-same-session.json`,
`parity-warpgemv/`).
