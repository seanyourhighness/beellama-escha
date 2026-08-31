# D-ARCH-02 — decode operator attribution

**Status:** `COMPLETE — W2 DECODE GEMM DOMINANT ON BOTH RUNTIMES`
**NEXT_GATE:** `D-ARCH-03 — decode W2 path mapping`

## Answer

W2 decode GEMM is the dominant decode stage on both runtimes (~83% of step
time) and accounts for ~82% of the Bee-vs-Escha per-step deficit.

| Decode stage | Bee ms/step | Escha ms/step | Delta | % of deficit |
|---|---:|---:|---:|---:|
| W2 K2+K3 decode GEMM | >= 17.87 | ~9.7 | ~8.2 | ~82% |
| non-W2 (attn/GDN/head/norm/copy) | <= 3.9 | ~1.9 | ~2.0 | ~18% |
| **step total** | **21.74** | **11.6** | **10.1** | **100%** |

Bee W2 = 400 projections/step x (rotate + `gen-splitk-fp32` matmul + finalize),
min-duration floor 17.874 ms/step; Escha W2 = fused `escham_gemv_bw` +
`had_in` + `had_epilogue` (83.4% of a 12-step torch trace, 384.73 ms profiled).

Evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/D-ARCH-02/2026-08-29/`
(`attribution.md`, `escha-decode-trace-001/`, and the retained Bee decode
route capture `/tmp/bee-decode-route.stderr`).
