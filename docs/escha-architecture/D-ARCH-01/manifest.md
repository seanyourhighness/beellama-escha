# D-ARCH-01 — matched decode baseline

**Status:** `COMPLETE — DIRECTLY COMPARABLE BASELINE ESTABLISHED`
**NEXT_GATE:** `D-ARCH-02 — decode operator attribution`

## Answer

Bee decode is 1.6x slower than Escha at c=1 (38.17 vs 60.88 tok/s aggregate)
and 8.3x slower at c=8 (59.77 vs 496.36 tok/s). Bee's per-stream step latency
degrades 22 -> 98 ms as concurrency grows while Escha's stays ~12-14 ms
(CUDA-graph batch capture). Bee's decode W2 path is `gen-splitk-fp32` for
every decode projection (4,000/4,000 records), not the MMA path.

## Table

| c | Bee tok/s | Escha tok/s | Bee/Escha |
|---|---:|---:|---:|
| 1 | 38.17 | 60.88 | 0.627 |
| 2 | 50.42 | 105.28 | 0.479 |
| 4 | 58.28 | 247.19 | 0.236 |
| 8 | 59.77 | 496.36 | 0.120 |

## Evidence

`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/D-ARCH-01/2026-08-29/`
(`baseline-summary.md`, `bee-decode-c1-8.json`, `escha-decode-c1-8.json`,
server logs `/tmp/bee-server-decode8.log`, `/tmp/escha-server-decode2.log`,
decode route evidence `/tmp/bee-decode-route.stderr`).

No optimization was applied in this gate. The two runtime configurations are
the production-candidate Bee build and the published Escha serving profile.
