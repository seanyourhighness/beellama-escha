# D-ARCH-04 — decode launch / graph / fusion comparison

**Status:** `COMPLETE — GRAPHS ALREADY LEVERAGED; GAP IS INSIDE THE REPLAYED GRAPH`
**NEXT_GATE:** `D-ARCH-05 — smallest decode correction (ESCHA_WARP_GEMV)`

## Answer

Bee CUDA graphs are captured and replayed once per decode step and deliver a
1.62x single-stream speedup (34.14 vs 21.05 tok/s at c=1); at c=4 the
concurrency-bound aggregate is unchanged (52.31 vs 51.68). The remaining
per-step gap (21.7 ms vs Escha 11.6 ms at c=1) is inside the replayed graph:
the W2 kernel family (17.9 ms floor) plus ~1,300 graph-internal launches,
versus Escha's fused `gemv_bw` graph.

Evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/D-ARCH-04/2026-08-29/`
(`summary.md`, `bee-graphs-on.json`, `bee-graphs-off.json`, server logs).
