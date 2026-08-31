# D-ARCH-06 — decode scaling (corrected path)

**Status:** `COMPLETE — WARP PATH HELPS STEP LATENCY AT ALL C; AGGREGATE SATURATES ~60-70 TOK/S`
**NEXT_GATE:** `D-ARCH-07 — final decode attribution`

## Answer

The corrected path (ESCHA_WARP_GEMV) was validated across c=1/2/4/8 (2 runs
each): per-step latency improves at every concurrency (19.6/29.5/45.2/76.5 ms
vs baseline 22.3/34.3/54.8/96.8 ms), but aggregate throughput saturates near
~60-70 tok/s and does not scale with concurrency (no decode batching in Bee).
The correction helps latency, not aggregate throughput, and only marginally at
c=8 (+4%). Saturation point: c=4-8 at ~60-70 tok/s aggregate.

All runs exit 0; no CUDA errors; parity 16/16 retained from D-ARCH-05.

Evidence: D-ARCH-05 `bee-warpgemv.json` / `bee-baseline-same-session.json`.
