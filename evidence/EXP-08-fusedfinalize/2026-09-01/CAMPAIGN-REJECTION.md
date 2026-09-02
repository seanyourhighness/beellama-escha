# EXP-08 — Matched Campaign REJECTION (2026-09-01)

## Result: REJECT (decisive negative)

Canonical matched 9-pair 2K campaign (AB BA BA AB AB BA BA AB AB, graphs ON,
fresh process per trial, warmup per arm), control build-cuda-base01 vs
candidate build-cuda-exp08-fusedfin (fused-finalize, gate
ESCHA_MMA_FUSED_FINALIZE_EXPERIMENT=1):

| arm | median tok/s | median ns | CV% |
|---|---|---|---|
| CONTROL | 2319.22 | 883.057 ms | 2.90 |
| CANDIDATE | 1895.11 | 1080.673 ms | 0.73 |

- Paired latency ratios (cand/ctl): 1.2593, 1.2486, 1.2181, 1.1668, 1.2275,
  1.2315, 1.2439, 1.1543, 1.1715
- Geometric latency ratio G = 1.2129, 95% CI [1.1830, 1.2436]
- Candidate-faster count: 0/9
- **Median tok/s gain: −18.29%** (1895.1 vs 2319.2)

## Interpretation

The fused-finalize epilogue is far slower than the separate-finalize path even
though it passed every pre-timing gate (Sol code review CONFIRM, resource 96/128
regs no spills, smem 22,016 B, route 736 fused + 64 split-K 0 fallback,
P2/P7 16/16 parity, decode +0.75%). The regression is structural: the warp-pair
batch loop (8 batches x 2 named barriers per warp pair) serializes the Hadamard
epilogue across only 2 warps per pair, whereas the separate
escha_finalize_dense kernel parallelizes across the full CTA tile. The register
transpose + shuffle butterflies + pair exchange are not competitive with the
separate kernel's layout.

## Action

- Candidate source REVERTED; promoted Stage 2 default unchanged.
- No promotion, no family/parity/decode follow-up campaign needed after the
  decisive gate failure (plan gate 7: <2% = REJECT+revert; this is a large
  negative).
- EXP-08 closes as negative evidence for warp-pair-fused finalize on SM120.
- Next: T1 mainloop re-planning remains the primary path to 3000+; any
  finalize fusion retry would need a full-CTA-parallel epilogue design, not the
  warp-pair batch structure.

Files: bench/run.log, bench/noise-run/*.json, scripts/analyze-campaign.py.
