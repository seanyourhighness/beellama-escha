# BASE-01 Phase 3b — Family-Gap Accounting (2026-09-01)

Method (Sol Gate 1 approved): symmetric graphs-off profiling; per-family shares
from CUDA-event hooks; share-scaled to each arm's measured graphs-off wall total
(A_off=816.2 ms avg, B_off=590.3 ms avg); canonical gap reconciled with measured
per-arm graphs-on/off deltas. Profiled totals include per-call event-sync
overhead (ESCHA ~1169.9 ms/run vs 816 wall; IQ3 ~1113.4 vs 590 wall), so shares
are relative; scaling to wall is the approved approximation. Non-projection work
(norms/attention/SSM/RoPE/embeddings) is therefore folded into the share scaling;
the <5% remainder rule is met by construction here, and Phase 4's depth matrix
provides the independent M-scaling check.

## Share-scaled family table (graphs-off boundary)

graphs-off gap = 225.9 ms (A 816.2 - B 590.3)

| family | A ms | B ms | gap ms | % of gap |
|---|---|---|---|---|
| ffn_up | 178.8 | 99.4 | +79.4 | +35.2% |
| ffn_down | 217.8 | 147.9 | +69.9 | +31.0% |
| ffn_gate | 172.6 | 108.1 | +64.5 | +28.6% |
| ssm_out_OR_attn_output | 65.0 | 41.9 | +23.1 | +10.2% |
| attn_kv (V) | 9.2 | 24.9 | -15.7 | -7.0% |
| attn_gate | 53.6 | 38.8 | +14.9 | +6.6% |
| ssm_beta_alpha (B-only) | 0.0 | 14.5 | -14.5 | -6.4% |
| lm_head (B-only) | 0.0 | 14.0 | -14.0 | -6.2% |
| attn_q | 32.3 | 19.4 | +12.9 | +5.7% |
| attn_qkv | 86.8 | 81.4 | +5.3 | +2.4% |
| PROJECTION GAP | | | +225.9 | 100.0% |

Top-3 FFN block (ffn_up + ffn_down + ffn_gate): +213.8 ms = **94.6% of the
positive gap** — and these are exactly the SAME-weights families (corr 0.83-0.87),
so the gap is a quantization-path/operator deficit, not a weight difference.
The differing-weight families (attn_gate, ssm_out, qkv-v) contribute small
positive and negative offsets that largely cancel.

## Canonical reconciliation

- graphs-on gap (measured, Phase 2): 242.7 ms
- graphs-off gap (measured): 225.9 ms
- per-arm graph delta: A +64.0 ms, B +47.2 ms -> delta contribution +16.8 ms
- reconciled = 225.9 + 16.8 = **242.7 ms** (exact match to measured canonical gap)

Interpretation: ~93% of the canonical gap is projection execution (graphs-off
boundary); ~7% is graph/orchestration overhead difference (both arms are faster
graphs-off on a one-shot 2048 run; ESCHA pays a slightly larger graph penalty).

Caveat (stated honestly): "projection gap = 100% of graphs-off gap" is by
construction of the share scaling. The independent check is Phase 4 (gap vs M)
plus the graphs-off total reconciliation above.

Files: `profile/ESCHA-AGGREGATE.json`, `profile/IQ3-OPPROFILE-AGGREGATE.json`,
`scripts/account-family-gap.py`.
