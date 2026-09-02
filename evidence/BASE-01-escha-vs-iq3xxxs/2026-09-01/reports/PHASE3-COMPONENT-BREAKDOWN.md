# BASE-01 Phase 3 — Same-Runtime Component Breakdown (2026-09-01)

Method: symmetric CUDA-event operator hooks, graphs OFF, attribution only.
- ESCHA arm: existing `ESCHA_PROFILE=1` hook (rotate/matmul/epilogue per call), 3 runs × 800 lines.
- IQ3 arm: Sol-approved `GGML_OP_PROFILE` hook (env-gated, isolated build-cuda-base01-profile), 3 runs × 994 lines (all executed; host_ms>0 filter excludes any no-op/reserve pass — none present in the retained runs), 2982 calls in aggregate.
- WSL limitation documented: nsys cannot capture CUDA kernel durations (CUPTI); only runtime-API capture. Symmetric CUDA-event hooks used per approved plan.

## ESCHA per-family (3-run aggregate, ESCHA_PROFILE total_ms)
| family | calls | total ms | med/call | share |
|---|---|---|---|---|
| ffn_down | 384 | 936.5 | 2.4022 | 26.7% |
| ffn_up | 384 | 769.0 | 1.9555 | 21.9% |
| ffn_gate | 384 | 742.3 | 1.8860 | 21.2% |
| attn_qkv | 288 | 373.2 | 1.1749 | 10.6% |
| ssm_out_OR_attn_output | 384 | 279.4 | 0.7139 | 8.0% |
| attn_gate | 288 | 230.6 | 0.7841 | 6.6% |
| attn_q | 96 | 139.1 | 1.4374 | 4.0% |
| attn_kv | 192 | 39.6 | 0.2027 | 1.1% |
| TOTAL | | 3509.7 | | 100% |

## IQ3 per-family (3-run aggregate, GGML_OP_PROFILE total_ms, executed lines only)
| family | calls | total ms | med/call | share |
|---|---|---|---|---|
| ffn_down | 384 | 836.6 | 1.3198 | 25.0% |
| ffn_gate | 384 | 611.9 | 1.5491 | 18.3% |
| ffn_up | 384 | 562.5 | 1.2807 | 16.8% |
| attn_qkv | 288 | 460.8 | 0.9663 | 13.8% |
| attn_gate | 288 | 219.4 | 0.7214 | 6.6% |
| ssm_out | 288 | 191.3 | 0.4576 | 5.7% |
| attn_v | 96 | 126.9 | 0.1206 | 3.8% |
| attn_q | 96 | 109.9 | 1.1559 | 3.3% |
| ssm_beta_alpha | 576 | 81.9 | 0.0676 | 2.5% |
| lm_head | 6 | 79.4 | 12.4656 | 2.4% |
| attn_output | 96 | 45.6 | 0.4505 | 1.4% |
| attn_k | 96 | 14.1 | 0.1423 | 0.4% |
| TOTAL | | 3340.2 | | 100% |

## Graphs-off whole-run totals (llama-bench, control build)
- ESCHA graphs-off: 801.9 / 814.6 / 832.2 ms (avg ~816.2)
- IQ3 graphs-off: 590.1 / 590.1 / 590.6 ms (avg ~590.3)
- Graphs-off gap: ~225.9 ms (of the canonical 242.7 ms graphs-on gap)

## Per-arm graphs-on/off delta (orchestration/graph contribution)
- ESCHA: graphs-on 880.2 − graphs-off 816.2 = +64.0 ms
- IQ3: graphs-on 637.5 − graphs-off 590.3 = +47.2 ms
- Delta contribution to gap: (64.0 − 47.2) = +16.8 ms (graph/orchestration overhead is
  slightly larger for ESCHA in absolute terms; both arms are faster graphs-off at
  single-shot prefill — graph capture/replay does not help a one-shot 2048-token run)

## Family-gap accounting (share-scaled to graphs-off wall, per Sol Gate 1 method)
projection_gap_off = sum over families of (A_share_f × A_off − B_share_f × B_off)
with A_off = 816.2, B_off = 590.3. See Phase 4/6 report for the full table and
closure calculation. Provisional dominant positive families: ffn_up, ffn_down,
ffn_gate (FFN block), ssm_out/attn_output, attn_gate, attn_q. Negative (IQ3
slower) families: attn_kv (V), ssm_beta_alpha, lm_head — small.

Files: `profile/A-escha-profile-{1,2,3}.stderr`, `profile/B-iq3-op-profile-{1,2,3}.stderr`,
`profile/ESCHA-AGGREGATE.json`, `profile/IQ3-OPPROFILE-AGGREGATE.json`,
`profile/A-escha-graphsoff-{1,2,3}.json`, `profile/B-iq3-graphsoff-{1,2,3}.json`.
