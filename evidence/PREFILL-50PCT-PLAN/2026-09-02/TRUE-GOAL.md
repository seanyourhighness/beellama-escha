# ESCHA-W2 — Sean's true delivery goal (corrected framing)

Date: 2026-09-02

## The actual goal (Sean, verbatim intent)

Deliver a **smart, fast, capable escha hybrid around 8.5 GB** on BeeLlama that:
1. Meets original escha W2 in SGLang in speed (~3030 tok/s warm / official
   runtime, 2048 prefill)
2. Is hopefully close to native GGUF speeds in BeeLlama (~3339 tok/s LowGPU)
3. Works with all BeeLlama native features (KVarN 3/2, DFlash2, 80K ctx, <12 GB)
4. Beats the current LowGPU-based release candidate (80K ctx, DFlash2 + KVarN
   3/2, under 12 GB)

The MMA-ready sidecar remains the core push — when a solution is found for its
consumer architecture, the program continues from there.

## Measured size/speed frontier (same-bin where possible)

| artifact | size GB | tok/s | note |
|---|---:|---:|---|
| canonical packed full-Escha | 8.619 | 2,426 | packed decode too slow |
| P-ARCH-21A (std FFN) | 8.483 | 2,814 | 8.5 GB but < 3030 |
| P-ARCH-23 (+gate Q2_K) | 8.599 | 2,938 | ~8.5 GB, near 3000 |
| P-ARCH-23G (+embed Q4_K) | 8.808 | 3,079 | just above 8.5 |
| P-ARCH-23I (+linear std) | 9.345 | 3,243–3,300 | hits speed, misses 8.5 |
| LowGPU IQ3_XXS | 9.571 | 3,323 | native reference |

## The core tension the sidecar must break

23I reaches SGLang/native-class speed (3243–3300) by standardizing the 96
linear-attention QKV/SSM + gates to Q2_K — which costs +0.75 GB over the
8.5 GB class. A fast-decoding PACKED sidecar keeps the ~2.5 bpw code stream
(→ ~8.5 GB) while decoding at standard-kernel speed. That is the only path
that meets ALL of: size ≤ ~8.6 GB, speed ≥3030 (target 3339), exact packed
Escha semantics preserved, and BeeLlama native features intact.

## A4 slice-1 result (the key learning)

Representation: **PASS** (growth 6.275%, overlay byte-exact, 0 bit mismatches).
Consumer: **FAIL** (four row-warps independently load/evaluate every fragment
weight → FP16 140 regs, decode ALU 93→220, direct-op 8.85× slower, 1 CTA/SM).
Conclusion: pre-resolving dw0/dw1/dsh is necessary but NOT sufficient. The
remaining problem is the four-row-warp duplication — same fundamental cost as
EXP-07/09/10.

## Next iteration (sidecar v2): fix the CONSUMER, keep the representation

The official `escham_code_gemm` avoids the four-warp decode duplication with
warp-collective decode + 45 KiB smem + 2×64-col bands + warp collectives
(40+40 WARPSYNC/ENDCOLLECTIVE, 136 LDS, 80 regs). A4 kept four independent
row-warp decoders. Sidecar v2 must adopt a genuinely collective consumer:
decode once per band into shared/collective state, all warps consume via
ldmatrix from that single publication — while keeping A4's pre-resolved
records (adjacent dw0/dw1 pairs, shared descriptor table). That is the
official-style structure that has NOT yet been tried with a pre-resolved
stream. If the collective consumer still cannot beat the shared-B control,
the packed-exact line is closed and 23/23I-family hybrids at 8.5–8.8 GB are
the delivery path.

Evidence dir: `evidence/PREFILL-50PCT-PLAN/2026-09-02/`.
