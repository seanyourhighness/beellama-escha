# 23J size decision — smaller IS possible, but the quant hypothesis is falsified

Date: 2026-09-02 · Question: build 23J only if smaller than 23I (9.345 GB)?

## Byte accounting (per-tensor, block-exact)

| family | 23I | LowGPU | LowGPU − 23I |
|---|---:|---:|---:|
| ssm_alpha/beta (96, F16 vs Q8_0) | 47.2 MB | 25.1 MB | **−22.1 MB** |
| embedding (Q4_K vs IQ4_XS) | 715.2 MB | 675.4 MB | **−39.7 MB** |
| GDN_gate (Q2_K vs mixed IQ) | 258.0 MB | 272.8 MB | +14.7 MB |
| linear_attn (Q2_K vs mixed IQ) | 918.7 MB | 1,045.1 MB | **+126.5 MB** |

## Verdict on the original A2 idea

"Re-quant gate/linear to LowGPU's mixed-IQ recipe" would make 23J **+141 MB
BIGGER** than 23I — fails Sean's smaller-than constraint outright.

The only smaller 23J keeps Q2_K on gate/linear (where 23I is already leaner)
and fixes the two waste spots:
- Policy A (ssm F16→Q8_0 only): **−22.1 MB** → 9.44 GB
- Policy C (+ embedding IQ4_XS): **−61.8 MB** → 9.28 GB
- Policy B (+ every LowGPU-smaller choice): **−165 MB** → 9.18 GB

## Why A2 (quant decode speed) is likely WRONG as the 2.5% explanation

LowGPU is 2.5% faster than 23I while moving MORE bytes on gate (+15 MB) and
linear_attn (+126 MB), with byte-identical FFN. If quant decode speed were the
driver, the larger LowGPU should be SLOWER on those families. It is not.
Therefore the residual 2.5% is structural, most plausibly:
1. `qwen35.escha.version=1` forces per-layer Escha semantic handling
   (A_log → -exp, grouped/interleaved GDN head order) in 23I; LowGPU has no
   marker and skips all of it.
2. 23I's LM head runs packed `LOWGPU_MUL_MAT`; LowGPU uses standard Q4_K
   output.weight.
3. F16 ssm_alpha/beta may take a non-quantized kernel path.

## Decision (Sean)

23J only if smaller → a −62 MB variant is buildable but does not target the
likely cause of the remaining 2.5%; the A2 quant-remix direction is dead.
Sean's stated priority: spend the Codex cap (reset ~12:20) on the **MMA-ready
sidecar (A4 / EXP-11 Attempt 3)**. This file records the size math so the
question does not need re-deriving.
