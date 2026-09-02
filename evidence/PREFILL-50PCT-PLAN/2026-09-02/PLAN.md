# ESCHA-W2 PREFILL — +50% target: highest-leverage new attempts

Date: 2026-09-02 · Branch `escha-w2-prefill` · Control: promoted Stage 2
(2319.22 tok/s / 883.1 ms matched-2K, graphs ON).

## Target math (the load-bearing fact)

| target | tok/s | ms/2k | vs control |
|---|---:|---:|---:|
| control | 2,319 | 883.1 | — |
| +20% | 2,783 | 735.9 | |
| +42.3% (23I measured, graphs OFF) | 3,300 | 620.6 | |
| +44.0% (LowGPU IQ3_XXS reference) | 3,339 | 613.2 | |
| **+50%** | **3,479** | **588.7** | |

**Critical implication:** +50% is *above* the LowGPU IQ3_XXS reference (+44%).
The plan must not merely match the standard-quant artifact — it must BEAT the
stock standard-GGML path by ~4%. That reframes every option.

## What the series proved (constraints)

1. **Packed-runtime incremental kernel work is exhausted** (EXP-02/03/06/07/08/
   09/10): register ceiling, spill, occupancy, or serialization failures at
   BM128×BN128/SM120. The canonical packed representation cannot be driven to
   LowGPU speed by local mainloop edits.
2. **Representation is the proven lever**: P-ARCH-21A (+21%), 23 (+27%),
   23G (+33%), 23I (**+42.3%**, 3300 tok/s) by putting the body on stock GGML
   Q2_K/IQ MUL_MAT. 23I is 9.345 GB — already 226 MB smaller than LowGPU —
   yet still ~1.2% SLOWER than LowGPU.
3. EXP-11 (load-time transcode cache) is the funded path to make that
   representation runtime-native without touching the canonical artifact. Its
   Attempt-1/2 gate issues are operational (cold wall, oracle byte-pinning),
   not representation issues — steady-state speed equals 23I's class.
4. External evidence (escha-port): even a clean-room ESCHAM port loses prefill
   ~2.3× vs standard quants; tensor cores won via register-freed band blocking,
   not arithmetic.

## The honest gap from 23I to +50%

23I (3300) → target (3479) is only **+5.4%**. That is a single modest,
well-chosen win away — NOT a moonshot. The question is where that +5.4% (and
headroom beyond) lives.

## Ranked new attempts (hammer the biggest upsides)

### A1 — Milestone-certify 23I under the canonical protocol (graphs ON) [BIGGEST, cheapest, highest certainty]

- **Why:** every 23I number (619–621 ms / 3300) was measured **graphs OFF**
  (`GGML_CUDA_DISABLE_GRAPHS=1`). The certified 2319 control and the EXP-08
  campaign used graphs ON. CUDA-graph capture historically added ~1–2% on the
  packed path; the standard body may capture even better (no per-op escha
  decode). If graphs-ON 23I measures 3339–3479, the +50% target is reached
  with zero new kernel work.
- **Deliverable:** canonical matched 9-pair graphs-ON campaign + 75-case
  medium quality suite on the frozen 23I artifact (`standard-ffn-gdn-q2k-
  embedq4-attn-linear.gguf`, sha known).
- **Gate:** ≥3339 tok/s graphs-ON (LowGPU parity) = representation line wins;
  ≥3479 = +50% achieved; quality ≥65/75, decode ≤2%.
- **Risk:** low. **Cost:** one campaign + one quality suite.
- This is Decision A executed — and it may alone deliver the target.

### A2 — Quant-decode-speed tuning on the standard body [cheap, +2–6%]

- **Why:** 23I chose Q2_K for FFN/gate/linear-attn (P-ARCH-23 found Q2_K
  beat Q4/Q6 on the packed path). But on STOCK GGML decode, IQ3_XXS/IQ4_XS
  family kernels may decode faster than Q2_K at similar size — LowGPU (IQ3_XXS-
  heavy, 9.571 GB) is 1.2% faster than 23I (Q2_K-heavy, 9.345 GB) despite
  being larger. Per-family decode-speed microbench on standard MUL_MAT can find
  the faster quant at ≤23I size.
- **Deliverable:** per-family decode-speed table (Q2_K vs IQ3_XXS vs IQ4_XS)
  on the 5120→17408 / 17408→5120 shapes; one re-quantized candidate if a
  family wins ≥2% while staying ≤9.35 GB.
- **Gate:** ≥+2% full 2K over 23I with quality unchanged.
- **Risk:** low. **Cost:** microbenches + one conversion.

### A3 — Full-body standardization incl. embedding + head at ≤10 GB [moderate, +1–4%]

- **Why:** 23I still executes the LowGPU packed LM head (output.lowgpu_codes/
  scales/zps) and keeps Q4_K embedding. The head is the last packed op on the
  hot path. Standardizing it (within the 10 GB cap that P-ARCH-22 flagged as
  tight) removes LOWGPU_MUL_MAT overhead.
- **Gate:** ≥+1.5% over 23I, artifact ≤10 GB, quality preserved.
- **Risk:** medium (head is 5120→152k; size pressure).

### A4 — EXP-11 Attempt-3: MMA-ready sidecar repack (the ONLY packed-exact path with +50% ceiling) [high risk, high cost]

- **Why:** every other path abandons exact packed execution. A fragment-ordered,
  pre-resolved-index repack of the Escha code stream could make warp-local
  decode cheap enough to run the PACKED artifact at standard speed — smaller
  than LowGPU AND at/above its speed. This is the only line that could push the
  canonical artifact itself past LowGPU.
- **Mechanism gate (before full build):** one repacked K2 5120→17408
  projection; size growth ≤25%; fp16 ≤104 regs; no spills; decode ALU −30%;
  direct-op matmul ≥15% faster at M=2048.
- **Risk:** high — EXP-09/10 show decode ALU/register coupling is brutal; the
  external escha-port stopped at the same wall.
- **Recommendation:** do NOT start this until A1–A3 are banked; it is the
  fallback if A1–A3 cannot cross 3339→3479 and exact-packed execution is a
  product requirement.

### A5 — Standard-kernel SM120 gap hunt [speculative, unknown]

- **Why:** if A1–A3 top out below 3479, the residual is ggml's own standard
  quant MUL_MAT kernels. The series' SM120 lessons (cp.async, fp16 mixed acc,
  occupancy) might expose a standard-kernel gap (e.g., mmq decode path for
  Q2_K/IQ3 on SM120). Low expected value (ggml is mature) but the only lever
  left that attacks the stock path directly.
- **Gate:** profile first; only implement on a measured ≥5% shape-level gap.

## Stop rules (what NOT to repeat)

- No new packed-kernel mainloop variants (EXP-02/07/09/10 negatives; EXP-11
  Decision-B stop).
- No finalize fusion without full-CTA-parallel proof (EXP-08).
- No donor-substitution without semantic verification (P-ARCH-21B).
- No per-call dequant/cuBLAS (only persistent cache).
- Don't chase +50% on the PACKED artifact via kernels; the only packed path is
  A4's representation change.

## Sequencing recommendation

1. **A1 now** (certify 23I graphs-ON — may already hit target; zero new code).
2. **A2 in parallel** (quant decode-speed microbench — informs A3 and any A1
   shortfall).
3. **A3 if A1+A2 < 3479.**
4. **A4 only if exact-packed is a hard requirement AND A1–A3 < 3479.**
5. **A5 only on measured stock-kernel gap.**

Evidence dir: `evidence/PREFILL-50PCT-PLAN/2026-09-02/`. Sol PLAN-gate review
queued (Codex usage cap resets ~12:20 PDT).
