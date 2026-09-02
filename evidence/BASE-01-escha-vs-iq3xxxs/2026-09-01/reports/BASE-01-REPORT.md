# BASE-01 — Decision Report: ESCHA W2 vs IQ3_XXS LowGPU Same-Runtime Breakdown (2026-09-01)

## Executive summary

Under one frozen BeeLlama binary (`build-cuda-base01`, HEAD be6bf478d, graphs ON,
`-p 2048 -n 0 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -ngl 99`, shared-2048
IDs), the canonical matched campaign measured:

- **ESCHA W2 median: 2326.77 tok/s (880.19 ms)** — the historical ~2356 tok/s is reproduced.
- **IQ3_XXS LowGPU median: 3212.63 tok/s (637.48 ms)** — the historical 3339 tok/s (measured on an older preview build 0b035b3a2) is NOT reproduced on this build; the **~3600 tok/s figure is NOT SUPPORTED by any recorded measurement** (it exists only as a stretch target in `docs/escha-w2-prefill-next-plan.md` §2).
- Paired-log G = 1.3863, 95% CI [1.3717, 1.4010], 9/9 B-faster, per-arm CV < 2% (1.6% / 0.74%), CV contingency not triggered.
- **Total latency gap: 242.7 ms** (the mission premise of ~300 ms is not supported by the canonical contract; the recorded P-ARCH-23F 614.7 ms / 3339 tok/s was a different build).

## 1. Is ~3600 tok/s reproducible under the canonical contract?

**No.** The canonical IQ3 median is 3212.6 tok/s. No artifact, session, or doc
in GBrain/wiki/repos contains a recorded 3600 measurement; it appears only as a
stretch target. The closest historical reference (3339 tok/s) was measured on a
pre-promotion preview build and does not reproduce on the current frozen binary
(3212.6, −3.8%).

## 2. How much of the gap is projection operators vs non-projection/runtime?

- Graphs-off whole-run: A 816.2 ms vs B 590.3 ms → **225.9 ms projection+runtime gap**.
- Share-scaled projection-family accounting: **projection families account for the
  full graphs-off gap** (closure 100.0% by the approved share-scaling method).
- Graph/orchestration delta (measured per arm, graphs-on − graphs-off): A +64.0 ms,
  B +47.2 ms → **+16.8 ms of the canonical gap is graph/orchestration overhead
  difference**. Reconciled canonical gap = 225.9 + 16.8 = **242.7 ms exact**.
- Caveat stated honestly: "100% projection" is by construction of share scaling
  (profiled totals include per-op sync overhead). The independent evidence is the
  Phase 4 M-scaling and the exact whole-run reconciliation. Non-projection GPU
  work (norms/attention/SSM/embeddings) is similar between arms and does not
  materially contribute; the ~7% graph delta is the only measurable non-projection term.

## 3. Which families account for ≥80% of the gap?

| family | gap ms | % of positive gap |
|---|---|---|
| ffn_up | +79.4 | 35.2% |
| ffn_down | +69.9 | 31.0% |
| ffn_gate | +64.5 | 28.6% |
| ssm_out/attn_output | +23.1 | 10.2% |
| attn_gate | +14.9 | 6.6% |
| attn_q | +12.9 | 5.7% |
| attn_qkv | +5.3 | 2.4% |
| attn_kv (V) | −15.7 | −7.0% |
| ssm_beta_alpha (B-only) | −14.5 | −6.4% |
| lm_head (B-only) | −14.0 | −6.2% |

**The FFN block (ffn_up + ffn_down + ffn_gate) = +213.8 ms = 94.6% of the positive
gap** — and these are exactly the SAME-weights families (proven correlation
0.83–0.87), so the gap is an operator/quantization-path deficit, not a weight
difference. The differing-weight families (attn_gate, ssm_out, qkv-v) contribute
small, largely cancelling offsets.

## 4. Time attributable to each stage

- ESCHA input rotation: ~5.4% of ESCHA projection time (rotate 189.7/3509.7 ms
  aggregated); transferable wall value in the gap is a few ms — below the 5%
  rule, do NOT run a rotate fusion solely for its historical bound.
- ESCHA packed GEMM: **the dominant term** (matmul ≥86% of every family;
  ffn block alone +213.8 ms share-scaled).
- Partial-buffer/finalize: ~7% of ESCHA projection time (epilogue 249.0/3509.7);
  ~10 ms class wall value. A warp-owned output-finalize fusion is the largest
  single fuseable bound (matches EXP-04/05 ranked next target) but the GEMM body
  itself is the wall.
- IQ3 decode/GEMM: one MMQ kernel per projection (route proof: 994 GGML_OP_PROFILE
  launches/run, 2982 across 3 runs; per-tensor-type `mul_mat_q<type,BK>` kernels,
  Q2_K 224 + IQ3_XXS 134 + mixed ladder); no separate rotate/finalize; this is the
  structural advantage.
- Launch/intermediate-memory differences: ESCHA issues rotate+GEMM+finalize+
  partial-buffer passes per projection (u_buf/p_buf), IQ3 one kernel; fixed
  overhead explains the small-M ratio (2.25× at M=128 vs 1.36× at 2048) but not
  the flat per-call mainloop penalty.

## 5. Is official ESCHA parity (~3225–3305 tok/s) credible inside BeeLlama?

**Not on the current ESCHA kernel.** To reach 3225–3305 tok/s ESCHA would need to
eliminate ~28–34% of its current 880 ms latency (to ~620–635 ms). The FFN packed
GEMM mainloop is ~1.2–1.8× slower per call than IQ3's MMQ at every M; no launch
fusion can close a mainloop deficit of this size. P-ARCH-23I reached ~3300 tok/s
only by replacing the ESCHA body with standard GGML tensors (artifact
substitution), which is outside this runtime-only scope. Within the current
kernel, 3000 tok/s is credible only via a broad FFN-mainloop win; 3225–3305 is
not credible without either a new ESCHA mainloop or artifact substitution.

## 6. Required matmul/fusion gain to reach targets (from 880.19 ms baseline)

| target | latency needed | ms to remove | required gain |
|---|---|---|---|
| 3000 tok/s | 682.7 ms | 197.5 ms | −22.4% |
| 3225 tok/s | 635.0 ms | 245.2 ms | −27.8% |
| 3305 tok/s | 619.7 ms | 260.5 ms | −29.6% |
| 3600 tok/s | 568.9 ms | 311.3 ms | −35.4% (no measured support) |

## 7. Ranked next targets (measured gap coverage, credible gain, risk)

1. **New broad ESCHA mainloop (packed-GEMM body redesign)** — covers ~95% of the
   gap (FFN block). Credible wall gain 15–25% if a decode-once/HMMA-heavy
   structure like the official `escham_code_gemm` (per EXP-05: 80 regs, 45 KiB
   smem, fp16 acc) is matched. Risk: high (mainloop rewrite, occupancy, split-K);
   correctness risk medium. Highest value; the only path to 3000+ tok/s.
2. **Warp-owned output-finalize fusion (into GEMM epilogue)** — covers ~6.7%
   (finalize + launch removal; ~10–15 ms wall). Credible +2–4%. Risk: low-medium;
   Sol PLAN already READY from EXP-04/05. Correctness low. Do NOT gate on it
   alone; it cannot reach any 3000+ target.
3. **Fused input rotation (shared up+gate / qkv+z transform)** — covers ~5.4%
   (~10 ms). Credible +1–3%. Risk: medium (Hadamard fusion, smem). Correctness
   low. Below 5% transferable rule unless combined.
4. **Operator ownership / no-partial redesign (eliminate u_buf/p_buf passes)** —
   covers the small-M fixed overhead (ratio 2.25→1.36 as M grows) but only a few
   ms at 2048. Credible +1–2%. Risk: high (split-K ownership). Low priority.
5. **Runtime/graph work** — measured delta contribution only +16.8 ms (~7%);
   graphs already hurt single-shot prefill on both arms. Credible +1–2% at most.
   Not a lever for the 12–16 GB target class.
6. **Shape-specific kernels for ffn_down (17408→5120, per-call 1.81×)** — the
   single worst per-call shape; if the mainloop rewrite is too broad, this one
   shape is the highest-value narrow target (EXP-06 BM64 was the wrong variable:
   tile aspect; the deficit is decode structure, not tile geometry).

## 8. Outcome-specific interpretation

- **≥80% of the gap is inside ESCHA projection execution → classified as an
  ESCHA operator architecture deficit.** (FFN block 94.6%; the packed-GEMM
  mainloop is the wall.)
- IQ3's conventional projections are fast and non-projection work matches →
  **BeeLlama is ruled out as the main ceiling.** The runtime overhead difference
  is only +16.8 ms (~7%).
- Non-projection/runtime differs materially only in the small graph delta; no
  artifact or execution-path confound explains the gap (both arms fully
  GPU-resident, same binary, same shapes; the only artifact difference is the
  documented gate/ssm_out/qkv-v donor mismatch, which is small and cancelling).
- Rotate/finalize fusion alone is below the 5% transferable rule → do not run it
  solely for its historical bound.
- The gap is **concentrated in the FFN block** → prefer a broad FFN/mainloop
  operator fix (target 1) over isolated tile tweaks; ffn_down is the
  highest-value single shape if scoped narrowly.

## 9. Sol verdicts

- Sol Gate 1: **PLAN=READY** (4 REVISE rounds resolved; verdict file SOL-GATE1-VERDICT.md).
- Instrumentation review: **INSTRUMENTATION=CONFIRM** (GGML_OP_PROFILE hook).
- Sol Gate 2 / Gate 3: pending in this report cycle (evidence submitted below).

## 10. Required closure items

- Instrumentation source (GGML_OP_PROFILE hook) is in the isolated profile build
  only; production `ggml-cuda.cu` at HEAD unchanged (hook added only in the
  profile build dir's working tree; canonical `build-cuda-base01` binaries
  unchanged). Revert working-tree source to be6bf478d before commit.
- Evidence manifest and commits pending (see closure section).
