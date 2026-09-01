# ARCH-01 — Dense-vs-MoE Architecture Audit (Escha W2)

Date: 2026-08-31
Branch: `escha-w2-prefill`, HEAD `06d6298b5` (local = remote, worktree clean)
Canonical artifact: `escha-w2-lowgpu-mono-parity.gguf`, SHA-256
`e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d`
Audit method: primary-agent source + artifact verification (the parallel
DeepSeek V4 Flash worker route was unavailable — API key returned HTTP 401
"invalid, blocked or out of funds" for all 4 workers — so every claim below
was produced and checked directly by the primary agent).

## 1. Target architecture (Part 1) — DENSE confirmed from the artifact

Direct GGUF metadata (gguf-py GGUFReader):
- `general.architecture` = **qwen35** (not qwen35moe)
- `qwen35.block_count` = 64
- `qwen35.embedding_length` = 5120 (hidden_size)
- `qwen35.feed_forward_length` = 17408 (intermediate_size)
- `qwen35.attention.head_count` = 24, head_count_kv = 4, key/value length = 256
- `qwen35.escha.version` = 1, `qwen35.lowgpu.version` = 1
- **`qwen35.expert_count` / `expert_used_count`: ABSENT**; no expert/moe GGUF keys
- **Zero expert/exps/moe tensors** in the 2,058-tensor inventory
- **400 `*.escha_code` tensors** across 10 projection families:
  attn_gate ×48, attn_q ×16, attn_k ×16, attn_v ×16, attn_output ×16,
  attn_qkv ×48, ssm_out ×48, ffn_gate ×64, ffn_up ×64, ffn_down ×64
- Representative shapes (no expert dimension):
  - `blk.0.attn_gate.escha_code` [32, 384, 320] → K=2, OC/16=384, IC/16=320
  - `blk.0.ffn_up.escha_code` [48, 1088, 320] → K=3, OC/16=1088 (OC=17408), IC/16=320
  - `blk.3.attn_q.escha_code` [32, 768, 320] → K=2, OC=12288, IC=5120
- **Dense here means no MoE routing.** The model is a dense Qwen3.5-hybrid of
  GDN/linear attention (48 layers) and full attention (16 layers, every 4th,
  i.e. il%4==3), with dense (non-routed) FFN.

Layer layout: 64 layers; linear-attention/GDN projections (attn_qkv, attn_gate,
ssm_out) exist on 48 layers; full-attention projections (attn_q/k/v/o) on the
16 full-attention layers (il%4==3). This matches the qwen35 hybrid convention
(no expert routing anywhere).

## 2. Dispatched operator (Part 2) — DENSE op proven, no MoE reaches execution

Loader (src/llama-model-loader.cpp):
- Line 1224: `const bool is_escha_dense = hparams.n_expert == 0;`
- Lines 1227–1230: every `escha_*`-suffix tensor is set to
  `GGML_OP_ESCHA_MUL_MAT` when dense, else `GGML_OP_ESCHA_MOE`; even a tensor
  whose declared `info.op == GGML_OP_ESCHA_MOE` is remapped to the dense op
  when `is_escha_dense` is true. **n_expert==0 is exactly what selects dense.**

CUDA dispatch (ggml/src/ggml-cuda/ggml-cuda.cu):
- Line 2403–2404: `GGML_OP_ESCHA_MOE` → `ggml_cuda_op_escha_moe`
- Line 2406–2407: `GGML_OP_ESCHA_MUL_MAT` → `ggml_cuda_op_escha_mul_mat`

Operator shape proof (escha-moe.cu):
- Dense op `ggml_cuda_op_escha_mul_mat` (line 1740) reads **6 srcs**:
  code, rin, rout, lut, dep, x — **no `ids` tensor**.
- MoE op `ggml_cuda_op_escha_moe` (line 596) reads **7 srcs** including
  `ids = dst->src[6]` (line 603) and uses `n_expert = code->ne[3]`,
  `n_ids`, `n_tokens` (lines 614–617). The dense path structurally cannot
  route experts (no ids input, no expert dimension).

Hot prefill kernel selection (escha-moe.cu):
- `use_mma` prefill branch (line 2075) launches
  `escha_matmul_dense_tiled_mma<K, BM=128, BN=128>` (line 2085; template
  specialization line 2097+) — a genuinely dense 2D-tiled GEMM with split-K
  partials and separate rotate/finalize. `ESCHA_NO_MMA=1` falls back to the
  fp32 FMA dense kernel.

Runtime route proof (promoted build, matched 2K, ESCHA_PROFILE):
- `consolidated.routeproof.stderr`: **800/800 `ESCHA_PROFILE` records,
  all `route=mma-fp16`; zero `moe`/`expert`/`topk` records.**

Model instantiation (src/llama-model.cpp): `qwen35` → `llama_model_qwen35`
(line 312, dense); only `qwen35moe` → `llama_model_qwen35moe` (line 314).

## 3. Ancestry (Part 3) — filename is historical; hot kernel is genuinely dense

- `GGML_OP_ESCHA_MOE` and `GGML_OP_ESCHA_MUL_MAT` both exist (ggml.h 601–602).
  The MoE op is live only for the separate `qwen35moe`/`qwen3moe` model
  classes (models.h 2205; llama-model.cpp 314), which the canonical artifact
  does not instantiate. It is unreachable for this model.
- The file `escha-moe.cu` contains both ops and the shared ESCHA codec decode
  primitives (codebook `escha_codebook_h`, dependency `escha_dep_pi`,
  rotations, LUT). Sharing the trellis decoder and rotations between MoE and
  dense is expected semantic reuse.
- The dense hot prefill kernel `escha_matmul_dense_tiled_mma<K,128,128>` is a
  **separate dense-GEMM design**: 2D CTA tile grid over (rows, output cols),
  split-K slices over IC, CTA-cooperative B decode into shared,
  `ldmatrix` + HMMA, double-buffered cp.async A. There is **no expert-axis
  dimension, no ids/grouped top-k, no per-expert slicing, no expert-local
  reuse**. The only remaining MoE-era tuning artifact in the hot path is the
  launch geometry parameters (BM/BN/NT) and the optional Blackwell split-K
  env (`ESCHA_BW_SPLITK`, line 1961) for decode shapes — not routing.
- **Filename assessment:** `escha-moe.cu` is **historical** for the dense
  prefill body; it accurately reflects that the ESCHA codec was first ported
  for the MoE family, but the dense tensor-core prefill kernel is a genuinely
  dense implementation, not "MoE with routing removed and rows expanded."

## 4. Official dense reference (Part 4) — the mixed-accumulator delta is real but does NOT transfer by toggling

- The correct official reference is `escha-runtime-qwen3dense` (SGLang wheel;
  see docs/environment-setup.md and runtime/wheel-src). Not qwen3moe.
- P-ARCH-19 (docs/escha-architecture-diff-ledger.md:479–486): on the original
  Escha runtime at matched 2K, `ESCHA_PREFILL_ACC=mixed` (fp16 MMA accumulate
  for IC≤6144, fp32 above) = **623.380 ms / 3285.31 tok/s**; forced fp32 =
  **1176.882 ms / 1740.19 tok/s** → **1.888× slower for fp32**. This shows
  the official accumulator policy is first-order on the official fused path
  (553.5 ms / +88.79%).
- P-ARCH-20 (ledger row 105): BeeLlama's own FP16-accumulator port
  (`ESCHA_MMA_FP16ACC_EXPERIMENT`, isolated 32× HMMA.16816.F16) was
  **REJECTED — FP16 was 44.22% SLOWER** (FP32 929.039 ms / 2204.43 tok/s vs
  FP16 1339.906 ms / 1528.47 tok/s). **The official mixed-acc gain does not
  transfer to the BeeLlama kernel by changing the accumulator alone.** The
  BeeLlama kernel decodes B into shared, uses ldmatrix+HMMA with fp32
  accumulate; the official fused `escham_code_gemm` path fuses rotations and
  the mixed policy inside a single operator. This is a structural
  (launch/fusion) difference, not an accumulator toggle.

  **CORRECTION (2026-09-01, EXP-04 Phase 1 contamination audit):** The
  P-ARCH-20 fp16-accumulator result is **contaminated** by an fp16 fragment
  store bug (OOB read of `tile_ah` beyond its 2 half2 elements and dropped
  `.y` lanes; fixed in EXP-04 Stage 2, `7b1880f41`). The measured fp16 route
  executed the faulty store, and P-ARCH-20 produced no valid correctness
  evidence (parity runs errored). The −44.22% number must NOT be used as
  evidence that fp16 MMA accumulation is intrinsically slow in BeeLlama and
  must NOT be used to rule out an accumulator-only toggle. EXP-04 Stage 2's
  structurally-gated mixed accumulator (IC≤6144 → fp16 acc, FP32 partials)
  measured +10% full-2K median with a correct store, 16/16 P2/P7, and no
  regressions. Full table:
  `evidence/EXP-04-phase1/2026-09-01/CONTAMINATION-AUDIT.md`.
- Comparison summary (matched 2K, RTX 5090):
  | path | ms | tok/s |
  |---|---|---|
  | Official qwen3dense mixed | 623.380 | 3285.31 |
  | Official qwen3dense fp32 | 1176.882 | 1740.19 |
  | BeeLlama EXP-01 async (this branch) | ~889 | ~2304 |
  | P-ARCH-23I standard-substitution ceiling | ~619–621 | ~3300 |
- The official artifact is larger (dense fp16 vs packed), so direct
  end-to-end parity is not claimed; the mixed-vs-fp32 same-artifact delta is
  consistent with structural dependence of the official path on
  accumulator/kernel policy (the causal mechanism is not proven by that
  delta alone).

## 5. Classification

**DENSE-CORRECT / PERF-ARCH-MISMATCH**

Rationale:
- The model graph is semantically correct dense Qwen3.5-hybrid (Part 1).
- The dispatched operator is `GGML_OP_ESCHA_MUL_MAT` (dense), proven by
  loader gate (n_expert==0), op src-count (no ids), dispatch mapping, and
  runtime 800/800 mma-fp16 with zero moe/expert records (Part 2). This is not
  filename-only; it is dispatch-proven.
- The dense kernel is genuinely dense-oriented (Part 3), so the
  "SEMANTIC-ARCH-MISMATCH" branch is excluded, and the filename is historical
  rather than a correctness defect.
However, the hot prefill implementation does NOT reproduce the official
dense fused architecture: BeeLlama keeps separate rotate → packed-GEMM →
finalize kernels with fp32 MMA accumulate and a shared-B/ldmatrix round
trip, while the official `escham_code_gemm` is a single fused operator with
a mixed fp16 accumulator policy. P-ARCH-19 (official mixed = 1.888×) plus
P-ARCH-20 (BeeLlama fp16 toggle = 0.56×, i.e. a net loss) are **consistent
with a structural difference and rule out an accumulator-only toggle**; they
do not by themselves prove which structural mechanism (rotation fusion,
launch structure, B-decode path, or mixed accumulator) is causal. That
support for a performance-architecture mismatch is strong but the causal
mechanism remains to be isolated by staged attribution (EXP-04).

## 6. Decision / next experiment (pending Codex/Sol CONFIRM)

Because the classification is DENSE-CORRECT / PERF-ARCH-MISMATCH:

**EXP-04 — dense fused-prefill parity (staged attribution).** The largest
proven structural delta from the official qwen3dense path is that the official
`escham_code_gemm` is a single fused operator with a mixed fp16 accumulator
policy, while BeeLlama keeps separate rotate → packed-GEMM → finalize kernels
with fp32 MMA accumulate. P-ARCH-19/20 prove divergence and rule out an
accumulator-only toggle, but do **not** isolate which mechanism (fusion of
rotations, mixed accumulator, launch structure, or B-decode path) carries the
official gain. EXP-04 must therefore be staged:

1. **Stage 1 — attribution:** measure the per-stage cost of BeeLlama's
   separate rotate / packed-GEMM / finalize launches (ESCHA_PROFILE already
   reports rotate_ms, matmul_ms, epilogue_ms per projection). Confirm the
   fuseable upper bound and which stage dominates (P-ARCH-14 found fused
   finalize neutral; EXP-02/EXP-03 negative — the packed GEMM body is
   expected to dominate, so the fused launch saving may be small).
2. **Stage 2 — structural candidate:** only after Stage 1 quantifies the
   fuseable bound, implement ONE architectural variable (e.g. fuse the
   rotate+GEMM epilogue, or a structurally-gated mixed-accumulator path with
   a quality contract), with SASS/profiler proof of the actual instruction
   and launch structure. Keep the accumulator change gated on a quality
   PASS; do not reduce EXP-04 to an accumulator toggle (P-ARCH-20).
3. **Gates (per experiment):** CV ≤2%, decode ≤2%, P2/P7, family regressions
   ≤5%; full-2K gain ≥20% for a breakthrough classification (≥5% to keep as a
   smaller positive). Rollback = revert the guarded operator.

Expected gain: **hypothesis only, not yet evidenced** — the P-ARCH-19 same-
artifact delta (1.888×) is consistent with the official fused path being
sensitive to accumulator policy, but BeeLlama's own toggles and fusions
(P-ARCH-20 −44%,
P-ARCH-14 neutral, EXP-02/03 negative) mean the fused-parity gain is
**unproven**; it must be measured by staged attribution before any promotion.

## 7. Worker note

The 4 requested DeepSeek V4 Flash workers (Nous Portal) all failed at
dispatch with HTTP 401 (API key invalid/blocked/out of funds). Per the
operating model, worker reports are leads; since none were produced, the
primary agent performed the full audit directly. This is documented rather
than blocking.

## Evidence files (ARCH-01 directory)
- baseline-facts.json (Phase 0: state, artifact hash, baseline, reference)
- This document
- (Route proof source: EXP-01 evidence consolidated.routeproof.stderr)
