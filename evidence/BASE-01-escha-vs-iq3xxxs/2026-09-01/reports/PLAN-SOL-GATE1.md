# BASE-01 — Sol Gate 1: Measurement PLAN (2026-09-01)

Mission: BASE-01 — ESCHA W2 vs IQ3_XXS LowGPU same-runtime breakdown.
Read-only attribution. No optimization implementation.

## 1. Repository state

- Repo: `/mnt/d/CODEX WORKSPACE/beellama-escha`, branch `escha-w2-prefill`
- Local HEAD: `be6bf478dd6f2e63bea81a646bc8b613dbca9935` == remote HEAD (github) — verified
- Merge-base with `dflash2`: `ba27edad2a84ff045a556df06661e821285c2fab`
- Worktree: clean except untracked local aids `results/`, `weights/` (never committed)
- Promoted implementation: EXP-04 Stage 2 mixed accumulator (IC<=6144 fp16 MMA acc / fp32 above), default at HEAD
- EXP-06 (BM64 down-proj): REJECTED + REVERTED (`eb6679159`); no remnants in tracked source (`git grep` empty)
- Leftover: env-gated `ESCHA_CAPTURE_DST_DIR` debug hook at HEAD (inert unless env set; never in timed runs)
- Full detail: `evidence/BASE-01-escha-vs-iq3xxxs/2026-09-01/STARTING-STATE.md`

## 2. Artifact manifests

### Arm A — ESCHA W2 (canonical full-Escha control)
- Path: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf`
- SHA256: `e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d` (matches certification `e307007f…` exactly)
- Size: 8,619,127,360 B (8.62 GB); 400 escha_code projections; no standard substitutions; 2058 tensors
- Quality: 65/75 gated 5-pack certification; historical canonical speed 2355.9 tok/s (Stage 2, graphs on)

### Arm B — IQ3_XXS LowGPU (original LowGPU GGUF)
- Path: `/mnt/d/CODEX WORKSPACE/beellama-release/models/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf`
- SHA256: `ad85e40a28aafd907eebb6ff6b21786b897dd750b0918427f1243d6d84ebcc72` (SHA256SUMS.txt verified)
- Size: 9,570,663,040 B (9.57 GB); 0 escha tensors; 851 standard-GGML tensors; mixed IQ ladder
- Quality: 66/75 gated 5-pack; recorded benchmark: avg 614.736 ms / 3338.76 tok/s (build `0b035b3a2`, P-ARCH-23F)
- **3600 tok/s claim: no recorded measurement exists** in GBrain/docs/sessions — it appears only as a stretch target in `docs/escha-w2-prefill-next-plan.md` §2. The canonical IQ3 reference is 3338.76 tok/s.

Candidate-disambiguation: `/home/sean/Models/cpu-gguf/Qwen3.8-27B-UD-IQ3_XXS.gguf` is EXCLUDED (65 blocks, MTP tensors, different pad id, different lineage).

Full detail: `evidence/BASE-01-escha-vs-iq3xxxs/2026-09-01/manifests/ARTIFACT-MANIFESTS.md`

## 3. Architecture / provenance comparison

Both: `qwen35`, 64 layers, hidden 5120, FFN 17408, heads 24/4, kv len 256, SSM state 128/conv 4/inner 6144, rope 64@1e7, gpt2 tokenizer, vocab 248,320, same bos/eos/pad ids. 48 GDN linear-attn + 16 full-attn (every 4th).

Dequantized projection correlation (Arm A reconstructed vs Arm B dequantized):
- SAME weights (corr scales with Arm B bits): ffn_up 0.873 (IQ1_S), ffn_down 0.873 (IQ1_M), attn_q 0.927 (IQ2_S), attn_k 0.955 / attn_v 0.955 (IQ4_XS), attn_qkv q+k block 0.931 (IQ3_XXS)
- DIFFERENT weights (corr < 0.05, no reorder window recovers): attn_qkv **v**-block (0.0002), attn_gate (0.043 — matches documented P-ARCH-21C donor mismatch), ssm_out (0.040)
- All 10 families now classified: ffn_gate K=2 corr 0.835 (same), attn_output (6144→5120) corr 0.940 (same). No unclassified families remain.

Classification: quantization-only for FFN + full-attn QKV + linear-attn qkv q+k;
**semantic/model difference** for linear-attn gate/ssm_out/qkv-v (LowGPU GGUF
carries different donor weights there — the documented P-ARCH-21C gate mismatch
extends to ssm_out and the fused-QKV v branch); storage-only for vocab
representation (same LowGPU vocab, packed I8 vs IQ4_XS/Q4_K).

Consequence: same architecture + same FFN/full-attn weights + same shapes → the
runtime comparison is valid as same-shape/same-architecture. The differing
families (gate, ssm_out, qkv-v) are part of the ESCHA path we attribute; they are
semantic confounds for quantization-only causality claims and conclusions are
stratified accordingly (Sol REVISE item 4).

Full detail: `evidence/BASE-01-escha-vs-iq3xxxs/2026-09-01/manifests/CORRELATION-SUMMARY.md`

## 4. Frozen binary (one binary, both arms)

- Build dir: `build-cuda-base01` (fresh, from HEAD `be6bf478d`, 2026-09-01)
- Binary SHA256: llama-bench `76485e115bc7c9212966c577906b8bbe7cfcb1cf39db44dbf5a48a80f7a6d82f`; libggml-cuda.so `d18663888721680a403ef063a7cc1c051b35f4d1ff7e6d4db4c293d1dc6f8534`; llama-server `d6e36c724e8a2830fc93e58f606922b965f073542bacf403aa806d112adb9a8c`
- Build command: `cmake -S . -B build-cuda-base01 -G Ninja -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc && cmake --build build-cuda-base01 -j 12 --target llama-bench llama-server`
- Toolchain: nvcc 13.0.88 (cuda_13.0.r13.0), cmake 3.28.3, ninja 1.11.1, gcc 13.3.0; driver 610.88; RTX 5090 32607 MiB SM120
- Graphs: enabled (GGML_CUDA_GRAPHS=ON, no disable env in timed runs)

## 5. Canonical benchmark commands + timing boundary

Command (identical for both arms; matches EXP-04 Phase 2 precedent):
```
build-cuda-base01/bin/llama-bench \
  -m <ARM> \
  --prompt-tokens-file <shared-2048.ids> \
  -p 2048 -n 0 -ngl 99 -b 2048 -ub 2048 \
  -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json
```
- Shared IDs: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids` (sha256 `695c3609bc35a32003a23be3ba1fbacc16cc94955548c2e855e91661c3f62350`)
- Timing boundary: `throughput = 2048 / measured_prompt_seconds` where measured_prompt_seconds = `avg_ns`/`median_ns` from llama-bench JSON for the n_gen=0 prompt test. No scheduler throughput.
- Warm-up: one unrecorded llama-bench run per arm before the recorded campaign.
- GPU-residency proof (Sol REVISE item 1): each arm is loaded with `-v` once (unrecorded) and the **full stderr retained** showing: `llama_prepare_model_devices: using device CUDA0 … N MiB free`; every `load_tensors: layer <i> assigned to device CUDA0` line (65 layers); tensor buffer lines; `CPU_Mapped` buffer size (vocab staging — documented, kernels still run on CUDA0); no compute offload to CPU. Additionally record `nvidia-smi` clocks/power/temp + VRAM used before/after each trial and model+context VRAM from llama-bench JSON (fields populated in `-v` runs). A per-arm "residency.md" captures the layer-assignment proof and any CPU_Mapped tensor names.

## 6. Matched campaign

9 pairs, pre-registered order: **AB BA BA AB AB BA BA AB AB** (A = ESCHA, B = IQ3).
Fresh process per trial (one llama-bench invocation = one trial), one unrecorded warm-up per arm.
Report: raw trials, median latency, median tok/s, mean, stddev, CV per arm; paired latency ratio; paired-log geometric ratio (G) with 95% CI; pairwise winner count; GPU clocks/power/temperature.
Stability rule: if per-arm CV > 2% (WSL host precedent 2-3%), use the pre-authorized paired-log analysis and state CV honestly. No promotion gate in this audit.

## 7. Instrumentation plan (attribution only — never canonical speed)

- Canonical speed: uninstrumented, graphs ON, no ESCHA_PROFILE, no env debug vars.
- **Identical full-operator boundaries (Sol REVISE item 2):** for each projection
  family the logical operator boundary is the full projection, inclusive:
  - ESCHA: input rotation + packed-code GEMM (all split-K slices/partials) +
    finalize reduction + output Hadamard/rotation + scaling/store + all launches
    inside the op. `ESCHA_PROFILE=1` total_ms already spans rotate→finalize
    inclusive; per-stage lines provide the internal split.
  - IQ3: every kernel inside the standard quantized MUL_MAT operator for that
    projection (decode/unpack + matmul body + epilogue + any helper launches),
    aggregated from nsys kernel traces by op-owner attribution.
  - Overlap rule: exclusive boundaries per op; no CUDA range counted twice; the
    aggregation for each arm must be cross-checked against that arm's whole-run
    graphs-off total (sum of ops + non-projection == total within tolerance).
- **Graphs-off vs graphs-on accounting (Sol REVISE item 3):** component
  accounting is closed inside the **graphs-off** boundary for each arm
  (measure per-arm graphs-off total latency with `GGML_CUDA_DISABLE_GRAPHS=1`,
  3 repeats). Then separately measure each arm's graphs-on total (canonical
  campaign) and report per-arm delta = graphs_on_total - graphs_off_total as the
  graph/orchestration contribution. The canonical gap is then reconciled as:
  `(A_off - B_off) + (deltaA - deltaB)`. Components are NEVER ratio-scaled into
  graphs-on latency; graphs-off data is used only for relative component shares
  and for the closed graphs-off boundary.
- ESCHA per-stage: existing `ESCHA_PROFILE=1` hook (CUDA events around rotate/matmul/epilogue boundaries; route tag; total/rotate/matmul/epilogue ms per call). Run graphs OFF for attribution only — historical precedent (EXP-04 Stage 1). Aggregated per-family + per-shape.
- IQ3 per-op: no ESCHA hook; use `nsys` (2025.3.2 available) kernel traces around the same prompt run (graphs OFF for symmetric boundaries) to aggregate standard GGML CUDA kernels by operator family (get_rows, norm, flash-attn, quantized MUL_MAT / MMQ / dequant kernels, etc.). Same `-p 2048 -n 0` contract, graphs off, labeled relative-attribution only.
- WSL limitation: occupancy/hardware counters unavailable (`ERR_NVGPUCTRPERM`); use CUDA-event/profile-hook + nsys kernel traces, document limitation.
- Symmetry: both arms profiled graphs-off with equivalent logical boundaries; never compare an ESCHA subkernel against an IQ3 full operator.

## 8. Statistical protocol

- n = 9 paired trials per arm, fresh process each.
- Primary metrics: median tok/s, median prompt latency (ms).
- Paired analysis is PRIMARY regardless of CV (Sol REVISE item 5): per-pair latency ratio r_i = t_A_i / t_B_i; geometric mean G = exp(mean(ln r_i)); 95% CI via t-distribution on ln r_i; pairwise winner count (B faster than A count).
- Dispersion: per-arm mean, sample SD (ddof=1), CV% (sample SD / mean) of latency and tok/s — sample convention, stated explicitly.
- **Preregistered CV>2% contingency:** if either arm's per-sample latency CV (sample SD) > 2%, run a second fixed balanced 9-pair block (same order) and report both blocks plus the combined estimate; if the combined 95% CI on G still spans 1.0, declare the comparison inconclusive rather than promote a finding.
- Accounting closure: components must explain >=95% of median latency difference or explicitly identify the unaccounted remainder.
- All raw JSON/stderr retained in evidence dir.

## 9. Attribution accounting method

total_gap = t_A_median - t_B_median
- Close accounting inside each arm's **graphs-off** total: projection_gap_off + non_projection_gap_off + orchestration_off = A_off - B_off (>=95% closure).
- projection_gap_off: sum over projection families of (ESCHA per-family projection time - IQ3 per-family equivalent) from symmetric graphs-off profiling (same inclusive operator boundary).
- non_projection_gap_off: norms, attention, SSM, RoPE, embeddings, residual ops, output/logits, memory ops (from nsys + profile data, both arms).
- orchestration_off: launch counts, allocation/sync remainder.
- Graph/orchestration contribution (separate): per-arm delta = graphs_on_total - graphs_off_total; report deltaA - deltaB explicitly.
- Canonical gap reconciliation: (A_off - B_off) + (deltaA - deltaB) == A_on - B_on, with each term measured, not inferred by scaling.

Projection families (all): attn_gate, attn_qkv, attn_q, attn_k, attn_v, attn_output, ssm_out, ffn_gate, ffn_up, ffn_down. Per family: shape, quant route, invocation count, aggregate ms, median per call, % of model time, CUDA symbol, grid/block, accumulator type, launch count.

ESCHA-specific: input rotation, packed-code GEMM, partial-buffer writes, split-K, finalize reduction, output Hadamard/rotation, u_buf/p_buf allocation/lifetime, n_slices/route for all 400 calls.
IQ3-specific: identify selected IQ3 path (MMQ vs dequant+cuBLAS vs other) from nsys symbols; quantized decode/unpack, matmul body, epilogue, launch count; forced-route diagnostics only labeled separately, one variable at a time.

## 10. Deliverables after plan

- Phase 2: matched campaign (above)
- Phase 3: component breakdown + family tables
- Phase 4: direct matched-shape operator comparison at M = 128/512/1024/2048/4096
- Phase 5: SASS/resource comparison for dominant shapes (cuobjdump)
- Phase 6: decision report + ranked next targets
- Gates: Sol Gate 2 (evidence review) before conclusions; Sol Gate 3 (final VERIFY)
- Closure: commit evidence/docs only; push; verify remote; GBrain update

## Sol Gate 1 requested output

Return one of:
- PLAN=READY
- PLAN=REVISE (with specific items)
- PLAN=NO-GO (with reasons)

Check specifically: timing boundary correctness; confounds (artifact equality, binary equality, offload, graphs); instrumentation symmetry; statistical soundness of the paired protocol; attribution closure method; any missing measurement that would invalidate the ~300 ms prefill gap accounting.
