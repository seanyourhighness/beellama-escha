# BeeLlama Escha — Current State

> Concise snapshot of what is true NOW (2026-08-31). See the experiment ledger
> (`docs/escha-prefill-experiment-ledger.md`) for full history and evidence.

## Mission

Add first-class Escha support and optimized execution to BeeLlama while
retaining compatibility with normal Qwen/GGUF models.

## Runtime semantic invariant

- **Standard Qwen / GGUF:** original BeeLlama semantics (unchanged).
- **Escha artifacts:** Escha-specific semantics applied **exactly once**, gated
  by metadata such as `qwen35.escha.version` (`escha_version`).
- No filename/artifact-name conditionals, no benchmark-specific behavior.

This invariant is enforced by the **`escha_version`-gated qwen35.cpp** changes
(decay representation + GDN head layout). Ungated Escha semantics were
corrupting standard GGUFs; the fix is a runtime/model-compatibility fix, not an
artifact workaround.

## Canonical artifacts

- **Canonical full-Escha control:** `escha-w2-lowgpu-mono-parity.gguf`
  sha256 `e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d`
- **Standard LowGPU control:** `Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf`
  (standard qwen35 GGUF, no escha markers)
- **Current P-ARCH artifacts (performance experiments):**
  - `...standard-ffn-gdn-q2k.gguf` (P-ARCH-23)
  - `...standard-ffn-gdn-q2k-embedq4.gguf` (P-ARCH-23G)
  - `...standard-ffn-gdn-q2k-embedq4-attn-linear.gguf` (P-ARCH-23I — prefill
    speed parity leader, ~619-621 ms / ~3300 tok/s)
- **Obsolete / broken:** `escha-w2-lowgpu-mono.gguf` — invalid conversion
  (residual RMSNorm offset −1.0; incorrect `ssm_beta`/`ssm_alpha` ordering). Do
  not use for scoring.

## Certified benchmark result (2026-08-31)

Club-3090 **medium** quality 5-pack (75 scenarios, no Docker), thinking
force-off, on the gated `build-cuda-qwen35-gated` runtime. **Judge PASS (exit 0).**

| Model | /75 | equiv/150 | reasonmath |
| --- | --- | --- | --- |
| **original-lowgpu** | **66** | **132** | **12/15** |
| base-escha-w2 | 65 | 130 | 11/15 |
| p-arch-23 | 65 | 130 | 10/15 |
| p-arch-23g | 65 | 130 | 10/15 |

Evidence: `escha-w2-lowgpu/evidence/club3090-medium/2026-08-31-gated/` (external).
In-repo bindings: `docs/escha-benchmark-2026-08-31/benchmark-summary.json`,
`raw-evidence-manifest.json` (SHA-256 of all raw result JSONs/logs/preflights),
`judge-manifest.json`, `DOD.md`, `JUDGE-EVIDENCE.md`, `sha256sums.txt`.
Judge evidence/DoD/verdict also in `~/.hermes/state/judge-escha-medium-20260831.json`.

## Benchmark configuration (certified)

- Runtime: `build-cuda-qwen35-gated/bin/llama-server`
  (commit `0b035b3a2-dirty`, escha_version-gated qwen35.cpp)
- F16 KV (`-ctk f16 -ctv f16`), 32K context (`-c 32768`), single slot (`-np 1`),
  flash attention on, Jinja, **thinking force-off**, **chat parsing enabled**
  (no `--skip-chat-parsing`).
- Same binary for all four models (no cross-runtime comparison).

## Current conclusions

1. Old `escha-w2-lowgpu-mono.gguf` artifact was **invalid** (conversion
   defects). Use the parity artifact as the full-Escha control.
2. **`escha_version` gating is required** in qwen35.cpp; ungated Escha
   semantics broke standard GGUFs (original-lowgpu recovered 5/75 → 66/75).
3. **Standard LowGPU is healthy** after gating (66/75, leads ReasonMath).
4. **Base Escha quality matches the P-ARCH substitutions** on the current
   quality suite (65/75 each — effectively tied).
5. **P-ARCH substitutions primarily represent performance experiments**, not
   quality improvements.
6. **Next development objective:** full Escha W2 **prefill performance** inside
   BeeLlama (see Active next phase).

### EXP-01 result (2026-08-31) — PROMOTED

SM120 async A-stage overlap is now the default Escha prefill route
(`escha-moe.cu`; sync fallback = `ESCHA_MMA_SM120_SYNC_FALLBACK` opt-in).
Matched-2K prefill on the canonical full-Escha artifact:
**1450 → 2302 tok/s (+58.8%)**, CV ≤2%, decode unchanged, P2/P7 100%,
standard-GGUF identical. Implementation commit `215aa4ac3`. Milestone
certification pending (full medium 5-pack next).

### Milestone certification (2026-08-31) — PASS

Full club-3090 medium 5-pack on the async-default consolidated build:
**65/75 (130/150 equiv)** — identical to certified baseline, **no quality
regression**. Evidence: `evidence/EXP-01-sm120-async/2026-08-31/milestone-cert/`.

### EXP-02 (2026-08-31) — REJECTED

Direct-fragment packed K2 GEMM made prefill slower (2327→2236 tok/s, −3.9%);
176 regs/thread hurt occupancy. Isolated + reverted; implementation tree
matches promoted EXP-01 (`215aa4ac3`). See ledger for details.

### EXP-03 (2026-08-31) — REJECTED (neutral)

Shared-B 256x64 balanced K2 tile: 2304→2294 tok/s (−0.42%), 128 regs no
spills, P2/P7 100%. Tile aspect does not move prefill. Combined with EXP-02,
B-decode amortization and fragment/tile layout are NOT the bottleneck — the
packed K2 matmul body is the wall. Reverted; tree matches `215aa4ac3`.

### ARCH-01 (2026-08-31) — DENSE-CORRECT / PERF-ARCH-MISMATCH

Architecture audit: the model is semantically dense (qwen35, n_expert==0,
GGML_OP_ESCHA_MUL_MAT, no MoE routing reaches execution) and the filename
`escha-moe.cu` is historical. BUT the hot prefill kernel does not reproduce
the official qwen3dense fused architecture (separate rotate/GEMM/finalize,
fp32 MMA acc vs official fused `escham_code_gemm` with mixed fp16 policy;
P-ARCH-19: 1.888× mixed-vs-fp32; P-ARCH-20: BeeLlama fp16 toggle alone loses
44% — consistent with a structural difference, not an accumulator-only
toggle). Next: EXP-04 dense fused-prefill parity — **staged attribution** (measure
the fuseable bound first; do not reduce to an accumulator toggle; the >20%
gain is hypothesis, not yet evidenced). Full audit:
`docs/escha-w2-architecture-provenance-audit.md`.

### EXP-04 Stage 1 (2026-09-01) — COMPLETE (measurement only)

Fuseable rotate/GEMM/finalize bound measured on the current default route
(`mma-fp16`, HEAD `4501b3ee1` = EXP-01 kernel), canonical artifact, 2k gate,
`ESCHA_PROFILE` 800/800 `route=mma-fp16`. Steady-state attribution:
**rotate 4.6% · matmul 88.6% · epilogue 6.7%**. The packed-GEMM body is the
wall in every family; the fuseable launch bound is ≈11.3% best case, so
launch fusion alone cannot reach the ≥20% breakthrough gate — Stage 2 must
target the GEMM body (structural mixed accumulator or B-decode/launch
structure with SASS proof). Timed control confirmed the banked baseline
(median 2284.7 tok/s, −0.75% vs ~2302; control CV 3.12% on 3 samples — not a
candidate gate; decode unchanged). Evidence:
`evidence/EXP-04-stage1/2026-09-01/`. **Sol/Codex review: VERDICT=CONFIRM**
(2026-09-01, all checks PASS; first round REVISE for CV overclaim + fingerprint
contract, fixed in `8ebcbdde5`). Stage 2 in progress.

### EXP-04 Stage 2 (2026-09-01) — RESULTS: SMALLER POSITIVE +10% median

Structurally-gated mixed accumulator (fp16 MMA acc for IC≤6144, fp32 above —
native Escha mixed policy), ONE variable under `ESCHA_MMA_MIXEDACC_EXPERIMENT`,
commit `7b1880f41`, Sol implementation review CONFIRM. SASS proof: fp16 kernels
only `HMMA.16816.F16` (97 regs, no spills), fp32 twins only `.F32` (128 regs);
route proof 800/800, 0 predicate mismatches. Matched 2k: **+10.0/+10.9/+10.5%
median over 3 runs** (2496–2508 vs 2251–2281 tok/s), decode no regression,
P2/P7 16/16, no family regressions (fp16 families −17.6 to −26% matmul).
**Classified SMALLER POSITIVE (≥5%, <20%).** CV 3.1–6.3% per arm (host noise,
flagged) exceeds the ≤2% letter — **Sol VERIFY VERDICT=CONFIRM** (round 1
REVISE on raw resource evidence + CV wording, both fixed in `ba4e93851`; all
8 checks PASS, gate CONDITIONAL on CV). NOT promoted to default; promotion
requires full milestone certification. Evidence:
`evidence/EXP-04-stage2/2026-09-01/`.

### EXP-04 Phase 2 — Stage 2 PROMOTED (2026-09-01)

Noise-resolution run with a Sol-pre-authorized protocol (9 matched ABBA pairs,
frozen binaries): paired-log **G=1.0973, 95% CI [1.0774, 1.1175], 9/9 →
PASS**; median **+9.31%** (2355.9 vs 2155.2 tok/s); primary per-arm CV≤2% not
met on this WSL host (control 3.24%, candidate 2.15%) so the pre-authorized
fallback decided. Same-session reconfirms: route 800/800 0-mismatch, SASS
.F16/.F32 split + REG 97/128 no spills, P2/P7 16/16, decode +1.63% (≤2%),
no family regressions, binaries unchanged since `7b1880f41`.
**Sol VERIFY VERDICT=CONFIRM, Stage 2 gate PASS, classification SMALLER
POSITIVE. Stage 2 is now the promoted prefill control.** Evidence:
`evidence/EXP-04-phase2/2026-09-01/`.

### EXP-04 Stage 3 — REJECTED (2026-09-01)

Bounded-K FP16 for the 17408→5120 family (n_slices=4, 272 tiles/slice < safe
384): Sol-planned and Sol code-reviewed; route 800/800 0-mismatch, SASS fp16
REG 97 no spills, rel-RMS 1.08e-3 vs FP32 twin (benign), P2/P7 16/16, decode
−1.31%. **Matched noise-protocol run: median +2.76%, paired-log G=1.0272
[1.0005,1.0546] → below the ≥5% gate → REJECT.** Split-K/finalize overhead
(−4% on the FP32 twin) eats most of the FP16 benefit (+8.7%). Guarded operator
**reverted**; promoted Stage 2 retained as the default prefill control.
Next variable per Stage 1 profile: fuse output rotation/scale into the GEMM
epilogue (finalize = largest fuseable bound, 6.7%), pending Sol PLAN gate.
Evidence: `evidence/EXP-04-stage3/2026-09-01/`.

### EXP-05 Phase 2 — reference mainloop audit: NO-GO (2026-09-01)

Read-only audit of official `escham_code_gemm` (wheel sha `735f4b7a…`,
runtime 623.8 ms vs Bee-Stage2 ~817–870 ms). Template `<A,K,BM,BN,BK,
FP16ACC,FEPI>` mapped (56 sm_120 symbols); default hot kernel `<1,K,128,64,2,
true,true>` = two 64-col bands, 80 regs, 45 KiB smem, fp16 acc. Official
faster on short-IC (gate 0.78×) but 1.78× slower on down_proj (its default
BM=128/BK=2 is a trap; BM=64 → 1.99 ms). **Sol Phase 3 gate NO-GO: measured
short-IC coverage = 10.96% aggregate matmul < 15% bar → EXP-05 closes as
negative evidence, no speculative two-band implementation.** Next targets
ranked: (1) fuse finalize into GEMM epilogue (6.7%, Sol PLAN READY), (2)
down_proj BM=64-style tile sweep (separate variable), (3) fuse input rotation
(4.6%). History corrected append-only (P-ARCH-09 short-IC = FP16 acc;
`-ub 2048` supersedes `-ub 512`). Evidence: `evidence/EXP-05-audit/2026-09-01/`.

## Closed work

- P-ARCH-22 (vocab representation): **CLOSED as size-capped** — do not reopen
  without new memory-tradeoff evidence.
- Quality benchmarking of the four control models on the 5-pack medium suite:
  **CERTIFIED** (this document). Do not re-run unless the runtime semantics
  change.
- The `escha_version` gating investigation: **RESOLVED** (root cause found,
  fix verified, recovery measured).

## Active next phase — ESCHA-W2 PREFILL

**Goal:** improve prefill throughput of the canonical full Escha W2 model inside
BeeLlama, preserving standard-Qwen compatibility and as much of the existing
low-bit Escha body as practical.

- Prefer runtime/kernel/layout/dispatch/fusion fixes over changing the model
  artifact.
- Avoid whole-model requantization/reconstruction drift unless measurements
  prove it necessary.
- Decode must not materially regress; quality/coherence must remain intact.
- Do not optimize around the benchmark.
- Reuse P-ARCH performance evidence without assuming their model substitutions
  are the desired final architecture.
- See `docs/escha-prefill-plan.md` (Codex/Sol-reviewed ESCHA-W2 PREFILL plan)
  for the ranked bottlenecks and the next experiment.
