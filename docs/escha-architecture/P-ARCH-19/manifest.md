# P-ARCH-19 — Escha mixed accumulator policy (mixed vs forced-FP32) at the matched 2k prefill boundary

**Status: CLOSED — arithmetic policy is first-order at the reference boundary.**
On the original Escha runtime, forcing FP32 accumulation for every prefill
shard costs **553.502 ms (+88.79% wall; 1.888x slower)** at the matched
2,048-token TTFT boundary: mixed **623.380 ms / 3285.31 tok/s** vs forced FP32
**1176.882 ms / 1740.19 tok/s**, with only `ESCHA_PREFILL_ACC` changed. The
run's decision threshold (≥50 ms) is cleared by more than 11x.

## Scope and control

Question (from `PREFILL-PARITY-REVIEW.md`): *How much does Escha mixed
accumulation buy by short-IC shape?* The same-shape K2/K3 residuals point here;
the result determines whether the remaining prefill defect is arithmetic policy
or kernel structure.

No CUDA kernel, model, loader, graph default, runtime wheel, or production
default was modified. Both arms ran the **same** model directory, serve.sh,
runtime wheel, and launch environment on the RTX 5090; the only changed
variable is `ESCHA_PREFILL_ACC` (`mixed` -> `fp32`). The mixed server was
stopped cleanly before the FP32 server was launched, so the A/B is
same-boundary and uncontaminated.

- Model: `weights/escha-w2-lowgpu-mono` (monolithic Escha checkpoint; shard
  SHA-256 `3677ea61...` / `e909dc0b...`, config `2fcf2d10...`, tokenizer
  `0997f410...`; re-verified 2026-08-30, matches the retained P-ARCH-05
  fingerprint record).
- Runtime: `runtime/sglang/serve.sh` (`6c478a47...`) +
  `escha-1.2.0+qwen3dense` wheel (`735f4b7a...`).
- Launch env (both arms): `ESCHA_PREFILL=fused ESCHA_PREFILL_FP16ACC=0
  ATTN_BACKEND=triton THINK=0 GRAPHS=1 CUDA_GRAPH_BS='1 2 4 8' CHUNK=2048
  RADIX=0 MEM=0.72 CTXLEN=65536 PORT=30019`.
- Input: exactly 2,048 prompt tokens + 1 completion token, temperature 0,
  streamed; token stream SHA-256
  `526dd2776e7a4840915ca478dc69cb4ecfb88539d83a7752b1f6ace9239d3fb9` and chat
  template SHA-256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`
  are identical in both arms.
- Runner: `evidence/P-ARCH-19/2026-08-30/{mixed-001,fp32-001}/run_controlled.py`
  (copied from the P-ARCH-05 controlled runner): one calibration, one warmup,
  three measured trials per arm; HTTP 200 and server-verified 2048-token prompt
  required for every trial.

## Independent variable and isolation

`ESCHA_PREFILL_ACC` has **one consumer and one call site** in the wheel wrapper
(`runtime/wheel-src/sglang_srt_layers_quantization_escha.py`):

1. `_PREFILL_ACC = os.environ.get("ESCHA_PREFILL_ACC", "mixed")` (line 227).
2. `_acc_mode_for(IC)` (lines 232-241): returns `1` (fp16 MMA accumulate) for
   `mixed` when `IC <= ESCHA_ACC_IC_MAX` (default 6144), else `0` (fp32); `fp16`
   forces `1`; anything else forces `0`.
3. The single call site passes the result as the **last op argument** of
   `torch.ops.escha.escham_code_gemm` (line 1042), i.e. accumulation is a
   per-call argument, not a process-global cuBLAS/torch mode.

`_PREFILL_ACC` is referenced nowhere else, so the knob cannot alter dispatch
predicates, K2/K3 geometry, CUDA-graph capture, workspace, or launch structure
— the adversarial-control claim, verified directly by the primary agent.
`ESCHA_PREFILL_FP16ACC=0` (the recon-path torch `allow_fp16_accumulation`
toggle) is pinned off in both arms and does not touch the fused path.

Resulting per-shard policy:

| mode | fp16-accumulate population | fp32-accumulate population |
|---|---|---|
| `mixed` | short-IC shards, `IC <= 6144` (including the 5120→17408 K2/K3 gate/up shapes) | long-IC shards (e.g. 17408→5120 down projections) |
| `fp32` | none | every shard |

## Matched 2k result (same session, same boundary, n=3 per arm)

| arm | trials (s) | median ms | median tok/s | vs mixed |
|---|---:|---:|---:|---:|
| `ESCHA_PREFILL_ACC=mixed` | 0.624288 / 0.622390 / 0.623380 | **623.380** | **3285.31** | — |
| `ESCHA_PREFILL_ACC=fp32` | 1.178824 / 1.176486 / 1.176882 | **1176.882** | **1740.19** | **+553.502 ms (+88.79%; 1.888x slower)** |

All six trials exited HTTP 200 with zero errors and server-verified exactly
2,048 prompt tokens + 1 completion token. Spreads are tight in both arms
(1.90 ms mixed, 2.34 ms fp32), so the 553.5 ms delta is ~240x the within-arm
spread. Reproducible medians/delta: `aggregate.py` / `aggregate.json`.

## Interpretation

The retained Escha reference (669.554 ms / 3058.75 tok/s, P-ARCH-05 lineage)
was itself measured with the mixed default at this same boundary; P-ARCH-19's
mixed re-measurement (623.380 ms / 3285.31 tok/s) sits inside that reference
family. Forcing FP32 makes the identical runtime ~1.76-1.89x slower than the
mixed/reference family — i.e. **the reference's speed is largely carried by the
mixed accumulator policy**. At the corresponding full-prefill-region boundary,
an all-FP32 Escha runtime (1740.19 tok/s) would sit below Bee's current best
prefill control (2228.75 tok/s), with the stated harness caveat (HTTP
TTFT-derived vs compute-only `llama-bench`; the project's P-ARCH-05 row treated
these as a matched full-prefill boundary with stated harness differences).

**Arithmetic policy vs structure:** the mixed policy is a first-order factor
with a magnitude (553.5 ms) larger than the entire remaining Bee residual
family (P-ARCH-16 249.9 ms, P-ARCH-17 175.1 ms, P-ARCH-18 75.5 ms vs LowGPU at
their respective boundaries). Structural factors are not exculpated — Bee's
ported path has its own kernel-structure deficits (P-ARCH-21 remains the
structural test) — but the parity target itself must first decide the arithmetic
contract: mirror the mixed policy (fp16 accumulate for short-IC shards, with
its documented numerics: mean-relative deviation 3.7e-3..6.9e-3, K-dependent;
mixed quality gates 72.0/107 PASS vs the 72.6 reference) or explicitly target
fp32 arithmetic and reset the goalposts.

Boundary caveat: the P-ARCH-19 rates are HTTP TTFT-derived and are NOT
comparable to Bee compute-only `llama-bench` tok/s; the A/B is internally valid
because both arms share the same harness, session, and one variable.

## Gate answers

1. **How much does Escha mixed accumulation buy at the matched 2k boundary?**
   **553.502 ms (+88.79% wall; 1.888x; -1545.12 tok/s)** at the identical
   HTTP TTFT boundary (3285.31 vs 1740.19 tok/s).
2. **Does `ESCHA_PREFILL_ACC` change anything besides accumulator mode?** No —
   single consumer (`_acc_mode_for`) and single call site (per-call op
   argument to `escham_code_gemm`); dispatch, geometry, graphs, and workspace
   are untouched.
3. **Which shapes does mixed affect?** Exactly the short-IC population
   `IC <= 6144` (including the 5120→17408 K2/K3 shapes); long-IC 17408→5120
   shards stay fp32 in both arms.
4. **Is the defect arithmetic policy or structure?** Policy is first-order at
   the reference boundary (553.5 ms ≫ 50 ms threshold). Structural deficits
   remain for Bee's port, but no parity claim can be made without first fixing
   the arithmetic contract.
5. **What next?** P-ARCH-20: one-shape Bee FP16 accumulator specialization on
   5120→17408 within quality bounds (opt-in, matched exactness + quality +
   prefill gate), or explicitly re-target parity at fp32 arithmetic.

## Advisory lanes

Six read-only Hermes/Nous DeepSeek V4 Flash lanes were started (implementation,
shape mapping, methodology, historical evidence, adversarial controls,
interpretation). Lanes 2/4/6 returned and independently confirmed that the only
intended runtime argument is the per-call accumulator mode and that the
short-IC K2/K3 shapes are the high-signal measurements. Lanes 1/3/5 were still
running when the prior session hit its usage limit and their outputs were not
persisted. Every decisive claim in this manifest (source isolation + paired
measurement) was verified directly by the primary agent, so the closure does
not depend on lane outputs.

## Evidence

`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-19/2026-08-30/`:

- `mixed-001/`, `fp32-001/` — `run_controlled.py`, `workload.json`,
  `prompt.txt`, `request.json`, `calibration.json`, `warmup.json`,
  `trial-1..3.json`, `summary.json`
- `aggregate.py`, `aggregate.json` — reproducible medians and delta
- `summary.md`, `launch-metadata.txt` — narrative + launch env + artifact hashes

**CLOSE P-ARCH-19 (arithmetic-policy factor established). NEXT_GATE: P-ARCH-20
— one-shape Bee FP16 accumulator specialization on 5120→17408 within quality
bounds (opt-in; matched exactness + quality + prefill gate), or explicitly
re-target parity at fp32 arithmetic.**
