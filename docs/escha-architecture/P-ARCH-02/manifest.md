# P-ARCH-02 — first post-fragment execution-path divergence

**Status:** `RESOLVED — FIRST DIVERGENCE PROVEN; CORRECTION UNSAFE ON CURRENT HOST`  
**Ledger row:** `Dispatch`  
**Question:** Where is the first execution-path divergence after P-ARCH-01's proven-equivalent HMMA B fragments?  
**Direct answer:** Bee's `mma_arch_ok` predicate excludes the legacy MMA path by default on Blackwell. For the representative 512-row operation on SM120, this single false predicate makes `use_mma=false` and selects `tiled-fma-fp32`; the Escha reference selects `escham_code_gemm<1,K,128,64,2,true,true>` containing `HMMA.16816.F16`.

P-ARCH-01 remains immutable. No packing, decode, or B-fragment work was repeated.

Raw evidence is at `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-02/2026-08-29/`.

## Run identity

| Field | Value |
| --- | --- |
| Date / operator | `2026-08-29 / Hermes (OpenAI GPT-5.6 Sol parent; four parallel delegated workers requested)` |
| BeeLlama revision | `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d` plus preserved local changes |
| Bee dispatch source | `ggml/src/ggml-cuda/escha-moe.cu`; SHA-256 `c8b609b6db63994cd60a3489de1dbfabb15858ba5a85da516da89c8e3fd05d3d` |
| Bee binary controls | Established 2k control: `build-cuda-verify/bin/llama-bench`, SHA-256 `d94d26ea0d603ee86ed3d947aec1c04a943800d5c2c22ea3ca7142614cc49ac6`; Codex fresh 512-row route proof: `build-cuda/bin/llama-bench`, SHA-256 `0283fb9a56b5544e14ac5b9bc052f8d42b27584a42477ad4963119763efe6114` |
| Model | `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf`; P-ARCH-01 SHA-256 `e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d` |
| GPU / runtime | RTX 5090, compute capability 12.0, driver 610.88, WSL; no `ESCHA_*` variables set at inspection time |
| Representative operation | `rows=512`, `IC=5120`, `OC=17408`, `K=2/3`; input activation F32; packed code/dep I16; rin/rout F16; dst F32 |
| Escha reference | Python dispatch `runtime/wheel-src/sglang_srt_layers_quantization_escha.py:1035-1043`; selected-symbol/SASS evidence in P-ARCH-01 `reference-sm120-hmma-evidence.txt` |
| Machine-readable result | external `dispatch-predicate-evaluation.json` |

## Architecture-diff ledger — P-ARCH-02 rows only

| Area | Bee | Escha | Equivalent? | Evidence | Expected Impact | Status |
|------|-----|-------|-------------|----------|-----------------|--------|
| Weight representation / decoded weights / B fragments | P-ARCH-01 proves checkpoint→GGUF payload bytes, all 256 decoded FP16 weights, and HMMA B-fragment bytes match. | Same logical FP16 weights and actual selected HMMA B ABI. | **PROVEN EQUIVALENT** | `docs/escha-architecture/P-ARCH-01/manifest.md` | Eliminates packing/permutation as blocker. | `CROSSED OFF` |
| Dispatch shape/type eligibility | `rows=512` gives `gen=false`; `K∈{2,3}`; `IC=5120`, `OC=17408`; `OC%128=0`; Turing and Blackwell MMA device code is compiled. | Fused `escham_code_gemm` requires `K∈{2,3}`, `IC%128=0`, `OC%128=0`; representative invocation selected it. | **PROVEN EQUIVALENT** for the compared shape/format gates | Bee `escha-moe.cu:1602-1625`; Escha Python `:1035-1043`; P-ARCH-01 selected-symbol evidence | These predicates do not explain the fallback. | `CROSSED OFF` |
| Architecture eligibility predicate | `mma_arch_ok = cc>=750 && (cc<1200 || ESCHA_FORCE_MMA)`; at `cc=1200`, env unset, it is false. | Exact selected SM120 fused kernels execute HMMA. | **PROVEN DIVERGENT — FIRST DIVERGENCE** | Bee `escha-moe.cu:1603-1612`; external predicate JSON; reference SM120 SASS | Sole failed Bee `use_mma` conjunct; directly prevents HMMA route. | `RESOLVED` |
| Dispatch decision / selected implementation | `use_mma=false`; route falls to `tiled-fma-fp32`; kernel `escha_matmul_dense_tiled<K,128,128,8,8>`. | `escham_code_gemm_kernel<1,K,128,64,2,true,true>`. | **PROVEN DIVERGENT** (downstream consequence) | Bee `escha-moe.cu:1621-1638,1805-1839`; route proof `/tmp/escha-prefill/P000-R1/route-proof.log`; reference selected symbols | Explains why Bee executes CUDA-core FMA instead of tensor-core HMMA. | `RESOLVED` |
| MMA primitive | Selected Bee route uses scalar FP32 FMA; unselected Bee route uses `load_ldmatrix` + `ggml_cuda_mma::mma`. | Selected symbols contain `HMMA.16816.F16`. | **PROVEN DIVERGENT** (downstream consequence) | Bee `escha-moe.cu:929,1111-1126`; P-ARCH-01 SM120 SASS | Large likely throughput consequence. | `RESOLVED` |
| Accumulator | Selected Bee path uses FP32 accumulators. Its unselected MMA path also declares FP32 `tile_c`. | Default mixed policy returns accumulation mode 1 for `IC=5120`; selected template's fifth argument is `2`, while complete native source is unavailable. | **PROVEN DIVERGENT** for policy mode; exact native accumulator implementation beyond SASS/template evidence is **UNKNOWN** | Bee `escha-moe.cu:838,1026-1030`; Escha Python `:227-240`; selected symbol | Downstream of first divergence; not expanded per stop condition. | `STOPPED` |
| Epilogue / fusion | Bee executes separate `escha_finalize_dense` after either prefill matmul route. | Reference fused op produces final F16 output; detailed native source is not shipped. | **UNKNOWN** | Bee `escha-moe.cu:1846-1848`; P-ARCH-01 output hashes | Downstream; not needed to identify first divergence. | `STOPPED` |

## Exact predicate evaluation

For `rows=512`, `IC=5120`, `OC=17408`, `K=2/3`, `cc=1200`, and no `ESCHA_*` variables:

- `gen = (512 <= 16) = false` — eligible for prefill routes.
- `use_cublas = false` — opt-in environment variable absent.
- `use_wmma_bw = false` — opt-in environment variable absent.
- `mma_arch_ok = (1200 >= 750) && ((1200 < 1200) || false) = false`.
- `OC % ESCHA_MMA_BN = 17408 % 128 = 0` — passes.
- `ESCHA_NO_MMA` absent — passes.
- Therefore `use_mma = false`, and the final dispatch branch selects `tiled-fma-fp32`.

The Blackwell clause in `mma_arch_ok` is the first meaningful post-fragment divergence. It is not an inferred blocker: every other `use_mma` conjunct is true and the recorded runtime route matches the source evaluation.

## Fresh Codex route proof recovered from session log

Codex rebuilt `build-cuda/bin/llama-bench` from this source (`0b035b3a2-dirty`, CUDA architectures `89;120a`) and ran a graph-disabled, profile-enabled 512-token sidecar. Although Codex's progress message said the run was still loading, the next tool result in the same JSONL records **status `completed`, exit code 0**.

- Command: `ESCHA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1 build-cuda/bin/llama-bench ... -p 512 -n 0 -r 1 -ngl 999 -fa on -ctk f16 -ctv f16 -o json`.
- Route count: `ROUTE_ROWS512_TOTAL 800`.
- Route identity: `ROUTE_ROWS512_COUNT tiled-fma-fp32 800` — **800/800** 512-row projection calls.
- Representative production shape: `ic=5120 oc=17408 rows=512 gen=0 route=tiled-fma-fp32`.
- Diagnostic throughput: one graph-disabled/profile-enabled 512-token sample, `614.562 tok/s`; this is route evidence, not the controlled 2k performance result.
- Evidence: external `codex-route-proof-512.log`, SHA-256 `8cec11aea89256c899842b5e6fd2447a8d1de8fad5d8b671ccfeb142dc9a3307`.

## Minimal correction and safety decision

The smallest route correction is already expressible without changing representation or kernels:

```text
ESCHA_FORCE_MMA=1
```

The source-equivalent one-line change would remove the `cc < GGML_CUDA_CC_BLACKWELL` exclusion from `mma_arch_ok`. Either correction would make `use_mma=true` for the representative operation and select `escha_matmul_dense_tiled_mma<K,128,128>`.

**It was not applied.** Existing experiment P001 records a reproducible indefinite WSL/SM120 stall under `ESCHA_FORCE_MMA=1`, with graphs both on and off. P002 replaced `cp.async` activation staging with synchronous loads and observed the same stall. P-ARCH-01 supplies new representation/layout evidence but does not contradict that execution-safety evidence. Enabling this route by default is therefore not an isolated safe correction on the current host.

Because the stop-condition correction is unsafe, no new build, correctness claim, or post-fix performance claim was made. This is intentionally not replaced with a speculative kernel rewrite.

## Controlled performance record

| Metric | Before | After |
| --- | --- | --- |
| Prefill contract | 2,048 prompt tokens, `-b 2048 -ub 512`, F16 KV, FA on, graph mode enabled | Not run — correction not safely implemented |
| Samples | `666.312 / 653.131 / 655.468 tok/s` | N/A |
| Median | `655.468 tok/s` | N/A |
| Kernel/path | `tiled-fma-fp32` | Expected `mma-fp16`, but not executed in this experiment |
| Correctness | Existing P1/P2/P5/P6/P7 baseline pass | N/A |

## Decision

- P-ARCH-01 representation/layout/fragments: **MATCH**.
- P-ARCH-02 first execution divergence: **Bee Blackwell architecture eligibility gate** at `escha-moe.cu:1611-1612`.
- Root cause: `cc=1200` fails `(cc < GGML_CUDA_CC_BLACKWELL || ESCHA_FORCE_MMA)` with the opt-in absent.
- Minimal route correction: `ESCHA_FORCE_MMA=1` or source-equivalent gate removal.
- Safety: **not safe on current WSL/SM120 evidence**; prior P001/P002 hangs remain unrebutted.
- Next investigation, only if authorized: diagnose the already-isolated Blackwell MMA-kernel stall. Do not revisit weight packing.
