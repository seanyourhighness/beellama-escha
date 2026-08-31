# BeeLlama Escha architecture-diff ledger — handoff

## Scope

Eliminate architectural unknowns by comparing one operation at a time. The only
question that advances a row is:

> Did this make the data presented to the kernel match the control?

Do not start tile-size sweeps, throughput tuning, or unrelated runtime work until
the next `NEXT` row has a measured answer. `RESOLVED` and `NOT A CONTROL` rows are
not active work.

## Controls and artifacts

| Item | Path / fact | Use |
| --- | --- | --- |
| Vanilla runtime control | `/home/sean/models/qwen3.8-27b-vanilla/Qwen3.8-27B-IQ4_NL.gguf` | Same BeeLlama CUDA runtime; valid dispatch/serving sanity control only. |
| Hybrid under study | `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf` | Qwen3.5 hybrid, 64 layers, 5,120 hidden, 17,408 FFN, packed Escha + LowGPU. |
| False dense control | `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-dequant.gguf` | Not a body-dense control: it still contains `*.escha_code`, `*.escha_rin`, and `*.escha_rout`; only the LowGPU vocabulary is F16. |
| Active source | `/mnt/d/CODEX WORKSPACE/beellama-escha` | Detached base `0b035b3a2` plus local Escha work. |
| Runtime reference | Local Escha/SGLang wheel source under `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/runtime/wheel-src/` | Consult only for the row being tested. |

The vanilla model is Qwen3.8 while the hybrid is Qwen3.5/Gated DeltaNet. It is
therefore **not** a per-tensor numerical control. Use it only to establish that
BeeLlama's standard CUDA path and host scheduling are healthy. A true per-operation
control still needs a dense materialization of the same Qwen3.5/Escha weights.

## Architecture-diff ledger

| area | vanilla | hybrid | Escha | status | impact | notes |
| --- | --- | --- | --- | --- | --- | --- |
| Control topology | Qwen3.8 IQ4_NL uses BeeLlama's normal Qwen graph. | Qwen3.5 hybrid has 48 GDN + 16 full-attention layers. | Qwen3.5 hybrid reference runtime. | `NOT A DIRECT TENSOR CONTROL` | critical | Do not attribute an operation-level difference to Escha until a same-topology dense control exists. |
| Dense-control artifact | Standard GGML quant weights are directly consumed by standard CUDA kernels. | `mono-dequant.gguf` still has packed body `escha_code/rin/rout`. | Reference consumes packed Escha body. | `ELIMINATED` | critical | The existing “dequant” artifact is LowGPU-vocabulary-only; it cannot prove body-layout equivalence. |
| Escha weight packing | GGML quant blocks are in the layout expected by their native CUDA kernel. | Production layer-0 K2 gate and K3 up tile `(0,0)` payloads match checkpoint/GGUF bytes; all 256 FP16 weights and captured HMMA B-fragment bytes match the Escha control. | `escham_reconstruct` is bit-identical and actual M=512 `escham_code_gemm<1,K,128,64,2,true,true>` uses `HMMA.16816.F16` with the recorded B-fragment ABI. | `RESOLVED` | critical | P-ARCH-01 evidence: `docs/escha-architecture/P-ARCH-01/manifest.md`. Did this make the data presented to the kernel match the control? **Yes.** No inverse permutation is required; proceed only to activation/rotation. |
| Fragment layout | Vanilla kernel owns its established CUDA fragment layout. | P-ARCH-01 proves the candidate MMA path's decoded B-fragment bytes match the reference HMMA ABI exactly. | Actual M=512 K2/K3 kernels use the same recorded HMMA B-fragment ABI. | `RESOLVED — PROVEN EQUIVALENT` | critical | `docs/escha-architecture/P-ARCH-01/manifest.md`; no inverse permutation is required. |
| Activation/rotation contract | Standard Qwen projection receives the graph activation expected by its weight kernel. | `escha_rotate_in_dense<float>` applies folded `rin` + Hadamard before safe FMA; the unselected MMA route changes activation storage to F16. | Reference fused op converts the input to F16 before `escham_code_gemm`; exact end-to-end activation parity remains unproved. | `OPEN — DOWNSTREAM OF P-ARCH-02` | high | Stop condition reached at dispatch eligibility; do not broaden into this row until the Blackwell MMA execution stall is isolated. |
| Projection epilogue | Vanilla graph uses its native bias/norm/SwiGLU composition. | Hybrid has separate packed matmul partials, fixed finalize/Hadamard/rout, then graph-level SwiGLU. | Reference exposes a fused Escha output path; native source is not shipped. | `OPEN — DOWNSTREAM OF P-ARCH-02` | high | Not needed to identify the first divergence. |
| Dispatch | Normal GGML CUDA dispatch for its quant type and shape. | At `rows=512`, `IC=5120`, `OC=17408`, `K=2/3`, every MMA shape/type predicate passes, but `mma_arch_ok` is false because `cc=1200` and `ESCHA_FORCE_MMA` is unset; this selects `tiled-fma-fp32`. | The same representative operation selects `escham_code_gemm<1,K,128,64,2,true,true>` with `HMMA.16816.F16`. | `RESOLVED — PROVEN DIVERGENT` | critical | **First divergence:** `escha-moe.cu:1611-1612`. P-ARCH-02: `docs/escha-architecture/P-ARCH-02/manifest.md`. Minimal route correction is `ESCHA_FORCE_MMA=1`/gate removal, but P001/P002 record a reproducible WSL/SM120 stall, so it was not applied. |
| LowGPU vocabulary head | Dense F16/Qwen output matrix control semantics. | Packed 3-bit LowGPU head; striped CUDA path. | LowGPU format reference. | `RESOLVED — IGNORE` | low for current body gap | Direct six-token CUDA test passed against independent packed reference; this is not the remaining body-kernel unknown. |
| GDN graph semantics | Not comparable: vanilla model lacks the same Qwen3.5 GDN topology. | Corrected GDN mapping/head interleave has prior token-parity evidence. | Qwen3.5/GDN reference. | `OUT OF CURRENT DIFF` | high but bounded | Reopen only if the same-topology dense control disagrees before an Escha projection. |

## Completed evidence retained from prior work

- Packed-tile logical decode against the reference reconstruct op: K2/K3 random
  payload maximum difference `0.0`.
- Qwen3.5 hybrid prefill reaches 512-row Escha calls. Current safe route proof is
  `tiled-fma-fp32`; its 2k production control median is 655.468 prompt tok/s.
- LowGPU six-token CUDA prefill test passes the independent packed reference.
- P001/P002 previously classified forced legacy MMA as a WSL/SM120 stall. P-ARCH-03
  disproved that classification for the current binary: forced MMA completed with
  exit 0, selected `mma-fp16` for 800/800 calls, passed P1/P2/P5/P6/P7 at 16/16,
  and measured 1243.72 tok/s median at the controlled 2k gate.
- Four-warp WMMA races output; transposed WMMA A storage diverges; FP16 activation
  WMMA fails token parity. Those findings remain closed.

## P-ARCH-04 resolved — SM120 MMA qualified and defaulted

P-ARCH-01 through P-ARCH-03 remain closed. P-ARCH-04 qualified the existing
`escha_matmul_dense_tiled_mma<K,128,128>` route across all 400 production Escha
projections (K2/K3; seven runtime IC/OC pairs), prefill rows 17/128/511/512, real
scheduler tails, prompts through 4096, three repeated correctness runs, and 40,800
profiled MMA calls without detected CUDA errors.

The minimal dispatch change makes MMA default through exactly SM120:

```cpp
cc >= GGML_CUDA_CC_TURING &&
(cc <= GGML_CUDA_CC_BLACKWELL || ESCHA_FORCE_MMA)
```

Generation rows <=16 remain on `gen-splitk-fp32`; `ESCHA_NO_MMA=1` retains
`tiled-fma-fp32`; unqualified architectures above SM120 remain opt-in. Post-change
automatic-route P1/P2/P5/P6/P7 passed 16/16 in 3/3 runs. The official controlled
2k baseline is now **1230.03 tok/s** (`1246.82 / 1230.03 / 1217.32`), versus
655.468 tiled-FMA (1.8766x / +87.657%). Evidence:
`docs/escha-architecture/P-ARCH-04/manifest.md`.

## P-ARCH-03 resolved — forced SM120 MMA completes

P-ARCH-01 and P-ARCH-02 remain closed. P-ARCH-03 established with hard process
evidence that the current `ESCHA_FORCE_MMA=1` route completes on SM120:

- graph-disabled profiled 512-token probe exited 0 in 145 seconds;
- captured process terminated, output was complete, and no benchmark/GPU compute
  process remained;
- `mma-fp16` selected for 800/800 512-row calls;
- diagnostic throughput was 894.768 tok/s;
- P1/P2/P5/P6/P7 matched the Escha/SGLang reference 16/16, including P5 at 1,544
  prompt tokens;
- controlled graph-mode 2k samples were 1243.72 / 1229.43 / 1254.32 tok/s, median
  1243.72 versus normal 655.468: 1.897x / +89.745%.

No execution-progress divergence was found and no kernel-internal instrumentation was
warranted. The remaining question is dispatch policy/safety validation: whether and
how to promote the currently opt-in route. P-ARCH-03 made no global source-default
change. Full evidence: `docs/escha-architecture/P-ARCH-03/manifest.md`.

## Handoff protocol

After every experiment, update exactly one affected ledger row with `status`,
`impact`, artifact path, and a direct answer to “Did this make the data presented to
the kernel match the control?” Then mirror this file to GBrain page
`projects/beellama-escha-architecture-diff-ledger`.

## P-ARCH-05 — exact shared-token reference result

| Layer | Bee | Escha | Comparable? | Evidence | Result |
|---|---|---|---|---|---|
| tokenization | persisted 2,048 raw IDs | same text independently re-tokenized to the same IDs | `PROVEN EQUIVALENT` | `P-ARCH-05/.../escha-controlled-server-002/shared-2048.{txt,ids,json}` | ID SHA-256 `695c3609bc35a32003a23be3ba1fbacc16cc94955548c2e855e91661c3f62350`. |
| workload | `llama-bench --prompt-tokens-file`, batch 2048 / ubatch 512, F16 KV, FA | `sglang.bench_one_batch`, static batch 1 / input 2048 / output 1 | `PROVEN EQUIVALENT` prefill input; harnesses intentionally differ | raw result files in `escha-controlled-server-002` | Both consume the same IDs after warmup. |
| HTTP timing | N/A | median TTFT 0.624309 s | `NOT COMPARABLE` | `summary.json` | Retained separately and not used in the result below. |
| prefill-region timing | 1243.98 / 1225.83 / 1249.86 tok/s; median 1243.98 | 3005.18 / 3058.75 / 3319.13 tok/s; median 3058.75 | `PROVEN EQUIVALENT` timing scope | `bee-shared-input-stdout.json`; `escha-shared-input-prefill.jsonl` | Median full-prefill duration 1.646329 s vs 0.669554 s (Escha/Bee 2.458842×). |
| selected path | `escha_matmul_dense_tiled_mma<K,128,128>` | fused `escham_code_gemm`; Triton attention | `PROVEN DIVERGENT` | P-ARCH-04 route evidence; `server.log` | Full-prefill result, not a single-operator measurement. |
| representative GEMM duration | unavailable | unavailable | `UNKNOWN` | N/A | Do not imply a per-kernel ratio. |

## P-ARCH-06 — matched prefill execution attribution

| Stage | Bee total ms | Escha total ms | Delta ms | Bee/Escha | Deficit share | Status |
|---|---:|---:|---:|---:|---:|---|
| W2 dense linear / GEMM | 1318.314 | 516.565 | **801.749** | 2.552× | **82.08%** | `RESOLVED — DOMINANT` |
| Non-W2-linear residual | 328.015 | 152.989 | 175.026 | 2.144× | 17.92% | `BOUNDED — NOT SPLIT WITHOUT BEE GPU TRACE` |
| **Matched full prefill** | **1646.329** | **669.554** | **976.775** | **2.459×** | **100.00%** | `RECONCILED` |

Direct CUDA-event aggregation shows Bee runs 1,600 512-row dense projections
through `escha_matmul_dense_tiled_mma<K,128,128>` for 1318.314 ms (1221.119 ms
inside the MMA body). Escha's prefill-only Torch GPU trace shows 400 fused
`escham_code_gemm` launches for 516.565 ms. The 801.749 ms direct-linear delta
is 82.08% of the P-ARCH-05 deficit, so P-ARCH-06 is closed without a code
change. Evidence: `docs/escha-architecture/P-ARCH-06/manifest.md` and
`P-ARCH-06/2026-08-29/trace-001/`. **NEXT_GATE: P-ARCH-07 — root-cause this
dominant W2 linear/GEMM divergence and identify the smallest isolated
correction.**

## P-ARCH-10 — K2 SM120 cross-tile overlap

| area | Bee baseline | SM120 async-overlap candidate | status | impact | notes |
| --- | --- | --- | --- | --- | --- |
| activation staging | synchronous `uint4` copy; no cross-tile overlap | existing double-buffered `cp.async` activation path selected only by `ESCHA_MMA_SM120_ASYNC_EXPERIMENT` | `RESOLVED — CASE A` | critical | Representative K2 5120→17408 M=2048: 3.812000→0.651196 ms, 95.78→560.62 TFLOP/s; P1/P2/P5 16/16, 3/3 complete. |
| B fragment layout / shared-B | `s_w` materialization then `ldmatrix` | unchanged | `NOT TESTED IN P-ARCH-10` | high | Direct-fragment decode was explicitly not combined with overlap. |

Did this make the data presented to the kernel match the control? **Yes for the
already-proven K2 B fragment mapping; the change only overlaps A staging.**
Full 2,048-token prefill measured 1761.71 tok/s versus the P-ARCH-04 1230.03
tok/s baseline (1.43225x, +43.23%).  Evidence:
`docs/escha-architecture/P-ARCH-10/manifest.md`.

**NEXT_GATE: P-ARCH-11** — validate the corrected path across K2/K3, prompt
lengths, ubatch sizes, memory/workspace, repeated stability, and numerical
regressions before default promotion.

## P-ARCH-11 — broad overlap validation and residual attribution

| area | Bee old | Bee async | Escha | status | impact | notes |
| --- | ---: | ---: | ---: | --- | --- | --- |
| K2 W2 aggregate MMA body | 650.211 ms | 384.887 ms | matched K2 subset retained | `PROVEN IMPROVED` | critical | 1.689x across six actual K2 shapes, 1,088 calls (P-ARCH-12 re-summed retained records; prior 1,472 was clerical). |
| K3 W2 aggregate MMA body | 555.989 ms | 321.249 ms | matched K3 subset retained | `PROVEN IMPROVED` | critical | 1.731x across both actual K3 shapes, 512 calls. |
| all W2 operator stage | 1302.513 ms | 799.633 ms | 516.565 ms | `PROVEN RESIDUAL` | critical | Async removes 502.880 ms; 283.068 ms remains versus Escha. |
| full matched prefill | baseline P-ARCH-04 1230.03 tok/s | 1761.71 tok/s | 3058.75 tok/s | `PROVEN IMPROVED` | high | Async 1.43225x full-prefill improvement; direct K2 W2 residual is now largest measured divergence. |

Did this make the data presented to the kernel match the control? **The already
proven K2/K3 fragment representation remains unchanged; the measured correction
only restores A-stage overlap.**  K2/K3 P1/P2/P5/P6/P7 deterministic prefixes,
prompt and ubatch envelope, completion, and CUDA safety checks pass.  The path
is a production-correction candidate, not yet the default.

Evidence: `docs/escha-architecture/P-ARCH-11/manifest.md`.
**NEXT_GATE: P-ARCH-12 — diagnose the bounded remaining K2 W2 code-GEMM
divergence; do not optimize it in that gate.**

## P-ARCH-12 — K2 W2 residual root-cause diagnosis

| area | Bee K2 | Escha K2 | status | impact | notes |
| --- | --- | --- | --- | --- | --- |
| Aggregate K2 residual distribution | Six 512-row shape aggregates total 442.862 ms; Bee invokes each logical M=2048 operation as four 512-row calls. | Grid-correlated fused trace totals 185.725 ms; 272 events total. | `RESOLVED — FAMILY-WIDE` | critical | Matched shape deltas total 257.137 ms. 5120→17408 is largest at 106.828 ms / 41.55%, but the next three rows lift the cumulative share to 87.51%; this is not a single-shape anomaly. |
| K2 execution geometry/codegen | `escha_matmul_dense_tiled_mma<2,128,128>`; 256 threads, 128 regs/thread, 13,824 B dynamic shared memory; rotate → MMA partial → finalize remain separate. | `escham_code_gemm_kernel<1,2,128,64,2,true,true>`; 256 threads, 80 regs/thread, 45,056 B shared; fused code-GEMM boundary. | `PROVEN DIVERGENT — H/F/C MIXED` | critical | First material K2-specific execution difference after overlap. It covers the 257.137 ms affected boundary; MMA-body portion is 199.162 ms (77.45%) and separately timed Bee rotate+epilogue is 57.975 ms (22.55%). These are boundary contributions, not a claim that registers alone cause the MMA delta. |
| Bee K2 decode and synchronization source path | K2/K3 share `escha_matmul_dense_tiled_mma<K,128,128>`; K changes packed-word constants only. DPT=8, shared-B/`ldmatrix`/HMMA and CTA barriers are structurally the same. | K2 uses its own templated 128x64 fused reference kernel. | `ELIMINATED AS FIRST PROVEN DIVERGENCE` | high | No source evidence supports a Bee-only K2 decode branch, fragment construction path, or barrier. Hardware-counter claims remain unavailable (`ERR_NVGPUCTRPERM`). |

P-ARCH-12 made no kernel or production-default change. The smallest plausible
P-ARCH-13 correction is a **K2-only** isolated validation of 128x64 geometry
and then, separately, fused-boundary work; K3 and the qualified SM120 async
overlap path remain out of scope. Evidence:
`docs/escha-architecture/P-ARCH-12/manifest.md`.

**NEXT_GATE: P-ARCH-13 — implement and validate only the smallest isolated K2
geometry/codegen correction proven by P-ARCH-12.**

## P-ARCH-13 — K2-only 128x64 output-geometry experiment

| area | Bee K2 128x128 control | K2 128x64 experiment | Escha | status | impact | notes |
|---|---:|---:|---:|---|---|---|
| K2 stage aggregate (rotate+MMA+epilogue, 4-run median) | 442.713 ms | 465.683 ms | 185.725 ms | `RESOLVED — PROVEN NON-DOMINANT` | critical | Symmetric repaired-profiler captures (3,200/3,200 records, all `mma-fp16`, exit 0). Geometry explains **-8.93%** of the 257.137 ms residual (adds 22.969 ms); MMA body +18.713 ms, rotate+epilogue +3.546 ms. |
| Uninstrumented full prefill | 1729.79 tok/s median (10 samples) | 1716.34 tok/s median (10 samples) | 3058.75 tok/s (P-ARCH-05) | `PROVEN NON-DOMINANT` | high | -0.78%; do not use for K2 attribution. |
| K2 codegen register/lifetime | `<2,128,128>`, 128 regs/thread | `<2,128,64>`, 85 regs/thread | `<1,2,128,64>`, 80 regs/thread | `UNKNOWN — NOT THE FIRST DIVERGENCE` | high | Register reduction does not convert to speed; doubled CTA count re-stages the A tile. |
| Deterministic correctness | P1/P2/P5/P6/P7 16/16 | 16/16 | reference | `PROVEN EQUIVALENT` | critical | Both current binaries pass after the profiler-only source change. |
| K3 / production defaults | `<3,128,128>` unchanged | `<3,128,128>` unchanged | — | `PROVEN UNCHANGED` | critical | Cubin symbols verified; no default promoted. |

Did this make the data presented to the kernel match the control? **The already
proven fragment representation is unchanged; the experiment only changed K2
output-tile width, and the symmetric measurement shows that geometry does not
close the measured gap.** P-ARCH-13 is closed as CASE C. The residual remains
attributable to the Bee rotate -> MMA partial -> finalize structure versus the
Escha fused code-GEMM. Evidence:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-13/2026-08-29/`
(`profiler-reuse-003/`, `uninstrumented-ab-001/`, `parity-reuse-001/`,
`analysis-001/`).

**NEXT_GATE: P-ARCH-14 — isolate the remaining K2 execution structure: Bee
rotation / MMA partial output / finalize and intermediate memory traffic versus
the Escha fused output transform/code-GEMM; identify the smallest isolated
fusion/lifetime correction.**

## P-ARCH-14 — remaining K2 execution structure (fused-finalize experiment)

| area | Bee control (128x128) | Fused single-slice finalize | Escha | status | impact | notes |
|---|---:|---:|---:|---|---|---|
| K2 rotate boundary | 31.949 ms | unchanged | fused in-kernel | `RESOLVED — BOUNDED` | medium | 12.4% of residual; input transform + u_buf round trip. |
| K2 MMA body | 385.378 ms | 397.200 ms (includes fused epilogue) | 185.725 ms | `RESOLVED — DOMINANT` | critical | 77.7% of residual; unchanged by geometry (P-ARCH-13) or boundary fusion (P-ARCH-14). |
| K2 finalize boundary | 25.565 ms | ~0.1 ms (fused) | fused in-kernel | `RESOLVED — NEUTRAL` | medium | In-kernel Hadamard+staging costs ~equal the separate finalize + fp32 partial read it removes. |
| K2 stage aggregate | 442.713 ms | 444.051 ms | 185.725 ms | `PROVEN NEUTRAL` | critical | -1.338 ms (-0.52% of residual), within run-to-run noise; 4 cold runs each. |
| Uninstrumented full prefill | 1729.79 tok/s | 1746.55 tok/s | 3058.75 tok/s | `PROVEN NEUTRAL` | high | +0.97%, same-session noise. |
| Deterministic correctness | 16/16 | 16/16 | reference | `PROVEN EQUIVALENT` | critical | Parity + compute-sanitizer 0 errors on the fused binary. |

Did this make the data presented to the kernel match the control? **The fused
finalize preserves the exact kernel inputs and output transform order; it only
changes where the output transform executes, and the symmetric measurement
shows that boundary is not the K2 divergence.** The smallest isolated
fusion/lifetime correction is measured neutral and rejected for production
assembly; the MMA-body execution remains the dominant residual, with
Escha-style decode/staging structure (direct-fragment decode, 45 KB shared)
as the remaining untested difference. Evidence:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-14/2026-08-29/`.

**NEXT_GATE: P-ARCH-15 — assemble the production-candidate experimental
configuration from ONLY individually proven corrections (SM120 MMA default,
SM120 async overlap, `-ub 2048` for matched 2k prefill) and run the regression
matrix (prompt 512/1024/2048/4096, ubatch 512/1024/2048, K2/K3).**

## P-ARCH-15 — production-candidate prefill regression matrix

| prompt | ubatch 512 tok/s | ubatch 1024 tok/s | ubatch 2048 tok/s | status | notes |
|---|---:|---:|---:|---|---|
| 512 | 1480.86 | 1686.44 | 1723.18 | `PASS` | exact shared-ID prefix |
| 1024 | 1691.81 | 1984.52 | 2068.16 | `PASS` | exact shared-ID prefix |
| 2048 | 1720.57 | 2046.08 | 2228.75 | `PASS` | reproduces P-ARCH-11 2235.26 (-0.3%) |
| 4096 | 1717.31 | 2068.94 | 2237.62 | `PASS` | exact shared-ID cycle |

Candidate = P-ARCH-04 MMA default + P-ARCH-10/11 async overlap + `-ub 2048`
(K2 128x64 and fused-finalize rejected). All twelve cells exit 0 with zero
CUDA-error lines; profiled 2k/ub2048 selects `mma-fp16` 800/800 at rows=2048;
parity 16/16 retained on the exact binary. 2k/ub2048 = **2228.75 tok/s**
(3.40x old tiled-FMA 655.468; 1.81x production default 1230.03; 72.9% of Escha
3058.75).

Did this make the data presented to the kernel match the control? **The
candidate preserves the proven fragment representation and only restores the
qualified A-stage overlap and ubatch envelope; it passes the regression
matrix.** Production defaults remain unchanged. Evidence:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-15/2026-08-29/regression-matrix-001/`.

**NEXT_GATE: P-ARCH-16 — final prefill attribution with the best proven
experimental configuration; determine whether further prefill optimization is
worthwhile before shifting to decode.**

## P-ARCH-16 — final prefill attribution (prefill investigation CLOSED)

| Stage | Bee ms | Escha ms | Delta ms | % of delta | status |
|---|---:|---:|---:|---:|---|
| K2 W2 | 350.756 | 185.725 | 165.031 | 66.2% | `BOUNDED — NOT CORRECTABLE BY GEOMETRY/FUSION` |
| K3 W2 | 300.377 | 330.840 | -30.463 | -12.2% | `PROVEN EQUIVALENT (Bee faster)` |
| all W2 | 651.132 | 516.565 | 134.567 | 54.0% | `RESOLVED` |
| non-W2 residual | 267.788 | 152.989 | 114.799 | 46.0% | `BOUNDED — UNSPLIT` |
| **full matched prefill** | **919.410** | **669.554** | **249.856** | **100.00%** | `CLOSED` |

Best proven configuration (async overlap + `-ub 2048`): Bee full prefill
2228.75 tok/s = **72.8% of Escha**, up from 40.7% at the P-ARCH-05 baseline.
The remaining delta is fragmented (K2 66%, non-W2 46%, K3 negative); the two
bounded K2 corrections (geometry, boundary fusion) were measured and rejected,
and the remaining Escha-style decode/staging change is beyond a bounded
experiment. Prefill architecture investigation CLOSED.

Did this make the data presented to the kernel match the control? **The
final-attribution configuration preserves the proven fragment representation
and the qualified overlap/ubatch envelope; no further bounded prefill
correction is available to close the remaining K2 gap.** Evidence:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-16/2026-08-29/final-attribution-001/`.

**NEXT_GATE: D-ARCH-01 — matched decode baseline (same model/GPU/token stream;
measure c=1, then c=2/4/8; no optimization yet).**

## D-ARCH-01 — matched decode baseline

| concurrency | Bee aggregate tok/s | Escha aggregate tok/s | Bee/Escha | status |
|---|---:|---:|---:|---|
| 1 | 38.17 | 60.88 | 0.627 | `PROVEN DIVERGENT` |
| 2 | 50.42 | 105.28 | 0.479 | `PROVEN DIVERGENT` |
| 4 | 58.28 | 247.19 | 0.236 | `PROVEN DIVERGENT` |
| 8 | 59.77 | 496.36 | 0.120 | `PROVEN DIVERGENT` |

Bee decode saturates near ~60 tok/s aggregate (step latency 22 -> 98 ms from
c=1 to c=8); Escha scales near-linearly via CUDA-graph batch capture (step
latency stays ~12-14 ms; server-log steady throughput ~86/151/290/584 tok/s).
Bee decode W2 path is `gen-splitk-fp32` for 4,000/4,000 profiled projections;
Escha uses its fused multi-shard decode kernels under CUDA graphs. Same model,
GPU, prompt, temperature 0, max_tokens 128.

Did this make the data presented to the kernel match the control? **The
baseline establishes the matched workload boundary; the gap is both per-kernel
(decode W2/attention) and launch/graph-batching.** Evidence:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/D-ARCH-01/2026-08-29/`.

**NEXT_GATE: D-ARCH-02 — decode operator attribution: per-step W2 (K2/K3),
attention, KV-cache, norms, output/head, copies, scheduler/graph gaps.**

## D-ARCH-02 — decode operator attribution

| Decode stage | Bee ms/step | Escha ms/step | Delta | % of deficit | status |
|---|---:|---:|---:|---:|---|
| W2 K2+K3 decode GEMM | >= 17.87 | ~9.7 | ~8.2 | ~82% | `RESOLVED — DOMINANT` |
| non-W2 (attn/GDN/head/norm/copy) | <= 3.9 | ~1.9 | ~2.0 | ~18% | `BOUNDED` |
| **step total** | **21.74** | **11.6** | **10.1** | **100%** | `RECONCILED` |

Bee W2 decode = 400 projections/step x (rotate + `gen-splitk-fp32` matmul +
finalize); per-family min-duration floor 17.874 ms/step of the 21.74 ms step
(>=82%). Escha W2 = fused `escham_gemv_bw` + `had_in` + `had_epilogue`,
83.4% of a 12-step torch trace. W2 is the dominant stage on BOTH runtimes and
carries ~82% of the deficit; prefill root causes do not carry over (the decode
W2 path is a different kernel family).

Did this make the data presented to the kernel match the control? **No source
change; the attribution shows the decode W2 kernel path is the first material
divergence to map in D-ARCH-03.** Evidence:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/D-ARCH-02/2026-08-29/`.

**NEXT_GATE: D-ARCH-03 — decode W2 path mapping: actual Bee decode kernel
(gen-splitk-fp32), launch geometry, registers, per-call timing, vs Escha
`escham_gemv_bw`; find the first material divergence.**

## D-ARCH-03 — decode W2 path mapping

| property | Bee decode W2 | Escha decode W2 | status |
|---|---|---|---|
| kernel family | `escha_matmul_dense<K,1>` gen-splitk-fp32 (R=1) | `escham_gemv_bw<1,K>` + `escham_mma_gemv<1,K>` (fused had_in/had_epilogue) | `PROVEN DIVERGENT` |
| launch structure | 400 x (rotate + matmul + finalize) triples/step, split-K up to 18 | fused multi-shard, CUDA-graph batches 1-8 | `PROVEN DIVERGENT` |
| registers / shared | K2 64 / K3 55 regs, 1,024 B | 64 regs, 2,048 B (gemv_bw), 256 threads | `NOT THE FIRST LEVER` |
| per-projection floor | ~0.047 ms (K2 5120->17408 incl. rotate+finalize) | ~0.024 ms equivalent | `PROVEN DIVERGENT` |
| experimental Bee alt | `escha_matmul_dense_warp<K>` (33/37 regs) opt-in `ESCHA_WARP_GEMV=1` | — | candidate for D-ARCH-05 |

The first material decode divergence is the W2 kernel family/launch structure:
Bee's generic R=1 FMA triple per projection versus Escha's specialized fused
Blackwell GEMV kernels. Evidence:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/D-ARCH-03/2026-08-29/`.

**NEXT_GATE: D-ARCH-04 — launch/graph/fusion comparison: quantify Bee CUDA-graph
benefit (graphs ON vs ESCHA_DISABLE_CUDA_GRAPHS), per-step launch counts, and
whether Escha removes launch boundaries Bee still pays.**

## D-ARCH-04 — decode launch / graph / fusion comparison

| concurrency | graphs ON tok/s | graphs OFF tok/s | ratio | status |
|---|---:|---:|---:|---|
| 1 | 34.14 | 21.05 | 1.62x | `PROVEN — GRAPHS ACTIVE` |
| 4 | 52.31 | 51.68 | 1.01x | `PROVEN — CONCURRENCY-BOUND` |

Bee captures and replays a CUDA graph once per decode step for the escha W2
ops; disabling graphs roughly doubles single-stream step time. The remaining
step gap (21.7 vs 11.6 ms) is inside the replayed graph — the W2 kernel family
and ~1,300 graph-internal launches — not missing graph capture. Escha replays
its fused `gemv_bw` graph at batches 1-8.

Did this make the data presented to the kernel match the control? **No source
change; the graph mechanism is already exercised, so the divergence remains the
W2 kernel family mapped in D-ARCH-03.** Evidence:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/D-ARCH-04/2026-08-29/`.

**NEXT_GATE: D-ARCH-05 — implement and validate the smallest decode correction:
enable the existing opt-in Blackwell warp-GEMV decode path
(`ESCHA_WARP_GEMV=1`), one variable, with parity + stability + throughput.**

## D-ARCH-05/06 — smallest decode correction (ESCHA_WARP_GEMV) and scaling

| concurrency | gen-splitk step ms | warp step ms | step delta | warp wall delta | status |
|---|---:|---:|---:|---:|---|
| 1 | 22.3 | 19.7 | -11% | -14% (TTFT 1.40 vs 0.54 s) | `CORRECT, NOT A WALL WIN` |
| 2 | 34.3 | 29.5 | -14% | -18% | `NOT PROMOTED` |
| 4 | 54.8 | 45.2 | -17.6% | -13% | `NOT PROMOTED` |
| 8 | 96.8 | 76.5 | -21% | +4% | `SATURATION ~60-70 TOK/S` |

The warp-GEMV path passes deterministic parity P1/P2/P5/P6/P7 16/16 and
improves per-step latency at every concurrency, but TTFT/wall aggregate is
worse at c=1-4, so it is not promoted. The W2 decode GEMM remains the dominant
stage (>=82% of step; Escha 83.4%); remaining step deficit ~1.7x at c=1 and
~8x aggregate at c=8. Closing it requires a dedicated Blackwell batched-GEMV
decode kernel — beyond a bounded experiment.

Did this make the data presented to the kernel match the control? **The
corrected path preserves outputs (parity 16/16) and improves steady-state step
latency but does not close the wall gap; it is measured and rejected for
promotion.** Evidence:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/D-ARCH-05/2026-08-29/`.

**NEXT_GATE: D-ARCH-08 — production regression of the combined candidate
(prefill async + ub2048; decode default gen-splitk; warp-GEMV excluded).**

## D-ARCH-08 — production regression (combined candidate)

Combined candidate (P-ARCH-04 MMA default + async overlap + `-ub 2048`
prefill; default `gen-splitk-fp32` decode; warp-GEMV excluded) passes the
assembled regression envelope: fresh 2k/ub2048 prefill 2209.86 tok/s (exit 0,
zero CUDA-error lines), decode c=1/c=8 complete, GPU stable, plus the
P-ARCH-15 matrix, P-ARCH-13/15 parity 16/16, compute-sanitizer 0 errors,
D-ARCH-04 graph A/B, and repeated clean process restarts. Production defaults
remain unchanged (candidate is experimental/opt-in).

Did this make the data presented to the kernel match the control? **The
combined candidate preserves the proven fragment representation and qualified
overlap/ubatch/graph settings; it passes regression.** Evidence:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/D-ARCH-08/2026-08-29/combined-001/`
and the per-gate evidence retained in P-ARCH-13..16 / D-ARCH-01..07.

**DECODE PROGRAM REACHED ITS CLEAN STOPPING POINT.** Remaining W2 decode gap
requires a dedicated Blackwell batched-GEMV kernel (beyond bounded
experiments). See `docs/escha-architecture/HANDOFF.md`.

## P-ARCH-17 — original Escha W2 artifact control

**CLOSED — Case B, model delta is material but partial.** Under the identical
Bee binary, RTX 5090, raw 2,048-ID input, `-b/-ub 2048`, and graphs-on prefill
boundary, original Escha W2 records a median **844.617 ms / 2424.77 tok/s**;
the current LowGPU hybrid records **905.168 ms / 2262.56 tok/s**. Original is
60.551 ms (6.69%) faster, removing 25.70% of the contemporaneous residual to
the retained Escha-runtime reference. It does not close the remaining 175.063
ms residual.

This is not a pure F16-vocabulary control: the artifacts share architecture,
shape/type descriptors, and 400 projection topology (272 K2 / 128 K3), but
781 of 2,052 common payloads differ. FFN K2/K3 W2 payloads are byte-identical;
differences are the F16-vs-LowGPU vocab boundary plus 48-layer GDN Escha,
SSM, norm, and bias lineage. Paired complete ESCHA route captures show an
identical ordered 400-record measured schema, all `mma-fp16`; no silent
loader conversion/repack/fallback occurs. Therefore the gain is model-level,
not a changed Bee K2/K3 route.

**NEXT_GATE: P-ARCH-18 — hybrid model delta root cause.** Isolate the F16
vocabulary boundary from the 781 changed GDN/body payloads before resuming any
new-kernel effort. Evidence and full caveats:
`docs/escha-architecture/P-ARCH-17/manifest.md` and
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-17/2026-08-30/`.

## P-ARCH-18 — original LowGPU Qwen3.8-27B control in BeeLlama

**CLOSED — CASE D.** The original LowGPU artifact
(`beellama-release/models/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf`,
TheWegemann HF repo, 9,570,663,040 B, SHA-256
`ad85e40a28aafd907eeb6ff6b21786b897dd750b0918427f1243d6d84ebcc72`, 851
tensors, no escha/lowgpu sidecars) loads directly in the same
`build-cuda-parch10-async` binary and runs the matched 2k prefill at a
same-session median of **594.037 ms / 3447.60 tok/s** (n=8, two captures) —
faster than the same-session hybrid (880.076 ms / 2327.07 tok/s) and original
Escha W2 (848.847 ms / 2412.69 tok/s), and faster than the retained
Escha-runtime reference (669.554 ms / 3058.75 tok/s). The LowGPU model
executes the whole 64-layer body on stock CUDA quant kernels (dequant +
tensor-core/MMQ; all projections standard `MUL_MAT`), bypassing Bee's ported
escha code-GEMM path (LUT/dep decode, rin/rout rotation, MMA partial +
finalize) entirely; that operator-path difference is the first responsible
cause of the delta, not file size (LowGPU 9.57 GB is larger than the hybrid
8.62 GB). The vocab boundary is not exculpated by this gate (`-n 0` does not
execute the LM head; embedding is one gather at 2k tokens). No kernel, loader,
or model was changed. Next bounded experiment: selective standard-quant
substitution for W2 projections (e.g., FFN or K2 layers) with a matched
quality + prefill gate, before any new-kernel effort.

Evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-18/2026-08-30/`
(identity-001, control-001..007, aggregate.py/json, hermes-lanes) and
`docs/escha-architecture/P-ARCH-18/manifest.md`.

## P-ARCH-19 — Escha mixed accumulator policy (mixed vs forced-FP32)

**CLOSED — arithmetic policy is first-order at the reference boundary.** On the
original Escha runtime at the matched 2,048-token TTFT boundary, forcing FP32
accumulation for every prefill shard costs **553.502 ms (+88.79% wall; 1.888x
slower)**: `ESCHA_PREFILL_ACC=mixed` median **623.380 ms / 3285.31 tok/s** vs
`ESCHA_PREFILL_ACC=fp32` median **1176.882 ms / 1740.19 tok/s** (n=3 per arm,
all trials HTTP 200 with server-verified 2048 prompt tokens; identical model
directory, serve.sh, wheel, launch env, prompt token stream, and request; only
the accumulator env changed, and the mixed server was stopped before FP32
launched).

Isolation: `ESCHA_PREFILL_ACC` has one consumer (`_acc_mode_for(IC)`,
wheel-src `sglang_srt_layers_quantization_escha.py:227,232-241`) and one call
site — the per-call accumulator argument of `torch.ops.escha.escham_code_gemm`
(line 1042). `mixed` = fp16 MMA accumulate for `IC <= 6144` (the short-IC
population, including the 5120→17408 K2/K3 shapes) and fp32 above; forced
`fp32` = fp32 everywhere. The knob cannot alter dispatch, geometry, graph
capture, or workspace. The retained Escha reference (669.554 ms / 3058.75
tok/s, P-ARCH-05 lineage) was measured with the same mixed default; a
forced-FP32 Escha runtime is ~1.76-1.89x slower than that family, so the
reference's speed is largely arithmetic-policy-carried. Magnitude 553.5 ms
exceeds the entire remaining Bee residual family (P-ARCH-16 249.9 ms,
P-ARCH-17 175.1 ms, P-ARCH-18 75.5 ms vs LowGPU). Structural factors are not
exculpated (P-ARCH-21), but the parity target must first fix the arithmetic
contract (mirror mixed with a quality gate, or re-target fp32).

Evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-19/2026-08-30/`
(mixed-001, fp32-001, aggregate.py/json, summary.md, launch-metadata.txt) and
`docs/escha-architecture/P-ARCH-19/manifest.md`.

**NEXT_GATE: P-ARCH-20 — one-shape Bee FP16 accumulator specialization on
5120→17408 within quality bounds (opt-in; matched exactness + quality +
prefill gate), or explicitly re-target parity at fp32 arithmetic.**

## P-ARCH-20 — Bee one-shape FP16 accumulator prototype

**CLOSED — REJECTED.** Native Escha's accumulator policy was mapped through
both the Python wrapper and the shipped wheel SASS. `_acc_mode_for(IC)` passes
`acc_mode=1` for `IC <= 6144`; at the matched K2 `5120→17408` shape the native
FP16 specialization contains `64 × HMMA.16816.F16` and zero FP32 HMMA, while
the forced-FP32 twin contains `64 × HMMA.16816.F32` and zero FP16 HMMA. This is
true FP16 MMA accumulation and a shorter accumulator lifetime, not output-only
narrowing.

Bee received the smallest rollback-safe arithmetic-only prototype:
`escha_matmul_dense_tiled_mma<K,BM,BN,FP16_ACC>`, compiled only with
`ESCHA_MMA_FP16ACC_EXPERIMENT` and dispatched only for
`M=2048, IC=5120, OC=17408, K=2`. Packed W2 codes, scales, weight layout,
activation staging, shared-B path, tile geometry (`128×128`), split/finalize
path, output layout, and model artifact were unchanged. Production defaults
were not changed. The experimental symbol was independently SASS-verified as
`32 × HMMA.16816.F16 / 0 × HMMA.16816.F32`; its existing FP32 twin is
`0 × F16 / 32 × F32`. The full-library FP16 HMMA count increased by exactly
32 (`147424` vs `147392`) while the compiled FP32 count stayed `158392`, proving
the intended specialization was present.

Matched 2,048-token `llama-bench`, RTX 5090, `-b/-ub 2048`, F16 KV, FA on,
identical 8.619 GB artifact, n=3:

- FP32 control: `987.781 / 929.039 / 920.661 ms`; median **929.039 ms /
  2204.43 tok/s**.
- FP16 accumulator: `1397.949 / 1323.860 / 1339.906 ms`; median
  **1339.906 ms / 1528.47 tok/s**.
- Delta: **+410.867 ms / 44.22% slower** (throughput ratio `0.6934×`).

Decision: **REJECT** under the `<10%` pivot rule. The arithmetic-only port moved
strongly in the wrong direction, so Bee's current `128×128` shared-B kernel is
not accelerated by simply halving accumulator precision/register footprint.
This does not disprove native Escha's measured mixed-accumulator advantage; it
shows that advantage depends on surrounding kernel architecture (native
`128×64` ownership/schedule, dependency structure, or dequant/MMA
interleaving) that Bee's current path does not reproduce. Register/occupancy
attribution and quality/parity were not completed because the performance hard
gate already failed decisively; no correctness claim is made for the rejected
candidate.

Did this make the data presented to the kernel match the control? **The A/B
kept the input representation and all structural dataflow constant and changed
only the MMA accumulator arithmetic; SASS proves the requested arithmetic path
executed, but it regressed full prefill and is rejected.**

Evidence:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-20/2026-08-30/{control-001,fp16acc-001}/`;
builds `build-cuda-p20-{control,fp16acc}`; source rollback snapshot
`/home/sean/beellama-escha-pre-p20-20260830.{patch,status,head}`.

**NEXT_GATE: P-ARCH-21 — one-shape direct-fragment/shared-B-bypass prototype
for `M=2048, IC=5120, OC=17408, K=2`, retaining FP32 accumulation initially.**
