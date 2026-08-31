# BeeLlama ↔ Escha Prefill Parity Review

- **Review date:** 2026-08-30
- **Scope:** prefill only, through P-ARCH-17
- **Method:** source, manifests, retained logs/profiles, GGUF CPU parsing, and arithmetic reconciliation only. No model, benchmark, CUDA kernel, profiler, compiler, GPU container, or GPU allocation was run for this review.

## 1. Executive conclusion

P-ARCH-17 improves the best observed Bee control from the prior `919.410 ms` neighborhood to `844.617 ms` for 2048 prompt tokens, or `2424.77 tok/s`, while the retained Escha reference is `669.554 ms`, or `3058.75 tok/s`. The current best observed Bee control is therefore **79.27% of Escha**, with **175.063 ms still to recover**.

The evidence does **not** yet justify a full Escha-style K2 port. It does justify continued work on the short-input W2 path, but the project should first resolve two cheaper confounds:

1. **Accumulator policy is not matched.** The original runtime's mixed policy uses FP16 accumulation for `IC <= 6144` and FP32 above that threshold; Bee's retained MMA implementation accumulates in FP32. The same `5120→17408` shape has almost the same residual whether its codes are K2 (`+66.968 ms`) or K3 (`+67.573 ms`). This is evidence for a short-IC/shape/accumulation issue, not a uniquely K2 decode issue.
2. **P-ARCH-17 is not a clean vocabulary-only control.** The original GGUF loads directly and uses the same Bee W2 routes, but 781 of 2052 common tensor payloads differ from the hybrid, including GDN codes, sidecars, norms, and SSM tensors. Its tensor ordering also differs substantially. The observed `60.551 ms` full-wall improvement occurs outside the measured W2 route, but it cannot yet be assigned specifically to the embedding, LM head, body values, or allocation/order locality.

The highest-probability explanations for the remaining gap are:

| Rank | Explanation | Plausible recoverable time | Confidence |
| ---: | --- | ---: | --- |
| 1 | Unmatched short-IC accumulation/code-generation policy | Roughly `65–120 ms`; upper bound is not additive with kernel-structure savings | Medium-high |
| 2 | Bee's short-IC W2 inner loop: shared decoded-B materialization, barriers, register lifetime, and output traffic | Remainder of the `168.886 ms` K2 boundary after accumulator isolation | Medium |
| 3 | Non-W2/model-layout/runtime residual exposed by P-ARCH-17 | Up to `31.746 ms` in the original-artifact accounting; exact subcomponents unknown | Medium-low |
| 4 | Launch/topology and graph accounting gaps | Likely single-digit to low-tens of ms; not yet independently measured | Low-medium |

**Recommendation: A. NOT YET.** Run the clean model factorial and accumulator-policy controls first. If a large same-shape residual remains, build a **single-shape direct-fragment prototype** for `M=2048, IC=5120, OC=17408`; do not generalize the K2 family until that prototype proves both speed and exactness.

## 2. Evidence quality and current measured parity

The requested `docs/escha-architecture/PROTOCOL.md` is absent. P-ARCH-08 is also absent. These are evidence gaps, not negative results. The structured GBrain timeline is useful history but is stale in at least one place (it leaves P-ARCH-13 open although the retained manifest closes it), so manifests and raw retained artifacts take precedence.

### Matched headline numbers

| Configuration | 2048-token time | Throughput | Escha parity | Evidence quality |
| --- | ---: | ---: | ---: | --- |
| Escha retained reference | `669.554 ms` | `3058.75 tok/s` | `100%` | High; retained trace and matched-ID reference |
| Bee P-ARCH-15 historical best | `919.410 ms` | `2227.51 tok/s` by arithmetic | `72.82%` | Medium-high; valid per-prompt outputs, parent runner exited 1 on an invalid `-f` option |
| Bee P-ARCH-17 hybrid control | `905.168 ms` | `2262.56 tok/s` | `73.97%` | High |
| Bee P-ARCH-17 original GGUF control | `844.617 ms` | `2424.77 tok/s` | **`79.27%`** | High |

The historical prose value `2228.75 tok/s` does not exactly reconcile with `919.410 ms`: `2048 / 0.919410 = 2227.51 tok/s`. Conversely, `2228.75 tok/s` implies `919.231 ms`. This `0.179 ms` discrepancy does not affect prioritization, but future manifests should derive throughput from the stored median rather than copy both independently.

P-ARCH-17 used the same Bee binary, exact 2048-token ID file, `-b 2048 -ub 2048`, graphs enabled, two alternating-order captures, and four timed samples per cell. Median headline values are therefore the newest valid matched comparison.

## 3. P-ARCH-01 → P-ARCH-17 evidence map

| Gate | Question | Audited result | Performance implication | Still relevant? |
| --- | --- | --- | --- | --- |
| P-ARCH-01 | Do Bee and Escha agree on packing/fragment mapping? | Retained probes were bit-identical; Escha symbols/SASS establish an SM120 HMMA path. | Packing arithmetic was not the first observed cause. | Yes, as a prerequisite; it does not prove full-kernel equivalence. |
| P-ARCH-02 | Where did the original dispatch diverge? | Bee's original SM120 route vetoed the intended MMA path. A historical stall inference was stronger than the evidence. | Dispatch had to be corrected before deeper comparison. | Yes; later P-ARCH-03 disproved “MMA itself stalls.” |
| P-ARCH-03 | Can forced SM120 MMA execute and help? | `655.468→1243.72 tok/s`; no retained hang. | MMA enablement was necessary and large. | Closed. |
| P-ARCH-04 | Can SM120 MMA be the default? | Official default about `1230.03 tok/s`. | Established the baseline for architectural work. | Closed. |
| P-ARCH-05 | What is the matched end-to-end gap? | Bee `1646.329 ms` / `1243.98 tok/s`; Escha `669.554 ms` / `3058.75 tok/s`. | Bee needed `976.775 ms`; parity was 40.67%. | Historical anchor. |
| P-ARCH-06 | How much is W2? | Bee W2 `1318.314 ms`, Escha W2 `516.565 ms`; W2 delta `801.749 ms`, non-W2 delta `175.026 ms`. | W2 explained 82.08% of that gap. | Yes, but superseded quantitatively by P-ARCH-17. |
| P-ARCH-07 | Does ubatch/chunking explain the gap? | Bee issued 4× W2 launches at `ub=512`; `ub=2048` allegedly cut W2 to `1189.650 ms` and full time to `1430.285 ms`. Raw 1024/2048 logs are not retained. | Chunking mattered, but did not explain most of the gap; workspace rose about 117 MiB. | Directionally yes; evidence is manifest-only. |
| P-ARCH-08 | — | No directory or manifest exists. | No conclusion can be attributed to this gate. | Missing evidence. |
| P-ARCH-09 | Is the dominant Bee kernel structurally expensive? | Manifest reports representative Bee `3.812 ms` vs Escha `1.203 ms`, 136 vs 80 registers, and 960 barriers. The evidence directory is absent; retained rows-512 logs are not the stated rows-2048 cell. Components total `3.842 ms` versus a `4.049 ms` whole-op value, leaving `0.207 ms`. | Strong hypothesis generation, weak quantitative proof. | Yes, with downgraded confidence. |
| P-ARCH-10 | Does the existing async staging path matter? | Full Bee throughput rose roughly `1230→1761 tok/s`. However, the candidate used `-ub 512`; the claimed `3.812→0.651 ms` and “faster than Escha” comparison mixes 2048- and 512-row calls. | Async overlap materially helps; the cross-runtime per-call superiority claim is invalid. | Full-wall result yes; per-call comparison no. |
| P-ARCH-11 | Is async improvement real under paired controls? | At rows 512, K2-wide sync `292.268/256=1.142 ms` versus async `165.519/256=0.647 ms`. Aggregate async Bee: K2 `442.862 ms`, K3 `356.771 ms`; Escha: `185.725/330.840 ms`. | Validates async staging; leaves a large aggregate K2 residual. | Yes. |
| P-ARCH-12 | What differs between K2 and K3 in Bee? | Both use one generic MMA template; packed constants differ. The gate inferred a K2-specific codegen/geometry/fusion problem. | Focus shifted to K2. | Partly stale: same-shape P15 data and Escha accumulation policy undermine “K2-specific.” |
| P-ARCH-13 | Does literal K2 `128×64` geometry fix it? | Registers fell `128→85`, CTA count doubled, time worsened `442.713→465.683 ms`, full throughput fell 0.78%. | Register count alone was not the limiter; re-staging A/doubled CTAs outweighed it. | Closed only for that literal implementation. |
| P-ARCH-14 | Does naive one-slice finalize fusion fix it? | K2 `442.713→444.051 ms`, neutral/slightly worse. Added in-kernel transform/barrier work offset traffic removal. | Existing-kernel fusion was not independently valuable. | Closed only for that implementation. |
| P-ARCH-15 | What is best complete Bee with async + `ub=2048`? | Median `919.410 ms`; stored prose says `2228.75 tok/s`. Parent runner failed on `-f`, but individual prompt runs completed without CUDA errors. | Reduced the gap to about `249.856 ms`. | Yes; superseded as best result by P-ARCH-17. |
| P-ARCH-16 | Where was the remaining P15 gap? | Bee K2/K3/all-W2/non-W2 `350.756/300.377/651.132/267.788 ms`; Escha `185.725/330.840/516.565/152.989 ms`. Component deltas sum `249.367 ms`, `0.489 ms` short of wall delta. | K2 dominated aggregate attribution; K3 offset some gap. | Quantitatively useful, interpretation revised below. |
| P-ARCH-17 | Does the original Escha GGUF change Bee performance? | It loads directly, uses the identical 400 W2 routes, and reaches `844.617 ms`; measured W2 is slightly slower (`659.883` vs hybrid `657.205 ms`). The `60.551 ms` gain is outside measured W2. | Artifact/runtime non-W2 effects are material, but the control is not vocabulary-only. | Most important current control; requires a clean factorial. |

## 4. What P-ARCH-17 changes

### 4.1 Exact artifacts and runtime treatment

| Property | Original Escha GGUF | Bee-compatible original control | Current hybrid |
| --- | --- | --- | --- |
| Artifact | `Escha-Qwen3.8-27B-W2.gguf` | The same file; no conversion was required | `escha-w2-lowgpu-mono-parity.gguf` |
| Size | `12,691,575,008 B` | Identical | `8,619,127,360 B` |
| SHA-256 prefix retained | `0d326e…` | Identical | `e307…` |
| Architecture/layers/width/FFN | qwen35 / 64 / 5120 / 17408 | Identical | Identical |
| Tensor count | 2054 | Identical | 2058 |
| Tensor types | F16 899, F32 753, I16 402 | Identical | F16 899, F32 753, I16 402, I8 4 |
| Embedding/head | Two dense F16 `5120×248320` tensors | Dense F16 loader path | Six LowGPU code/scale/zero-point tensors |
| W2 dispatch | Named Escha tensor load, no runtime repack | Same Bee `ggml_escha_mul_mat` path | Same Bee path |
| W2 launch sequence | 400 calls: 272 K2 + 128 K3 | Identical | Identical |
| Graph topology | Dense embed and dense output ops | Same | LowGPU embedding/output ops; body topology otherwise corresponding |
| Workspace implication | Much larger dense vocab tensors | Same | Smaller packed vocabulary, different workspace/operator path |

The phrase “Bee-compatible Escha W2” should not imply a derivative file: P-ARCH-17 passed the original GGUF directly to Bee. Loader lookup is by tensor name. Escha W2 code tensors are created in their stored I16 form and consumed without a hidden conversion or repack.

### 4.2 Metadata, payload, layout, and conversion differences

The artifacts are architecturally compatible, but not “basically the same” at the byte level:

- Both expose 2052 common tensor descriptors, and all 2052 common names, shapes, and types match.
- Of those common payloads, only 1271 are byte-equal; **781 differ**.
- Differing payloads include 144 GDN Escha-code tensors, 144 biases, 96 route tensors, 48 input transforms, 157 norms, and 192 SSM tensors. Approximate differing bytes by family are 1.384 GB codes, 55.1 MB SSM, 3.15 MB bias, 2.67 MB norm, 1.57 MB route, and 0.59 MB input transforms.
- All FFN and full-attention W2 code payloads compare equal. The changed W2-like codes are in GDN projections, not the measured FFN W2 route family.
- Hybrid-only metadata is `qwen35.lowgpu.version`; five common descriptive/count scalars differ, while tokenizer and model hyperparameter arrays match.
- No common tensor occupies the same ordinal position. Across common tensors, mean absolute ordinal shift is `352.9` (range `-451` to `+1610`). Named loading preserves logical identity, but allocation order/cache/TLB locality is not controlled.

The conversion/build path in [`convert_escha_to_gguf.py`](../../convert_escha_to_gguf.py) folds `rin*s_in` and `rout*s_out` to FP16, writes packed codes as I16, preserves the shared LUT/dependencies, applies the residual-scale norm convention, maps GDN alpha/beta, and selects either dense-F16 or LowGPU vocabulary output. The hybrid assembly script is retained at `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/source/build_monolithic.py`; it merges the verified original-body lineage with the selected vocabulary representation. The existing `mono-dequant.gguf` is **not** a valid clean control: compared with the hybrid, 257 common payloads still differ (161 norms and 96 SSM tensors). A fresh dense-vocabulary artifact must be built from the current hybrid body lineage for the next control.

### 4.3 P-ARCH-17 route and timing findings

| Measured Bee W2 boundary | Hybrid | Original GGUF | Original − hybrid |
| --- | ---: | ---: | ---: |
| Rotate | `28.341 ms` | `28.510 ms` | `+0.169 ms` |
| Matmul | `588.238 ms` | `589.746 ms` | `+1.508 ms` |
| Epilogue | `40.625 ms` | `41.623 ms` | `+0.998 ms` |
| K2 total | `352.962 ms` | `354.611 ms` | `+1.649 ms` |
| K3 total | `304.244 ms` | `305.271 ms` | `+1.027 ms` |
| All W2 | `657.205 ms` | `659.883 ms` | `+2.676 ms` |
| Full prompt wall | `905.168 ms` | `844.617 ms` | **`-60.551 ms`** |

Therefore the original artifact did not make Bee's W2 kernels faster. Within the retained boundary it made them `2.676 ms` slower. The full-wall win is outside that boundary, approximately `63.227 ms` after accounting for the W2 movement. That derived value is diagnostic, not a directly timestamped component.

The manifest's statement that the LM head is not executed under `-n 0` is incorrect. The qwen35 graph builds the output projection, and the retained LowGPU diagnostic reports one vocabulary multiply (`tokens=1`, about `0.946 ms`) before its profiling path aborts. Escha's retained trace similarly contains a `1.593 ms` LM-head event. P-ARCH-17 therefore changes both input embedding and one-token output projection representation, although those two kernels alone are far too small to explain 60.551 ms.

## 5. Reconciled remaining timing gap

The only internally closed post-P-ARCH-17 accounting is to subtract the one graph-disabled Bee route profile from the graph-enabled wall median, matching the convention used by P-ARCH-16. This introduces a boundary mismatch, so the two additive top-level rows below are medium confidence rather than exact GPU decomposition.

| Additive component | Bee original control | Escha | Delta | % of `175.063 ms` gap | Confidence |
| --- | ---: | ---: | ---: | ---: | --- |
| K2 W2 | `354.611 ms` | `185.725 ms` | `+168.886 ms` | `96.47%` | Medium-high |
| K3 W2 | `305.271 ms` | `330.840 ms` | `-25.569 ms` | `-14.61%` | Medium-high |
| **All W2** | **`659.882 ms`** | **`516.565 ms`** | **`+143.317 ms`** | **`81.87%`** | Medium-high |
| **Residual outside measured W2** | **`184.735 ms`** | **`152.989 ms`** | **`+31.746 ms`** | **`18.13%`** | Medium-low |
| **Total** | **`844.617 ms`** | **`669.554 ms`** | **`+175.063 ms`** | **`100%`** | High |

The following rows are nested or unmatched and must **not** be added again:

| Subcomponent/boundary | Bee original | Escha | What can be concluded |
| --- | ---: | ---: | --- |
| W2 rotate | `28.510 ms` | Fused/unknown | Bee pays this inside all-W2; no like-for-like Escha boundary. |
| W2 matmul | `589.746 ms` | `516.565 ms` includes Escha fused W2 boundary | Boundaries are asymmetric. |
| W2 output transform/finalize | `41.623 ms` | Fused/unknown | Bee pays separate traffic; P14 tested one implementation only. |
| Attention/GDN | Unknown | `21.085 ms` | Bee event trace is missing. |
| Normalization/activation | Unknown | `12.606 ms` | Bee event trace is missing. |
| Copies/conversions | Unknown | `36.230 ms` | Bee event trace is missing. |
| LM head | Original-F16 Bee unknown; hybrid diagnostic `~0.946 ms` | `1.593 ms` | Too small and not captured under identical instrumentation. |
| Other classified GPU | Unknown | `11.691 ms` | No matched Bee sub-boundary. |
| Escha host/trace remainder | Unknown | `69.784 ms` | Includes time outside classified Escha events; not a Bee host-gap measurement. |
| Model-specific operations | Embedded in Bee's `184.735 ms` residual | Embedded in Escha residual | P17 proves materiality but not attribution. |
| Unclassified/graph/scheduler | Embedded | Embedded | Exact split is missing; forcing it would double count. |

The P-ARCH-16 arithmetic discrepancy (`0.489 ms`) and P-ARCH-09 discrepancy (`0.207 ms`) should remain explicit. Neither is large enough to change the architectural decision, but both show why future gates need one canonical timestamp boundary.

## 6. Bee versus Escha K2 architecture

### 6.1 Dominant Bee execution map

For `M=2048, IC=5120, OC=17408`, Bee's retained kernel uses `BM=128`, `BN=128`, 256 threads/eight warps, and one slice. The grid is `16×136 = 2176` CTAs. Its conceptual execution is:

```text
packed I16 code + per-block scale/zero metadata (global)
  → load packed payload and scale state
  → double-buffer A tiles with cp.async (shared)
  → decode eight B values per participating thread
  → materialize decoded B[128][16] (shared)
  → CTA publication barrier
  → ldmatrix A/B into warp fragments
  → HMMA.16816 with FP32 tile accumulators (registers)
  → write FP32 partial buffer
  → separate finalize: read partial, transform, write FP32 destination
```

Per reduction tile and CTA, A staging and decoded-B materialization are each about 4096 bytes. There are 320 reduction tiles. Logical traffic across 2176 CTAs is about 2.85 GB of A shared writes and 2.85 GB of decoded-B shared writes, plus their shared reads; these are logical shared-memory movements, not claims about DRAM traffic. Packed K2 code traffic is about 512 bytes per CTA/tile, `~340 MiB` logical across the launch, against a physical packed tensor around 22.3 MB. The FP32 partial buffer is about 142.6 MB, then finalize reads it and writes another roughly 142.6 MB destination.

Ownership and synchronization:

| Stage | Bee owner | Memory/representation | Synchronization/lifetime |
| --- | --- | --- | --- |
| Packed and scale load | CTA threads | Global packed code and metadata | Coalescing depends on packed lane mapping. |
| A stage | CTA threads | Double-buffered shared A | Async producer/consumer barriers per K tile. |
| K2 decode | Threads, 8 decoded values/thread | Registers from packed codes/scales | Decode temporaries overlap fragment/accumulator lifetime. |
| B construction | CTA threads | Shared `B[128][16]` | CTA-wide publication barrier. |
| Fragment load | Warps | `ldmatrix` shared→register fragments | Warp-level fragment ownership. |
| MMA | Eight warps | FP32 register tile | Long-lived accumulators; retained reports range around 128–136 registers in prior variants. |
| Partial output | CTA/warp stores | Global FP32 partial buffer | One store pass. |
| Finalize | Separate launch | Reload partial; rotate/transform | One read plus one destination store; extra launch. |

### 6.2 Escha execution map and proof limits

The shipped Python wrapper describes direct decode into MMA B fragments, and the retained binary/SASS establishes SM120 HMMA symbols for K2/K3 shapes such as `<1,K,128,64,2,true,true>`. The native CUDA source is not retained. Consequently, thread ownership, exact scale access, shared-memory bytes, barrier count, and output-fusion implementation cannot be treated as source-proven.

The most defensible conceptual map is:

```text
packed code + scale metadata
  → shape/accumulation-policy dispatch
  → fragment-oriented decode (wrapper claim; binary-compatible inference)
  → HMMA.16816
  → fused or lower-traffic output handling (trace/symbol inference)
```

The **first proven dispatch-level divergence** is accumulator policy: the retained Escha wrapper chooses FP16 accumulation for `IC <= 6144` and FP32 otherwise, while Bee retains FP32 accumulation. The **first credible inner-loop divergence** is Bee's decoded-B shared materialization/publication/reload versus Escha's reported fragment-oriented decode, but that requires a targeted experiment because original native source is absent.

This distinction matters. The prior root-cause model jumped directly to the visually dramatic shared-B path even though an earlier, measurable policy difference was not isolated.

## 7. K2 versus K3 as an internal control

### 7.1 Family totals are misleading

| Family/shape | Bee P15 | Escha | Delta |
| --- | ---: | ---: | ---: |
| K2 `5120→1024` | `8.205 ms` | `4.270 ms` | `+3.935 ms` |
| K2 `5120→6144` | `44.749 ms` | `23.220 ms` | `+21.529 ms` |
| K2 `5120→10240` | `69.364 ms` | `36.338 ms` | `+33.026 ms` |
| K2 `5120→12288` | `27.681 ms` | `14.896 ms` | `+12.785 ms` |
| K2 `5120→17408` | `144.672 ms` | `77.704 ms` | **`+66.968 ms`** |
| K2 `6144→5120` | `56.085 ms` | `29.297 ms` | `+26.788 ms` |
| K3 `5120→17408` | `146.603 ms` | `79.030 ms` | **`+67.573 ms`** |
| K3 `17408→5120` | `153.774 ms` | `251.810 ms` | **`-98.036 ms`** |

For the identical `5120→17408` geometry, K2 and K3 have nearly identical residuals. K3's aggregate appears close only because Bee is `98.036 ms` faster on the reverse, long-IC projection. That reverse shape also crosses Escha's accumulator-policy threshold and changes operand orientation and CTA geometry.

### 7.2 Revised inference

What differs in the residual-heavy cases is not solely “K2 versus K3.” It is primarily:

- short input width (`IC=5120/6144`) versus long input width (`IC=17408`);
- Escha FP16 versus FP32 accumulation policy;
- output width and CTA aspect ratio;
- code/scale work per output tile;
- register lifetime and occupancy under those shapes.

K2 remains the dominant **aggregate budget** because most K2 calls are short-IC projections. It is not yet established as the dominant **quant-format defect**. A clean accumulator-policy experiment has higher information value than immediate direct-fragment implementation.

## 8. Rejected hypotheses: what was actually disproved

### P-ARCH-13 — K2 128×64 geometry

The tested change halved `BN`, doubled CTA count, and re-staged the same A work across twice as many CTAs. Registers improved from about 128 to 85, yet K2 worsened `22.970 ms` and full throughput fell 0.78%.

It disproves: “halve Bee BN to 64 while leaving the rest of its ownership/staging structure intact.”

It does not disprove:

- two 64-wide warp bands within the same 128-wide CTA;
- a 128×64 Escha template whose grid/ownership is not Bee's doubled-CTA construction;
- geometry changes coupled to lower shared memory or shorter accumulator lifetime;
- specialization for only the dominant short-IC shape.

### P-ARCH-14 — single-slice finalize fusion

The implementation changed K2 `442.713→444.051 ms`. It removed a launch and global partial-buffer pass, but introduced transform work, staging, and barriers in an already register/shared-memory-heavy kernel.

It disproves: “move the current finalize implementation into the current one-slice Bee kernel and expect an independent win.”

It does not disprove:

- output fusion after a direct-fragment change shortens register lifetime;
- warp-owned output transformation without CTA-wide staging;
- fusion enabled by a different tile or accumulator layout;
- eliminating the partial buffer in a kernel designed around direct destination ownership.

### Other conclusions that must be weakened

- P-ARCH-10 proved async overlap's full-stack value, not Bee per-call superiority to Escha; the compared row counts differ.
- P-ARCH-12 did not prove K2-specific code generation; same-shape K2/K3 timings contradict that strong form.
- P-ARCH-17 did not isolate “the original W2 representation”; W2 codes for the measured FFN family were already equal, while many non-W2 payloads and ordering changed.
- The Escha direct-fragment description is credible but not source-audited because native CUDA source is absent.

## 9. Overlooked and bounded opportunities

| Rank | Opportunity | Expected speedup | Complexity | Correctness risk | Evidence strength | Independent test? |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Match/force Escha accumulation policy, then prototype Bee FP16 accumulation on one short-IC shape | High | Low for reference control; medium for Bee | Medium-high | High | Yes |
| 2 | Fresh current-body dense-vocabulary artifact with identical tensor lineage/order where possible | Medium information; possibly medium speed | Low-medium | Low | High | Yes |
| 3 | Same-CTA decoded-B bypass/direct fragment for one shape | High if structural theory is right | High | High | Medium | Yes |
| 4 | Shorten K2/K3 decode and scale temporary lifetimes without changing math | Medium | Medium | Low-medium | Medium | Yes, compiler counters + timing |
| 5 | Reduce shared-B footprint or use per-warp slices while preserving CTA count | Medium | Medium-high | Medium | Medium | Yes |
| 6 | Remove/reduce partial-output traffic only after ownership changes | Low alone, medium in combination | Medium | Medium | P14 says weak alone | Conditional |
| 7 | K2/short-IC scale-load specialization and coalescing audit | Low-medium | Medium | Low-medium | Medium-low | Yes |
| 8 | Existing one-band-per-warp/shared-X concept, updated for the async/FP32 baseline | Medium | Medium-high | Medium | Historical design only | Yes |
| 9 | Dormant WMMA path | Low expected | Medium | Medium | Prior route evidence incomplete and historical attempt worse | Yes |
| 10 | cuBLAS/full-dequant diagnostic | Information only | Medium | Low | Historical gate blocked | Yes, but lower value |
| 11 | Alternate split-K/slice thresholds | Low for dominant `n_slices=1` | Low | Low | High | Yes |
| 12 | Graph/workspace reuse and launch synchronization trace | Low-medium | Low-medium | Low | P17 residual indicates need | Yes |

No retained evidence shows a dormant SM120 template that can simply be toggled to parity. The old WMMA and full-dequant branches are diagnostics, not leading solution paths. The most promising “existing” lever is the already-exposed accumulator policy, followed by template-level lifetime/shared-footprint changes.

## 10. Combinatorial effects

| Dependency | Why interaction is plausible | Prerequisite | Clean proof/disproof |
| --- | --- | --- | --- |
| Direct-fragment decode + reduced decode/register lifetime | Removing shared B may not improve occupancy if code/scale temporaries still overlap long-lived accumulators. | Measure generated resource usage for one unchanged-math prototype. | Same shape, same accumulator/output boundary; compare kernel time and registers separately. |
| New tile ownership + output fusion | P14's added barriers may disappear if the new ownership already holds destination fragments. | A faster one-shape inner kernel. | Add fusion only after prototype is stable; measure kernel + removed finalize together. |
| Reduced shared memory + higher occupancy | Shared footprint may be the limiting resource after registers fall. | Resource/occupancy model for the exact compiled variant. | Hold geometry/math fixed, reduce only shared-B scope, observe resident CTA and time. |
| FP16 accumulation + fragment decode | FP16 may reduce register/issue pressure, changing the benefit of shared-B removal. | Accuracy gate and isolated accumulation timing. | Four-cell factorial: FP32/FP16 × shared-B/direct-fragment for one shape. |
| Current-body artifact + graph/workspace reuse | Tensor ordering and vocabulary op type may alter allocation and graph scheduling. | Clean body-equal dense-vocab artifact. | Alternating-order wall tests plus operator timestamps; identical W2 route required. |
| 128×64 internal warp bands + shared-A reuse | P13 failed because it doubled CTAs and A staging; same-CTA bands can preserve reuse. | Explicit ownership design. | Same CTA grid and A bytes, change only warp-to-N mapping. |

Neutral independent results should not be blindly combined. Each pair above has a measurable prerequisite and a two- or four-cell control.

## 11. Parity budget in milliseconds

For 2048 tokens:

```text
Escha:             2048 / 3058.75 = 669.555 ms  (retained: 669.554 ms)
P15 Bee:           retained 919.410 ms
P17 hybrid Bee:    2048 / 2262.56 = 905.168 ms
P17 original Bee:  2048 / 2424.77 = 844.616 ms  (retained: 844.617 ms)

Need to recover from current best: 844.617 - 669.554 = 175.063 ms
```

| Budget source | Observed/current upper bound | Interpretation |
| --- | ---: | --- |
| Artifact switch already recovered versus P17 hybrid | `60.551 ms` | Real wall improvement, but not cleanly attributed; already included in current best. |
| Remaining K2 W2 boundary | `168.886 ms` | Largest current residual; accumulator and structure overlap inside it. |
| K3 W2 boundary | `-25.569 ms` | Bee is already faster in aggregate; do not “spend” this twice. |
| Remaining non-W2 boundary | `31.746 ms` | Maximum recoverable before Bee matches Escha outside W2; subcomponents unknown. |
| Total remaining attributable | `168.886 - 25.569 + 31.746 = 175.063 ms` | Closes exactly by construction. |

Parity is mathematically plausible: the positive K2 and non-W2 residuals total `200.632 ms`, enough to overcome the K3 credit and recover `175.063 ms`. It is not yet experimentally guaranteed. If accumulator matching recovers only `65–100 ms`, a structural W2 change must recover roughly another `50–80 ms`, with the remainder from non-W2/scheduling. If neither clean control nor a one-shape structural prototype yields that scale, a major contributor is still unclassified.

## 12. Ranked solution paths

1. **Resolve comparison validity first.** Build a current-body F16-vocabulary control and force matched Escha accumulator modes. These are cheap and can invalidate broad kernel work.
2. **Specialize short-IC accumulation/lifetimes.** If FP16 is both accurate and fast, introduce it only for the proven shapes; otherwise retain FP32 and isolate decode/materialization.
3. **Prototype one direct-fragment shape.** Preserve the exact codebook/scale math and launch geometry while eliminating the shared decoded-B round trip for `M=2048, 5120→17408`.
4. **Layer dependent output/tiling changes.** Only after the inner prototype wins, test same-CTA 64-wide warp bands and ownership-native output fusion independently.
5. **Generalize by shape family, not by K label.** The evidence partitions by short versus long IC more strongly than K2 versus K3.
6. **Re-profile the non-W2 residual after W2 convergence.** Do not optimize unclassified host/graph time before it has a matched timestamp boundary.

## 13. Ranked bounded experiments

### Experiment 1 — Current-body vocabulary factorial

#### QUESTION

How much of P-ARCH-17's `60.551 ms` improvement comes from dense versus LowGPU embedding/head representation when all body payloads and, as far as the writer permits, tensor order are held constant?

#### WHY

The original GGUF differs in 781 common payloads and complete tensor ordering; the existing dequant artifact still differs in 257 body payloads.

#### CHANGE

Generate one fresh dense-F16 vocabulary artifact from the exact current hybrid body lineage. Change only the vocabulary representation.

#### CONTROL

Same Bee binary, exact IDs, graph mode, batch/ubatch, body tensors, metadata, ordering policy, alternating execution order, and W2 route audit.

#### MEASUREMENT

Median full prompt time; input-embedding and one-token LM-head timestamps; W2 K2/K3 route totals; CPU tensor-payload manifest and output exactness.

#### EXPECTED OUTCOMES

**CASE A:** Most of 60 ms follows dense vocabulary. The LowGPU vocabulary path or its graph/workspace interaction is a primary non-W2 target.

**CASE B:** Only ~1–5 ms follows vocabulary. Original body values/order caused most of P17's gain.

**CASE C:** W2 routes move materially. Allocation/order or measurement interaction exists; do not call it a non-W2-only result.

#### EXIT CONDITION

Body equality is proven, route count/order is identical, and the wall delta is assigned with confidence intervals.

#### ESTIMATED VALUE

High.

#### RISK

Low.

### Experiment 2 — Escha accumulator-policy isolation

#### QUESTION

How much of Escha's short-IC advantage is caused by mixed FP16 accumulation rather than quant format or fragment decode?

#### WHY

All residual-heavy projections use `IC<=6144`, and identical K2/K3 `5120→17408` shapes have identical residuals. The wrapper exposes the policy directly.

#### CHANGE

Force Escha FP32 accumulation for the retained matched prompt; compare with its mixed policy.

#### CONTROL

Same original artifact, IDs, runtime, graph/batch settings, route order, and all kernel/template choices except accumulator mode.

#### MEASUREMENT

Per-family and per-shape W2 time, especially both `5120→17408` families and `17408→5120`; output error against the mixed reference.

#### EXPECTED OUTCOMES

**CASE A:** Short-IC time regresses toward Bee by ≥50 ms total. Accumulation is a leading cause and Bee should test a bounded FP16 path.

**CASE B:** Regression is 10–50 ms. Accumulation matters but cannot close the structural residual.

**CASE C:** Regression is <10 ms. Shared-B/codegen/output structure becomes the leading explanation.

#### EXIT CONDITION

A retained matched trace attributes the accumulator-only delta by shape.

#### ESTIMATED VALUE

High.

#### RISK

Low for the diagnostic.

### Experiment 3 — Bee FP16 accumulator, one short-IC shape

#### QUESTION

Can Bee recover Escha's short-IC advantage with accumulator specialization while meeting an explicit correctness budget?

#### WHY

This is the smallest implementation that follows Experiment 2 without changing staging, geometry, or decode.

#### CHANGE

Add an opt-in FP16-accumulation specialization only for `M=2048, IC=5120, OC=17408`; keep packed math and output boundary unchanged.

#### CONTROL

Same Bee artifact, kernel geometry, async staging, shared-B path, launch count, finalize, and prompt.

#### MEASUREMENT

Shape kernel time; full K2/K3 W2 time; register/shared resource report; exact/relative operator error and end-to-end output acceptance gate.

#### EXPECTED OUTCOMES

**CASE A:** ≥25% shape speedup with acceptable error. Generalize by short-IC shape.

**CASE B:** Speedup but unacceptable error. FP16 is diagnostic only; pursue structural FP32 work.

**CASE C:** <10% speedup. Accumulator policy does not explain the Bee implementation gap.

#### EXIT CONDITION

One opt-in shape has paired performance and correctness evidence; no default change.

#### ESTIMATED VALUE

High conditional on Experiment 2.

#### RISK

Medium-high.

### Experiment 4 — Shared-B materialization isolation

#### QUESTION

Does removing only the decoded-B shared publication/reload materially reduce the same-shape residual?

#### WHY

This is the central structural hypothesis, but it has not been isolated and the original native source is absent.

#### CHANGE

For one `M=2048, 5120→17408` kernel, construct MMA B fragments without the full CTA shared-B array. Preserve accumulator mode, tile/grid, codebook math, A staging, and output handling.

#### CONTROL

Current async Bee kernel and the exact same artifact/prompt; no fusion or tile change.

#### MEASUREMENT

Kernel time, barriers, registers, shared memory, occupancy estimate, and operator exactness.

#### EXPECTED OUTCOMES

**CASE A:** ≥20% shape win. Direct-fragment structure is validated.

**CASE B:** 5–20% win. Useful, but additional lifetime/output work is required.

**CASE C:** <5% or regression. The visually obvious materialization is not the primary bottleneck under fixed FP32 geometry.

#### EXIT CONDITION

The single changed boundary has paired timing and exactness evidence.

#### ESTIMATED VALUE

High.

#### RISK

High.

### Experiment 5 — Decode/scale register-lifetime specialization

#### QUESTION

Can short-IC resource pressure be reduced without changing arithmetic or fragment representation?

#### WHY

P13 showed that lowering registers through a topology change is insufficient; it did not test narrower lifetimes under a fixed grid.

#### CHANGE

Rescope or pipeline K2/K3 code/scale temporaries so they die before or between MMA issue groups; one template parameter only.

#### CONTROL

Same grid, shared-B representation, accumulator type, output path, and codebook operations.

#### MEASUREMENT

Generated register count, spills, static shared memory, occupancy estimate, and same-shape time.

#### EXPECTED OUTCOMES

**CASE A:** Resource drop and ≥10% win: retain and use as a prerequisite for fragment work.

**CASE B:** Resource drop without speed: issue/memory/barriers dominate.

**CASE C:** No resource drop: compiler live ranges require a more structural boundary.

#### EXIT CONDITION

One compiler/resource delta and paired timing result.

#### ESTIMATED VALUE

Medium-high.

#### RISK

Medium.

### Experiment 6 — Same-CTA 64-wide warp bands

#### QUESTION

Can Escha-like N ownership be tested without doubling CTAs or A staging as P13 did?

#### WHY

P13's geometry result is confounded by doubled grid work.

#### CHANGE

Keep a 128-wide CTA/output tile and assign two internal 64-wide warp bands; preserve total CTA grid and A tile loads.

#### CONTROL

Same accumulator, decode, shared footprint initially, finalize, and prompt.

#### MEASUREMENT

Same-shape time, A bytes staged per CTA/grid, registers, barriers, and output exactness.

#### EXPECTED OUTCOMES

**CASE A:** Win with unchanged A traffic: ownership, not literal BN, mattered.

**CASE B:** Neutral: geometry is secondary.

**CASE C:** Regression: cross-warp/fragment overhead outweighs ownership benefit.

#### EXIT CONDITION

CTA count and A traffic are demonstrably unchanged and performance is paired.

#### ESTIMATED VALUE

Medium.

#### RISK

Medium.

### Experiment 7 — Ownership-native output fusion

#### QUESTION

After a faster one-shape kernel exists, can its native destination ownership remove partial-buffer traffic profitably?

#### WHY

P14 tested fusion in the old ownership model and added barriers; a new fragment/warp layout may change the economics.

#### CHANGE

Fuse only the single-slice output transform into the winning prototype.

#### CONTROL

All inner-loop code, geometry, accumulator, and artifact remain fixed.

#### MEASUREMENT

Combined kernel plus finalize time, removed bytes/launches, registers, barriers, exactness.

#### EXPECTED OUTCOMES

**CASE A:** ≥5% combined win: retain for that shape.

**CASE B:** Neutral: traffic is hidden or resource cost offsets it.

**CASE C:** Regression: keep separate finalize.

#### EXIT CONDITION

Combined boundary is faster without changing inner-loop timing or correctness.

#### ESTIMATED VALUE

Medium, conditional.

#### RISK

Medium.

### Experiment 8 — Matched non-W2 timeline

#### QUESTION

After the artifact and W2 controls, where does the remaining non-W2 wall delta occur?

#### WHY

The current `31.746 ms` residual has no matched Bee component breakdown, and P17's 60 ms artifact effect is unassigned.

#### CHANGE

Add/fix CPU-collected event timing boundaries for embedding, LM head, attention/GDN, norms/activations, copies, graph launches, and host waits; do not change execution.

#### CONTROL

Exact models, IDs, graphs, batch/ubatch, and W2 route; instrumentation overhead measured with paired on/off runs.

#### MEASUREMENT

One additive matched timeline whose event sum reconciles with wall within 1% or an explicitly named remainder.

#### EXPECTED OUTCOMES

**CASE A:** One kernel family explains >15 ms: create a bounded optimization gate.

**CASE B:** Many small launches explain it: graph/scheduler work is justified.

**CASE C:** Host/remainder dominates: repair timing boundaries before GPU optimization.

#### EXIT CONDITION

At least 99% of both wall times is classified or the unmatched boundary is named and quantified.

#### ESTIMATED VALUE

Medium-high after W2 controls.

#### RISK

Low.

## 14. Direct-fragment kernel decision

### Decision: A. NOT YET

There are two higher-value bounded experiments first: the clean current-body vocabulary factorial and Escha accumulator-policy isolation. They can materially change both the estimated structural budget and the correctness contract.

If those controls leave at least ~50 ms attributable to the `5120→17408` shapes, advance to **B. prototype one shape**, not a full family. The prototype should copy or reproduce only:

- the verified packed-code/scale mapping;
- fragment-compatible lane ownership;
- the accumulator policy selected by the preceding accuracy gate;
- direct construction of B fragments or an equivalently narrow per-warp staging scope;
- output ownership only if measured separately after the inner kernel wins.

It should **not** blindly copy:

- wrapper-level launch heuristics whose native implementation is unavailable;
- Escha's entire K2/K3 dispatch family;
- mixed accumulation without a Bee correctness gate;
- fused output in the first prototype;
- a `128×64` spelling that doubles Bee's CTA grid;
- inferred shared-memory/barrier behavior not proven by retained native source.

## 15. Proposed next P-ARCH progression

| Gate | Bounded question | Reason | Exit condition | Expected information gain |
| --- | --- | --- | --- | --- |
| P-ARCH-18 | Does dense versus LowGPU vocabulary explain P17 when the current body is byte-identical? | P17 is a dirty artifact control. | Payload manifest, exact route, matched wall/operator timing. | Very high; isolates up to 60.551 ms of historical movement. |
| P-ARCH-19 | How much does Escha mixed accumulation buy by short-IC shape? | Same-shape K2/K3 residuals point here. | Mixed-vs-FP32 per-shape trace and error report. | Very high; determines whether the defect is arithmetic policy or structure. |
| P-ARCH-20 | Can Bee FP16 accumulation win on one `5120→17408` cell within quality bounds? | Cheapest implementation after P19. | Opt-in single-shape timing + exactness/quality gate. | High; may recover a large fraction without new decode architecture. |
| P-ARCH-21 | Under matched accumulation, does bypassing shared decoded B win on one shape? | Direct test of the structural theory. | Fixed-grid/fixed-output paired kernel, resource, and exactness data. | Very high; go/no-go for a direct-fragment family. |
| P-ARCH-22 | Do same-CTA 64-wide bands or lifetime reduction add value to the winning prototype? | Revisits P13 without doubled A staging. | One-variable paired results; no combinatorial sweep. | Medium-high. |
| P-ARCH-23 | Does ownership-native output fusion become positive after the structural change? | P14 was implementation-specific. | Combined-boundary win or closure. | Medium. |
| P-ARCH-24 | What is the fresh matched full-stack residual? | Current subcomponent boundaries are asymmetric. | Additive timeline reconciled within 1%. | High; selects W2 generalization versus non-W2 work. |
| P-ARCH-25 | Can the proven short-IC solution generalize without regressing long-IC K3? | Families should follow shape evidence, not K labels. | All retained shape families pass speed and correctness gates. | High. |
| P-ARCH-26 | Does Bee meet matched prefill parity? | Final objective. | Exact artifact/IDs/settings; alternating runs; median Bee ≥95% initially, then statistically indistinguishable or ≥100% target; full evidence retained. | Definitive. |

P-ARCH-18 should also restore the missing exact operator-byte oracle previously left open and record peak workspace. That closes the practical hole left by the absent P-ARCH-08 rather than inventing a retroactive gate result.

## 16. Top three next experiments

1. **P-ARCH-18: current-body dense-vocabulary factorial.** It is cheap, fixes the most important P17 confound, and tells us whether the 60.551 ms artifact effect is actionable.
2. **P-ARCH-19: force Escha FP32 accumulation.** It directly tests the strongest new alternative explanation with no Bee kernel redesign.
3. **P-ARCH-20: one-shape Bee FP16 accumulator specialization, conditional on P19.** It is the smallest plausible recovery path. If P19 is small or P20 fails quality/speed, replace it with P-ARCH-21's fixed-geometry shared-B bypass.

## 17. Evidence provenance and unresolved limitations

Primary evidence reviewed includes every retained P-ARCH manifest from 01 through 17 (with P-ARCH-08 absent), P-ARCH-17 raw stdout/route captures and runner scripts, earlier P15/P16 route/timing evidence, the architecture ledger and handoff, the prefill experiment ledger, CPU-parsed GGUF metadata/tensor payloads, Bee loader/graph/conversion code, Bee CUDA W2 source, the shipped Escha Python wrapper, retained Escha trace/SASS fragments, and GBrain's **“BeeLlama Escha architecture-diff ledger — handoff”** and **“P-ARCH-17 original Escha W2 inside BeeLlama control”** pages.

Key in-repository anchors are the [architecture ledger](../escha-architecture-diff-ledger.md), [handoff](HANDOFF.md), [prefill experiment ledger](../escha-prefill-experiment-ledger.md), [P-ARCH-17 manifest](P-ARCH-17/manifest.md), [P-ARCH-16 manifest](P-ARCH-16/manifest.md), [P-ARCH-15 manifest](P-ARCH-15/manifest.md), [Bee W2 kernel](../../ggml/src/ggml-cuda/escha-moe.cu#L958), and [qwen35 embedding/head graph](../../src/models/qwen35.cpp#L227). The original-runtime wrapper is retained outside this repository at `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/runtime/wheel-src/sglang_srt_layers_quantization_escha.py`; its direct-fragment description is at lines 201–217 and its accumulator policy at lines 227–240.

Six Hermes/Nous DeepSeek V4 Flash advisory lanes were attempted for historical evidence, P17 artifact controls, Bee source, original-runtime evidence, timing, and adversarial hypotheses. The historical audit returned useful findings; the other portal workers launched but remained silent past bounded review windows and were terminated. The primary review independently checked every decisive claim against retained local evidence rather than treating worker output as authority.

Limitations that remain material:

- Original Escha native CUDA source is absent; direct-fragment details below the wrapper/SASS boundary are inferred.
- P-ARCH-07 raw high-ubatch logs, the P-ARCH-09 evidence directory, P-ARCH-08, and `PROTOCOL.md` are absent.
- Bee subcomponent timing after P17 does not yet provide a fully additive attention/norm/copy/host breakdown.
- Profile and full-wall cells use different graph instrumentation boundaries.
- No new performance or correctness experiment was executed in this review by design.

The next decision should be evidence-driven: if accumulator matching removes most of the short-IC delta, specialize arithmetic conservatively; if it does not, the one-shape shared-B/direct-fragment experiment becomes justified. A generalized new K2 kernel before those controls would spend the most engineering effort while the comparison still contains cheaper unresolved variables.
