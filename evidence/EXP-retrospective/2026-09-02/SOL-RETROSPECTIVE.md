# ESCHA-W2 PREFILL optimization series — Sol retrospective

Date: 2026-09-02  
Branch: `escha-w2-prefill`  
Scope: analysis only; no source change, build, or benchmark was performed.

## Executive conclusion

The series found one large runtime win and one useful arithmetic win, then reached a structural wall:

- EXP-01's asynchronous A staging removed a real serialization and improved the canonical full-Escha artifact from about 1450 to 2302 tok/s.
- EXP-04 Stage 2's correct mixed-accumulator policy added about 9.3%, producing the current roughly 2300-2327 tok/s / 880-890 ms control depending on the matched session.
- The remaining roughly 243-270 ms gap to the same-runtime IQ3_XXS/LowGPU path is overwhelmingly in the FFN projection path, not graphs or the separate input/output transforms. BASE-01 attributes 213.8 ms, 94.6% of the positive gap, to `ffn_up + ffn_gate + ffn_down`; the per-call `ffn_down` deficit remains about 1.7-2.1x from M=128 through M=4096.
- At Bee's BM128xBN128, 256-thread CTA, shared-B is not incidental overhead. It is the broadcast structure that decodes a B tile once and reuses it across four row-warp groups. EXP-02, EXP-07, and EXP-09 removed that round trip in different ways and paid for it with duplicated codebook/dependency work, longer live ranges, 124-176 registers/thread, and—in the decisive cases—loss of the second resident CTA/SM.
- P-ARCH-23I is the cleanest causal evidence in the entire series: on the same BeeLlama runtime, replacing/reconstructing the body into standard GGML tensors reaches 619-621 ms / about 3300 tok/s, within about 1.2% of the LowGPU reference. Runtime-only work has not produced comparable evidence. If preserving the exact packed full-Escha execution path is not a hard product requirement, parity is already demonstrated and the rational next step is artifact certification, not another BM128xBN128 micro-optimization.

The breakthrough question is therefore a product/architecture decision: either promote a standard-quantized derived Escha artifact, transcode the packed artifact once at load time into an MMA-ready persistent representation, or fund a genuinely new packed kernel/representation. More local surgery on the current mainloop is very unlikely to recover 20-30%.

## Experiment-by-experiment findings

Numbers below retain their original benchmark boundaries. Cross-session values are used as evidence of direction and mechanism, not silently treated as paired measurements.

| Experiment | Hypothesis | Result | Why it worked or failed | Durable lesson |
| --- | --- | --- | --- | --- |
| **P-ARCH-19** | The official dense Escha runtime's mixed policy—FP16 MMA accumulation for IC <= 6144, FP32 above—is first-order. | Official same-artifact/same-request mixed: **623.380 ms / 3285.31 tok/s**; forced FP32: **1176.882 ms / 1740.19 tok/s**. Mixed is 1.888x faster. | This is a valid isolation at the official operator boundary: the accumulator argument was the only knob. It proves the official fused schedule depends strongly on the shorter FP16 accumulator path. It does **not** identify whether register lifetime, instruction count, fusion, or their coupling carries the gain. | Mixed accumulation is necessary reference behavior, but an official-runtime delta is not portable proof for Bee's different shared-B kernel. Copy the arithmetic policy only with a correct store and resource/SASS proof. |
| **P-ARCH-20** | Port FP16 accumulation to Bee for only K2 5120->17408/M2048 while holding layout and staging fixed. | Reported FP16 **1339.906 ms / 1528.47 tok/s** vs FP32 **929.039 ms / 2204.43 tok/s**, nominally -44.22%; rejected. Later contamination audit invalidated the conclusion. | The FP16 fragment store read beyond `tile_ah` and dropped `.y` lanes; parity never produced valid correctness evidence. The experiment measured a faulty output path, not the cost of FP16 accumulation. EXP-04 Stage 2 fixed the seam and showed the correct policy is beneficial. | Do not use P-ARCH-20 as negative evidence against FP16 accumulation or as proof of a structural mismatch. Its durable lesson is procedural: fragment-layout correctness must precede performance attribution. |
| **P-ARCH-21A** | Replace all 192 FFN gate/up/down Escha sidecars with compatible standard quant weights; retain attention/GDN/vocab. | **727.724 ms / 2814.31 tok/s**, 8.483 GB, coherent output, decode no regression. | It bypassed the packed Escha decode/rotate/partial/finalize path exactly where later BASE-01 showed 94.6% of the gap lives, and used mature stock quantized matmul kernels. | The FFN operator/representation path—not generic Qwen graph scheduling—is the dominant gap. Artifact substitution is not merely a ceiling; it is causal localization. |
| **P-ARCH-21B** | Add 48 standard `attn_gate` projections from the donor LowGPU GGUF. | **697.046 ms / 2938.15 tok/s**, but garbage generation; rejected as built. | The donor tensor with the same apparent role was not the checkpoint's `in_proj_z`: correlation was about 0.04, versus about 0.835 for the valid FFN donor. F16 reconstruction from the checkpoint was coherent, proving the loader/standard path and orientation were sound. Bias was not causal. | Tensor names and shapes do not establish semantic identity. Correlate or reconstruct each family before substitution; never infer compatibility from a neighboring export. |
| **P-ARCH-23** | Reconstruct the true GDN gate from Escha sidecars, quantize it to Q2_K, and keep the standard FFN body. | **697.032 ms / 2938.27 tok/s**, 8.599 GB, quality pass and no decode regression. | It retained 21B's fast standard operator path while restoring the correct checkpoint projection. Q2_K was sufficient for the size and observed quality gate; Q4_K was slower and Q6_K tied despite being larger. | Reconstruction plus standard low-bit storage is a viable deployment architecture. More bits do not automatically produce more speed or measured quality. |
| **P-ARCH-23G** | Replace the packed LowGPU embedding representation with standard Q4_K/Q6_K `get_rows` storage. | Q4: **665.188 ms / 3078.96 tok/s**, 8.808 GB, quality/parity no regression; about 32 ms faster than P-ARCH-23. | Standard `get_rows` captured essentially all of the over-cap F16 embedding diagnostic's gain without its memory cost. This was representation/dispatch overhead, not LM-head GEMM, because the prefill run used `-n 0`. | About 32-35 ms of the old artifact trajectory was embedding representation. It is real but far too small to explain the canonical full-Escha runtime gap. |
| **P-ARCH-23I** | Reconstruct/quantize the incompatible linear-attention QKV/SSM projections and put all attention projections, FFN, gate, and embedding on standard GGML paths. | **619-621 ms / about 3300 tok/s**, 9.345 GB; coherent output and improved lightweight parity. | It removed packed Escha execution from the body instead of trying to make its per-tile decoder emulate a mature standard quant GEMM. The remaining Escha tensors are codec tables, not hot body projections. | This is the only demonstrated parity path. It establishes that the Qwen graph and Bee runtime can hit the target; the missing performance is in packed representation/operator execution. It still needs milestone-quality certification before being called the final model. |
| **ARCH-01** | Determine whether Bee accidentally runs a MoE-derived execution architecture for a dense model. | **DENSE-CORRECT / PERF-ARCH-MISMATCH.** Artifact, loader, op source count, dispatch, and 800/800 route records prove dense execution with no expert routing. | The filename `escha-moe.cu` is historical. The real mismatch is microarchitecture: Bee uses separate rotate -> shared-B GEMM -> finalize; the official kernel uses a coupled two-band/warp-collective/fused design and mixed accumulation. | Do not spend time removing imaginary MoE routing. Compare actual dense dataflow, resource lifetime, representation, and launch structure. Also retain the later correction: P-ARCH-20's negative was contaminated. |
| **EXP-01** | Make SM120 double-buffered `cp.async` A staging the default. | Sync **1412.5 ms / 1449.9 tok/s**; async **889.5 ms / 2302.4 tok/s**, **+58.8%**; consolidated default reproduced it; quality milestone passed. | It overlapped a genuine activation-staging dependency without duplicating work or increasing the expensive B/accumulator live state. | This was the right class of optimization: hide a proven pipeline stall while preserving ownership and occupancy. It is banked and should not be rediscovered. |
| **EXP-02** | Decode packed K2 weights directly into warp-owned MMA fragments, removing shared-B store/barrier/reload. | **916.0 ms / 2235.7 tok/s** vs **880.1 ms / 2327.0 tok/s**, -3.9%; 176 regs/thread. | Each M warp repeated codebook/dependency decode and retained fragment/address state. The saved STS/LDSM/barrier cost was smaller than duplicated ALU and the occupancy loss to one CTA/SM. | “Remove shared memory” is not an optimization when shared memory is the CTA's decode-broadcast mechanism. A retry must prove cooperative decode and <= control-class registers before timing. |
| **EXP-03** | Keep shared-B but change K2 tile 128x128 -> 256x64 to halve B decode per CTA and reuse it over more rows. | **892.7 ms / 2294.1 tok/s** vs **889.0 ms / 2303.7 tok/s**, -0.42%; neutral. | B decode amortization alone did not move the wall. The larger M tile increases A footprint/row work while the smaller N tile changes scheduling; these costs offset the nominal decode reduction. The packed GEMM body remained about 2 ms for the target family. | Standalone CTA aspect changes are not the missing breakthrough. Tile shape must be derived as part of a different pipeline/ownership design, not swept as an independent knob. |
| **EXP-04** | Attribute the fused bound, adopt correct mixed accumulation, then extend FP16 safely to long-IC down projection. | Stage 1: rotate **4.6%**, matmul **88.6%**, finalize **6.7%**. Stage 2: correct mixed accumulation, paired **+9.31%**, 97/128 regs, promoted. Stage 3: four-slice bounded FP16 for 17408->5120, net **+2.76%**, rejected. | Stage 2 shortened the accumulator path for every safe IC<=6144 family without changing topology; family matmuls improved 17.6-26%. Stage 3's FP16 gain was real (+8.7% versus its FP32 twin), but four-way split-K and partial/finalize traffic imposed about 4%, leaving less than the 5% wall gate. | The correct arithmetic policy is a useful smaller positive, not the breakthrough. Launch fusion has only an 11.3% theoretical bound and less recoverable. Do not buy FP16 depth safety with enough split-K traffic to erase the gain. |
| **EXP-05** | Audit and, if coverage justified it, mirror the official `escham_code_gemm` two-band mainloop. | Official short-IC families were 0.78-0.90x Bee, but covered only **10.96%** of aggregate Bee matmul savings; no-go. Official default down projection was **1.78x slower** than Bee and improved about 1.8x under its BM64-family selection. | The official wheel is not uniformly faster at direct-op level: aggregate official direct-call time was 963 ms vs Bee Stage 2's 831 ms because down projection was poor. Its server wall win therefore conflates short-IC mainloop, fused epilogue, graph/overlap, and runtime boundary. The measurable short-IC budget was below the 15% implementation bar. | Treat the official kernel as an architectural clue, not a drop-in target. Do not clone a visible two-band property while retaining Bee's payload organization and expect the official result. |
| **EXP-06** | Transfer the official down-projection BM64 win to Bee by changing only K3 17408->5120 from BM128 to BM64, retaining BN128 and the rest of Bee's kernel. | Target family **3.30810 ms** vs control **2.25125 ms**, +46.95% slower; full smoke -7.40%; rejected and reverted. | The official “BM64” selection was a coupled different specialization (`BM64/BN32/BK3`, 65 regs, 35.8 KiB), whereas Bee changed only M coverage and retained BN128/WN2/decode/finalize/split-K. Bee doubled row CTAs and reduced reuse without importing the official register and pipeline structure. | Never port an environment knob by name. Re-derive the complete selected kernel geometry and ownership. For Bee, BM-only reduction is closed. |
| **EXP-07** | Remove shared-B via register-decoded B while avoiding EXP-02's duplication: first all-row ownership, then disjoint 8-column fragments/two passes. | v1: 124/154 regs; v2: 125/141 regs, FP32 HMMA fell 32->24 and barriers did not improve. Both failed pre-timing gates and were reverted. | Full 128-row or two-pass ownership expands A/address lifetime. The seemingly natural MT4/NTT2 alternative cannot decode once because two row warps need the same B and registers cannot cross warps. The valid disjoint-column design replayed A and still could not preserve full-tile work within the resource ceiling. | There is no register-only ownership rearrangement at this CTA geometry that both decodes B once and cheaply broadcasts it. Cross-warp staging is required—which is precisely the role shared-B already serves. |
| **EXP-08** | Fuse Hadamard-128, normalization, and `rout` into the single-slice GEMM epilogue. | All pre-timing gates passed (96/128 regs, no spills, exact output, decode pass), but matched campaign fell **2319.22 -> 1895.11 tok/s**, -18.29%; 0/9 wins. | The warp-pair design processed eight batches with two named barriers per pair and only two owner warps. It replaced a highly parallel full-CTA finalize kernel with serialized transpose/shuffle/exchange work. Eliminating the partial round trip did not compensate. | Fusion is beneficial only if it preserves or increases parallelism. Warp-pair finalize is closed; a future epilogue must be full-CTA parallel, and even its measured upper bound cannot deliver parity. |
| **EXP-09** | Rewrite the mainloop around two 128x64 bands with direct warp-local B construction, bounded 32x64 per-warp output, two barriers/tile, and no shared-B round trip. | Structural goals were met, but FP16 K2/K3 used **145/150 regs**, FP32 **176**, versus control 97/128; rejected before timing. Decode ALU grew about 2-2.7x and LDS 4x. | Four row warps per band each recomputed the payload-to-weight index/codebook chain. Streaming one B fragment bounded fragment storage but did not bound decode temporaries or eliminate 4x decode redundancy. This is not the full official microarchitecture: it retained Bee's payload/A/partial/finalize contracts while copying only warp-local B and two-band ownership. | The official schedule cannot be decomposed into “remove shared-B + use two bands.” A viable rewrite needs a different staged representation or warp-collective publication that shares decode without reviving long register lifetimes. That is a new architecture, not an EXP-09 revision. |

## Recurring failure pattern

### 1. Local overhead was mistaken for globally removable work

EXP-02/07/09 treated shared-B stores, reloads, and a barrier as pure overhead. At BM128xBN128 they are the cost of decoding each 16x128 B tile **once per CTA** and broadcasting it to four independent row groups. Registers are warp-private; removing shared-B forces either:

1. four copies of the same decode across row warps;
2. one warp to own too many rows/columns and carry large A/accumulator state;
3. replay of A and incomplete HMMA coverage; or
4. another cross-warp publication mechanism.

The first three were measured and lost. The fourth is shared/collective staging under a different name and must be evaluated as a whole pipeline.

EXP-08 made the analogous mistake at the epilogue: a separate launch and partial round trip looked removable, but the separate kernel provided full-CTA parallelism. The fused version saved traffic while serializing the actual Hadamard work.

### 2. The experiments copied one visible reference feature without its coupled architecture

- EXP-06 copied “BM64” but not the official specialization's BN/BK/register/shared-memory structure.
- EXP-09 copied two 64-column bands and warp-local B but retained Bee's payload representation, A pipeline, partial buffer, and separate finalize. The official symbol has 45 KiB shared memory, 80 registers on the hot FP16 family, 136 LDS operations, warp collectives, 64 HMMA occurrences, and an in-kernel Hadamard. EXP-09's 9.7 KiB/145-176-register result is evidence that it did **not** reproduce the property that makes the official design economical.
- EXP-05 showed why isolated copying was unjustified: the official direct-op aggregate was not faster, and the short-IC win budget was only 10.96%.

### 3. The fixed geometry sits on a hard occupancy/live-state boundary

The promoted control uses 256 threads, BM128xBN128, eight warps, FP16/FP32 register counts of 97/128, and shared B. With a 65,536-register SM, 128 registers/thread is exactly 32,768 registers/CTA, allowing two CTAs/SM before other limits. A 145-176-register candidate consumes more than half the register file and drops to one CTA/SM. EXP-02 and EXP-09 did exactly that; EXP-07 FP32 did as well.

At M=2048 the natural grid is already large: 16 row tiles times 136 output tiles for 17408-wide FFN projections, and 16 times 40 for 5120-wide down projection. The problem is not a lack of total CTAs. It is the issue/latency balance inside each CTA and the ability to keep two CTAs resident while combining decode ALU with MMA work.

### 4. Decode ALU and MMA are coupled, not independently optimizable

The W2/W3 payload requires dependency mapping, ring-word selection, funnel shifts, and codebook reconstruction before MMA. Bee's control precomputes thread-invariant address terms and cooperatively materializes a tile. Warp-local variants multiply that work across M warps. EXP-09 reduced barriers 3->2 and removed B STS/LDSM, yet decode ALU grew about 2-2.7x and registers rose 48-53 over control. The saved memory instructions were not the dominant cost under that ownership.

The correct mixed accumulator improved safe short-IC families, but long-IC `ffn_down` still needs FP32 accumulation or a topology that avoids Stage 3's split-K tax. Thus the residual is not “too little MMA” alone; it is the balance among decode, accumulator footprint, publication, and occupancy.

### 5. The clean same-runtime evidence points to representation

BASE-01's same-runtime LowGPU comparison puts 94.6% of the positive gap in FFN. P-ARCH-21A removes only the packed FFN path and immediately reaches about 2814 tok/s. P-ARCH-23I removes the remaining packed body projections and reaches about 3300 tok/s. In contrast, graph/orchestration explains only about 16.8 ms and rotate/finalize together have an 11.3% projection-time upper bound.

The actual structural blocker is therefore: **the canonical packed sidecars are not in a representation that Bee's current BM128xBN128 SM120 kernel can decode and feed to MMA at standard-quant efficiency while retaining two-CTA residency.** Shared-B is the best discovered compromise for that representation; it is not the source of the whole 243-270 ms gap.

## Ranked breakthrough hypotheses

### 1. Promote and certify the P-ARCH-23I derived artifact

**Mechanism.** Reconstruct checkpoint-correct projections once, store them in standard low-bit GGML formats, and execute mature stock quant kernels. This removes the packed Escha decoder, rotations, partial buffer, and finalize from the body.

**Why this differs.** It is already measured at 619-621 ms / about 3300 tok/s. It does not assume the current packed kernel can be incrementally tuned into parity.

**Smallest discriminating experiment.** Freeze the P-ARCH-23I artifact and run the canonical paired 2K campaign plus the same medium 75-case quality certification used for the other controls. Accept as the deployment breakthrough if median prefill is at least 3200 tok/s, decode is no worse than 2%, standard-GGUF behavior remains isolated, artifact stays below 10 GB, and quality is no worse than one case below the canonical full-Escha 65/75 control.

**Risk.** It changes the artifact/representation and may violate a requirement that the exact full-Escha W2 packed artifact remain canonical. Its current quality evidence is lightweight; P-ARCH-23I was not one of the four artifacts in the recorded 75-case certification.

### 2. Runtime load-time transcode into a persistent standard low-bit cache

**Mechanism.** Preserve the canonical GGUF as the source of truth, but reconstruct/transcode selected sidecars once at model load into Q2_K/IQ3-class standard tensors, discard or avoid uploading the redundant packed hot tensors, and dispatch stock `MUL_MAT` for prefill/decode. Start with all FFN families, where P-ARCH-21A and BASE-01 predict the largest return.

**Why this differs.** The existing `ESCHA_CUBLAS_PREFILL` path dequantizes full weights per call, so it pays conversion repeatedly and is not a useful ceiling. A persistent cache pays conversion once and attacks representation directly while keeping the original file/hash as the interchange artifact.

**Smallest discriminating experiment.** Transcode only one layer's three FFN tensors at load time and validate numerical orientation/quality; then a bounded all-FFN prototype with memory accounting. Continue only if all-FFN full prefill gains at least 15%, peak resident memory remains within the 5090 target, and load-time conversion is deterministic/cacheable. P-ARCH-21A's approximately 2814 tok/s is the expected sanity target.

**Risk.** It is artifact substitution performed at runtime: load latency, extra host/GPU memory during conversion, cache invalidation, and quantization reproducibility become product concerns. If packed execution itself is the goal, this is not philosophically different from P-ARCH-23I.

### 3. New cooperative deep pipeline: retain broadcast, overlap B decode, enlarge K stage

**Mechanism.** Keep CTA-cooperative decode and shared/collective publication, but double-buffer decoded B and stage two or four 16-wide K tiles so decode of tile N+1 overlaps MMA on tile N. Use BK32/BK64 logical stages, scoped producer/consumer barriers, and possibly dedicated producer warps only if consumer MMA coverage remains sufficient. The design target is fewer synchronization points and hidden decode latency without 4x warp-local decode.

**Why this differs.** EXP-02/07/09 eliminated the broadcast; this preserves it. EXP-03 changed only CTA aspect; this changes temporal pipelining. It is also closer to the official kernel's high-shared-memory/low-register character than EXP-09's minimal-smem direct decode.

**Smallest discriminating experiment.** A K2 5120->17408 microkernel with unchanged BM128xBN128 and output contract, double-buffered shared B for BK32, compile/SASS gated before model timing. Proceed only with FP16 <=97-104 registers, FP32 <=128, no spills, at least two-CTA residency, and target-family matmul >=10% faster. Require aggregate full-2K >=5% before extending families.

**Risk.** Extra shared memory may itself cap occupancy; producer warps can starve MMA; more in-flight state can recreate the register problem. If cooperative decode compute rather than publication latency is dominant, overlap will be small.

### 4. A genuinely coupled `ffn_down` specialization

**Mechanism.** Design for K3 17408->5120/M-prefill as its own kernel rather than changing BM alone: reduce FP32 accumulator footprint with a narrower N ownership, choose BK/pipeline depth jointly, preserve decode-once reuse over M rows, and tune CTA threads/grid as a package. A persistent dequantized/cached cuBLAS or standard-quant path for `ffn_down` is also a valid member of this hypothesis; per-call full dequantization is not.

**Why this differs.** EXP-06 changed only BM and retained Bee's BN128/WN2 structure. P-ARCH-13 was a K2 width-only experiment. Neither tested a complete long-IC FP32 specialization. Stage 3 proves FP16 arithmetic can help but split-K is the wrong safety mechanism.

**Smallest discriminating experiment.** First compare two resource-complete paper designs, then compile only one: for example BM128xBN64 with a reduced thread/warp map and BK32 staging, exact K3 shape only. Fail fast unless registers fall materially below 128, no spill occurs, and direct `ffn_down` matmul improves at least 20%; keep only for at least 5% full-wall gain with no depth regression from M128 to M4096.

**Risk.** `ffn_down` was about 23.3% of Stage 2 aggregate matmul; even a large 30% family win yields only roughly 6-7% of matmul before overhead. It can reach the lower 2400+ target but cannot alone deliver 3300 parity. Narrower N may duplicate A traffic and repeat EXP-03's offsetting cost.

### 5. Repack the Escha code stream into an MMA-ready decode representation

**Mechanism.** Change the sidecar layout—not necessarily its bit rate—so payload words are already ordered by CTA/band/fragment and dependency/ring addressing is pre-resolved. The runtime would cooperatively expand a compact, contiguous fragment stream rather than execute `dep_pi`/modulo/funnel/address chains in every consumer. A modest metadata/index expansion is acceptable if it preserves most of W2's size advantage.

**Why this differs.** All rejected direct-fragment kernels retained the current code stream and therefore repeated its expensive indexing in warp-private registers. This hypothesis changes the producer representation so warp-local or producer-warp decode can actually be cheap. It is a middle ground between exact packed execution and full standard-quant substitution.

**Smallest discriminating experiment.** Offline repack one K2 5120->17408 projection into fragment order, write a direct-op decoder for only that representation, and compare bytes, size, registers, decode-ALU SASS, and matmul time. Continue only if size growth is <=25%, FP16 registers stay <=104, decode ALU drops by at least 30%, and the target family gains >=15%.

**Risk.** The current codec's compression may fundamentally depend on its ring/dependency structure, so pre-resolving it could approach dense/standard-quant size. This is a new artifact format and conversion contract, with significant correctness and tooling cost. It may converge technically on direction 1 with more complexity.

Graph-level up+gate transform sharing remains worth at most a smaller cleanup after one of the above. The measured rotate bound is 4.6-5.4%; a grouped op could reuse rotated A and reduce launches, but it is not a credible parity mechanism by itself. Likewise, CUDA graphs cannot erase the measured FFN body deficit.

## What to stop doing

1. **Stop removing shared-B at BM128xBN128 without a pre-build ownership proof and a compile gate of control-class registers.** EXP-02, EXP-07, and EXP-09 are three independent negatives. “One B fragment live” is insufficient if decode temporaries and work are still repeated four times.
2. **Stop standalone tile-aspect sweeps.** 256x64 was neutral, BM64 was 47% slower in the target family, and the official knob was a coupled kernel selection. A new shape is admissible only as part of a fully specified pipeline/warp/register design.
3. **Stop warp-pair finalize fusion and do not combine input/output fusion into a claimed breakthrough.** EXP-08 was -18.3%; rotate plus finalize has only an 11.3% best-case projection bound. A future finalize design must prove full-CTA parallelism first.
4. **Stop treating P-ARCH-20's -44.22% as evidence.** It was contaminated. The correct mixed policy is already promoted; another accumulator-only experiment is redundant unless it solves long-IC accumulation without split-K overhead.
5. **Stop speculative clones of the official short-IC mainloop.** EXP-05's credible winning coverage was 10.96%, and the official direct-op aggregate was slower because of down projection. The external server number is a target, not proof that one visible kernel feature transfers.
6. **Stop per-call full dequantization as a production proposal.** cuBLAS is relevant only with persistent/load-time conversion or a measured reuse strategy.
7. **Stop graph-only, launch-only, microbatch, LUT, or scheduler work as the next breakthrough attempt.** Their measured/credible bounds are single-digit percentages and do not address the flat FFN family deficit.
8. **Stop donor substitution without semantic verification.** P-ARCH-21B's wrong `attn_gate` proves name/shape compatibility is not weight compatibility.

## Recommended next action for Sean

Make the architecture requirement explicit before authorizing another implementation:

### Decision A — parity/product outcome is primary

Freeze P-ARCH-23I and run one milestone certification. Promote it if all of the following hold:

- paired canonical 2K median >=3200 tok/s and <=640 ms;
- medium suite >=64/75, with no unexplained pack-level collapse versus the 65/75 canonical Escha control;
- decode regression <=2%, stable through the depth/context matrix;
- artifact <=10 GB and exact reconstruction/quantization fingerprints retained;
- the same gated runtime preserves standard Qwen behavior.

This is the highest-confidence path and is already within reference parity in development measurements.

### Decision B — the exact packed full-Escha execution path is a hard requirement

Authorize **one** architecture-level discriminating experiment: direction 3, cooperative BK32 double-buffered B on the dominant K2 family, with the resource gate before correctness/timing. Continue the packed-runtime program only if it achieves all of:

- no duplicated row-warp decode;
- at least two resident CTAs/SM by measured resources (no spills; FP16 near control, FP32 <=128 registers);
- target-family matmul >=10% faster;
- full 2K >=5% faster, at least 4/5 samples over control median, and no family/depth regression >5%.

If that gate fails, stop incremental runtime work on this kernel. The remaining choices are the load-time transcode cache (direction 2) or a separately funded new sidecar representation/kernel project (direction 5). Do not proceed to another direct-fragment, aspect-ratio, or fusion variant.

### My recommendation

Choose Decision A unless “execute the exact packed W2 sidecars directly” is itself a product requirement. P-ARCH-23I has already answered the performance question; the runtime-only series has answered the incremental-kernel question. The next useful evidence is certified quality for the parity artifact, not a tenth local mainloop rewrite.
