# A4 MMA sidecar V2 review and breakthrough recommendation

Date: 2026-09-02  
Scope: read-only review of the K2 `blk.0.ffn_gate` V2 mechanism result and the
remaining path to a compact, exact PACKED Escha prefill kernel.

## Executive verdict

**V2's `CONFIRM-REJECT` is correct for V2 and for the descriptor-driven,
decode-once-to-shared consumer. It is not sufficient evidence to close every
exact-packed implementation.** V2 is bit exact and fixes V1's 4x decode,
register, and occupancy failures, but it is decisively slower than the same
standalone shared-B control: **2.408634 ms versus 1.591085 ms (1.514x time)**.
It would need a 33.94% time reduction merely to tie control and a 43.85%
reduction to pass the original 15%-faster gate (1.352422 ms).

The important correction is that V2 did **not** reproduce the official
`escham_code_gemm` mainloop. The official kernel's 40
`WARPSYNC.COLLECTIVE` + 40 `ENDCOLLECTIVE` pairs are in its fused Hadamard
epilogue around `SHFL.BFLY`; they are not a decoded-B publication mechanism.
The official hot mainloop instead performs warp-local direct decode, has no
complete shared-B write/reload, and executes a deeply unrolled schedule: the
raw K2 symbol has **64 HMMAs and four distinct funnel/decode groups between a
mainloop barrier and its loop backedge**, versus V2's 16-HMMA, one-K16 loop
body. Its 45,056 B of shared memory is real deep A/payload staging. V2 copied
the allocation size but used only a two-slot A ring and a single decoded-B
tile.

Therefore the missing piece is not another form of collective publication.
It is an **official-shaped, four-K16 software pipeline that interleaves fixed
packed-word extraction, A-fragment loads, and HMMAs while keeping B fragments
in registers**. To make that viable for Bee, the hot loop must also avoid
V1/V2's runtime descriptor gathers. This is the only untried exact-packed
direction with a credible mechanism for both removing publication latency and
hiding decode ALU.

The probability assessment is deliberately conservative:

- Reaching the **~3030 tok/s** packed target is plausible only if Bee reaches
  near-official mainloop scheduling across the dominant families and preserves
  the official-class epilogue/graph advantages.
- Reaching **3339 tok/s** while retaining exact packed residency is not yet
  supported by the evidence. The same-bin packed control is 2529 tok/s, so
  3030 requires +19.81% and 3339 requires +32.03%. The official runtime proves
  roughly the first target, not the second.
- The highest-certainty delivery route remains the measured standard-GGML
  hybrid frontier: P-ARCH-23 is 8.599 GB / 3023 tok/s and P-ARCH-23G is
  8.808 GB / 3228 tok/s. That is a delivery fallback, not an exact-packed
  breakthrough.

## 1. What V2 proves—and what it does not

### The result is real

The V2 harness is a valid discriminating slice result:

| property | control | V2 | finding |
|---|---:|---:|---|
| direct op, M=2048 | 1.591085 ms | 2.408634 ms | V2 1.514x slower |
| FP16 registers | 85 | 96 | both retain 2 CTA/SM |
| FP32 registers | 116 | 128 | both retain 2 CTA/SM; V2 is at the ceiling |
| dynamic shared | 13,824 B | 45,056 B | both still report 2 CTA/SM |
| integer address/decode ALU | 93 | 137 | V2 +47.3% |
| bit mismatches | -- | 0 / 35,651,584 | exact |

The benchmark alternated the arms, used warmups and repeated timed launches,
and compared the entire output bitwise. This is strong enough to reject V2;
the 0.818 ms difference is far outside a marginal-call decision.

V2 also conclusively repairs V1's failure. Per CTA/K16 it evaluates 2,048
weights once, rather than V1's 8,192 evaluations. FP16 registers fall from
140 to 96, FP32 from 168 to 128, spills remain zero, and residency returns
from one to two CTAs/SM. The fact that it is still slower is valuable: **4x
decode duplication was a V1 defect, but eliminating duplication is not by
itself the winning architecture.**

### Where the 1.51x loss lives

The current evidence cannot assign exact milliseconds to individual causes;
there is no V2 ablation or hardware-counter decomposition. The defensible
attribution is the extra serialized critical path visible in source and SASS:

1. Each band first stages a 68-word record to shared memory.
2. Each thread loads four runtime descriptor words from shared.
3. Each descriptor is unpacked into two offsets and two shifts.
4. Four adjacent record-word pairs are gathered from shared, funneled, and
   evaluated by the codebook.
5. The results are stored to shared B and published through a 128-thread
   cooperative-group synchronization.
6. All four row warps reload that B tile with `ldmatrix`, then issue MMA.

The matched FP16 symbols make the added machinery visible. V2 has 137 versus
93 `IMAD+LOP3+IADD3`, 42 plain `LDS` versus 6, the same 10 `LDSM`, 16 HMMAs in
both, and a more complicated synchronization envelope: four static `BAR`,
four `MEMBAR.ALL.CTA`, and two `WARPSYNC.ALL` occurrences versus control's
three static `BAR` and no `MEMBAR`/`WARPSYNC`. These are static symbol counts,
not normalized dynamic instruction counts, but the one-K16 loop bodies are
matched, so the direction is meaningful.

The **45 KiB allocation is not itself the measured cause**. V2 intentionally
reserves the official class while using only about 14,880 B: 2,048 B hot
descriptor words, 8,192 B for the two active A slots, 544 B for records, and
4,096 B for decoded B. Roughly 30 KiB is inactive reserve. Since control and
V2 both achieve two CTAs/SM, the footprint does not explain the 1.51x result.
The relevant costs are actual descriptor/record/shared-B traffic and the
serial publication schedule.

### Could a small V3 reverse it?

**Descriptor table in constant/global memory: no.** V2 already pays the
global descriptor read only once per CTA, compacts the 512 hot words into
2 KiB of shared memory, and reuses them across 320 K stages. Moving 512
lane-distinct accesses to constant memory is not a broadcast; moving them to
global trades shared loads for cached global loads. It can remove the entry
copy/barrier, but it does not remove the four hot descriptor reads, bitfield
unpacks, address formation, record gathers, codebook work, shared-B publish,
or B reload. It is not credible as a 34-44% time reduction.

**Fully sequential per-warp materialization: only if "sequential" does not
mean pre-extracted indices or values.** A true `LDG -> codebook/MMA` stream of
the two 16-bit indices per descriptor requires at least 16 bits/weight. For
this K2 gate, canonical packed code is 22,282,240 B, while the exact fp16
matrix is 178,257,920 B: exactly **8x**. Even the current 23,674,880 B ring
payload would grow 7.53x. Materializing repeated adjacent operand pairs has
the same fundamental problem. That violates the <=25% sidecar gate and the
8.5 GB model objective. A 17-word-ring permutation can remain at +6.25%, but
then some lane-to-window mapping, shuffle, or address arithmetic remains; it
is not a free straight stream.

**Wider K work: yes, and it has the best expected value.** K3 by itself does
not help; it increases packed work. A wider **BK64 software stage (four K16
tiles)** can amortize synchronization and expose independent shared loads,
decode ALU, and tensor-core work to the scheduler. The official SASS is direct
evidence for precisely this shape. V2's one-tile phases expose no such
latency-hiding window. This is the one structural change that could plausibly
recover 34%+, although it should not retain V2's shared-B publication if the
goal is official-class performance.

So: V2 should stay rejected. A descriptor-placement V3 should not be funded.
A final V3 is justified only as a new **deep-pipeline/direct-fragment
mechanism experiment**, not as an incremental patch to V2.

## 2. What the official kernel actually does

### Correct structural comparison

| feature | Bee shared-B control | V1 sidecar | V2 sidecar | official K2 hot kernel |
|---|---|---|---|---|
| B decode ownership | once/CTA to shared | repeated by 4 row warps/band | once/band to shared | warp-local/direct; therefore repeated across row warps |
| B handoff | STS + CTA barrier + LDSM | B registers | STS + band publication + LDSM | B registers; 0 decoded-B `STS.U16` |
| K schedule | one K16 loop body | one K16 loop body | one K16 loop body | four-K16-class unrolled body |
| HMMA in static hot body/symbol | 16 | 16 | 16 | 64 |
| A/payload staging | 2-slot A ring, 13.8 KiB total | 2-slot A ring | 2 active A slots despite 45 KiB allocation | deep staging, 45,056 B genuinely addressed |
| FP16 registers | 85 harness / 97 runtime | 140 | 96 | 80 |
| decoded-B publication sync | full CTA | none | band group | none |
| `40+40` collectives | none | none | none | Hadamard epilogue, not B publication |

V2's plan equated its band publication with the official
`WARPSYNC/ENDCOLLECTIVE` pairs. The raw SASS disproves that mapping. The
collective pairs occur after the mainloop, each enclosing butterfly shuffles
in the fused output transform. They explain part of the official end-to-end
advantage by avoiding Bee's separate finalize path, but they do not coordinate
weight decode.

The official mainloop instead uses its shared memory as a **latency-hiding
pipeline**. Between the barrier at the top of its mainloop and the backedge,
the K2 SASS contains four packed funnel/extraction groups, 64 HMMAs, and large
runs of scalar LDS interleaved with codebook ALU and HMMAs. It also issues
multiple `LDGSTS` copies for future A data. The 136 LDS count is not evidence
of a shared-B round trip; it is the feed schedule for several in-flight A and
packed-payload fragments.

### Which proposed gap is the real one?

- **Band geometry: not the gap.** V2 already covers a 128x128 CTA as two
  contiguous 64-column bands with four warps each, matching the official grid
  inference. The official gate sweep also preferred BM128 over BM64 (1.34 ms
  versus 1.53 ms).
- **True producer-warp specialization: not the gap.** The official kernel
  does not reserve producer warps. All warps retain MMA ownership and perform
  direct packed decode. Removing two of eight MMA warps would cut nominal
  tensor ownership by 25% unless the CTA geometry or instruction family also
  changes, and it adds a handoff/barrier that the official kernel avoids.
- **A-fragment/decode/MMA overlap plus K-stage depth: this is the gap.** They
  are two faces of the same software pipeline. Deep staging creates enough
  independent work; the generated schedule interleaves LDS, fixed shifts,
  codebook operations, and HMMA rather than running record publication,
  complete decode, B publication, B reload, and MMA as serial phases.
- **Native fixed extraction is the enabling representation property.** The
  official hot loop does not perform V2's four runtime descriptor reads and
  bitfield unpacks. V1 matched direct B registers but retained a shallow,
  descriptor-driven gather and paid 140 registers/220 address ALU. V2 matched
  decode-once but retained the descriptor gathers and a publication boundary.
  Neither combined fixed extraction with the deep direct-fragment schedule.

The official kernel therefore wins **despite** warp-local decode duplication,
not because it solved duplication with collectives. It makes duplicated decode
cheap enough and overlaps it with useful tensor/A-load work. That is the
central architecture lesson from V2.

## 3. Ranked breakthrough hypotheses

Ranking is against the complete goal: exact Escha semantics, compact resident
representation, and a credible path past the shared-B control.

| rank | direction | expected value | principal risk | size impact | judgment |
|---:|---|---|---|---|---|
| **1** | **(b, refined) Fixed-window packed stream + official-shaped four-K16 direct-fragment pipeline** | Highest among exact-packed options | difficult code generation/register scheduling; a descriptor-free <=25% layout must be proven | target +6.25%, hard cap +25% | **Only credible packed breakthrough** |
| **2** | (a) dedicated decode producer warps feeding MMA consumers | Low-medium | loses 2/8 or more MMA warps, requires handoff buffers and scoped barriers; no official precedent | small | mechanism may hide decode but likely starves tensor work |
| **3** | (c) narrower 64-column/128-thread CTA geometry | Low | does not reduce total decode over all M tiles; reduced row reuse and more CTAs | none | official BM64 gate was 14% slower; Bee BM-only EXP-06 was 46.95% slower |
| **4** | (e) hot-FFN fp16-fragment overlay, packed cold paths | High speed certainty, low goal value | only a tiny subset can fit, too little wall coverage; a useful subset adds GBs | severe | can make selected ops fast, cannot plausibly keep ~8.5 GB and gain 20-32% wall |
| **5** | (d) broad predecoded fp16 fragment-order representation | Highest raw speed, disqualifying objective fit | ceases to be compact packed residency | K2 projection 8x packed | not a solution to Sean's size/speed conjunction |

The parenthetical claim that fp16 is “~2x growth from 2.5 bpw” is incorrect:
16 / 2.5 = **6.4x**, and this measured K2 tensor is 2 bpw, so it is **8x**.
Fragment order changes access, not information size. A predecoded-fp16 cache
is useful as a diagnostic upper bound or an opt-in large-memory mode, but it
cannot be called the ~8.5 GB breakthrough.

The refined rank-1 direction also has a hard truth: if a descriptor-free
fixed-window stream cannot be represented under +25% without storing u16
indices, repeated operand pairs, or fp16 values, the sidecar representation
adds no useful degree of freedom. In that case the right implementation target
is the canonical packed stream with an official-shaped generated schedule,
not another sidecar format.

## 4. Recommended final experiment: `V3-PIPE4`

### Hypothesis

V2 loses because one K16 tile is executed as a serial
record-stage/decode/publish/reload/MMA transaction. An exact packed consumer
will beat shared-B only if four K16 tiles are simultaneously available and
the compiler can interleave fixed packed-word extraction and A LDS with 64
HMMAs, with no decoded-B publication boundary.

### Frozen scope

- Standalone `blk.0.ffn_gate`, K2 5120 -> 17408, M=2048 only.
- Same deterministic input, output layout, FP16 accumulate, and bitwise
  comparison as V2.
- No loader/full-model integration until the slice passes.
- Standard Qwen and non-Escha paths remain out of scope; any eventual dispatch
  remains gated by `qwen35.escha.version` and architecture capability.
- No fp16/u16 weight materialization. Representation growth remains <=25% and
  reverse-transform exact.

### Required mechanism—not optional implementation flavor

1. Group four K16 records into one BK64 superstage and genuinely use the deep
   A/payload staging arena.
2. Keep all eight warps as MMA warps. Do not introduce dedicated producers.
3. Build one B fragment at a time in registers and consume it immediately;
   there must be no complete shared decoded-B tile, decoded-B `STS`, B
   `LDSM`, or decoded-B publication barrier.
4. Remove runtime descriptor-table reads from the steady-state loop. Use a
   fixed K2/K3 generated mapping over an admissible <=25% packed-word layout.
   If achieving a straight stream requires storing pre-extracted 16-bit
   indices or repeated word pairs beyond the cap, reject the representation
   before writing the kernel.
5. Fully unroll the four-stage accumulator accesses. EXP-10's schedule was
   never timed because runtime-indexed accumulator arrays were homed to 128/
   256 B of stack; `V3-PIPE4` must produce `STACK=LOCAL=0` by construction.

### Explicit mechanism gate

All hard rows must pass:

| row | hard threshold | why it discriminates |
|---|---|---|
| representation | <=25% growth; byte-exact reverse; no u16/fp16-per-weight stream | preserves the actual goal |
| hot-loop descriptor traffic | zero descriptor-table LDG/LDS in the repeated superstage | proves V1/V2 gather path is gone |
| pipeline shape | 64 FP16 HMMAs per four-K16 repeated body; no more than one repeated CTA rendezvous per four K16; future-A async copies precede/interleave the body | proves real depth rather than a 45 KiB reservation |
| B path | zero decoded-B shared store, zero B `LDSM`, zero decoded-B publication barrier | matches the official direct-fragment mechanism |
| resources | FP16 <=104 regs, FP32 <=128; STACK/LOCAL/spills zero; >=2 CTAs/SM | prevents V1/EXP-09/EXP-10 recurrence |
| normalized decode/address ALU | count the repeated four-stage region and divide by four; <= control per-K16 in both modes, with SHF reported separately | avoids the misleading whole-symbol/unroll comparison |
| correctness | zero bit mismatches over all 35,651,584 outputs | preserves exact semantics |
| direct operator | **<=1.352422 ms** median under the V2 alternating protocol | at least 15% faster than the 1.591085 ms shared-B control |

Record two diagnostic thresholds without weakening the hard gate:

- `<=1.591085 ms`: proves the packed direct pipeline can beat shared-B at all.
- `<=1.40 ms`: official-class neighborhood; enough to justify family
  expansion even if the frozen 15% row narrowly misses.

Run one attribution ablation only if the full candidate passes resources but
misses time: compile the identical four-stage schedule with descriptor reads
restored. The delta answers whether fixed-window extraction or pipeline depth
delivered the gain. Do not launch a sequence of descriptor-memory-placement
variants.

### Decision after the experiment

- **Pass <=1.352422 ms:** expand to K3/down and qkv shapes, preserving the
  complete coupled geometry; then run a family-weighted wall budget before
  integration. Do not infer full-model 3030 from one gate tensor.
- **Between 1.352422 and 1.591085 ms:** the mechanism is real but insufficient;
  continue only if the family-weighted upper bound reaches the 19.8% wall gain
  needed for 3030.
- **>=1.591085 ms, any resource failure, or representation >25%:** close the
  exact-packed runtime line. The remaining rational delivery choice is
  P-ARCH-23/P-ARCH-23G-class standard-GGML hybridization.

## Evidence anchors

- `TRUE-GOAL.md`: size/speed conjunction and measured frontier.
- `V2-GATE.md`, `V2-GATE.json`, `v2-raw/benchmark.json`,
  `v2-raw/harness.sass`, and `v2-raw/cuobjdump-resources.txt`: V2 result.
- `SLICE1-GATE.md`: V1 4x-decode/register/occupancy failure.
- `FRONTIER-RESULTS.md`: same-bin P-ARCH-23/23G/23I evidence.
- `EXP-05-audit/2026-09-01/AUDIT.md` and `hot-k2-sass.txt`: official
  resources, direct-op timings, mainloop, and epilogue collectives.
- `EXP-09-mainloop-rewrite/2026-09-02/REJECTION-REPORT.md`: shallow
  warp-local direct-decode register/ALU failure.
- `EXP-10-coop-bk32/2026-09-02/REJECTION-REPORT.md`: cooperative overlap
  schedule invalidated by accumulator homing, not by measured timing.
- `ggml/src/ggml-cuda/escha-moe.cu`: promoted shared-B control structure.
