# A4 / EXP-11 Attempt 3 — discriminating Slice 1 plan

Date: 2026-09-02  
Scope: exactly `blk.0.ffn_gate`, K2, 5120 -> 17408, direct-op only.  No
loader, graph, model-runtime, or canonical GGUF mutation is in scope.

## Decision being tested

Build `escha-mma-cache-v1` as an exact accelerator overlay and compare a
standalone copy of the promoted packed MMA mechanism with a direct-fragment
consumer.  The candidate must clear every frozen mechanism gate.  A failure
ends Attempt 3 before any model integration.

## Projection and canonical accounting

The source tensors are read from the funded canonical GGUF:

| tensor | shape | bytes |
| --- | ---: | ---: |
| `blk.0.ffn_gate.escha_code` | `[320][1088][32]` i16 | 22,282,240 |
| `blk.0.ffn_gate.escha_rin` | `[5120]` f16 | 10,240 |
| `blk.0.ffn_gate.escha_rout` | `[17408]` f16 | 34,816 |
| **canonical packed code + rotations** | | **22,327,296** |

There are 320 K-stages, 136 output CTAs, two 64-column bands per CTA,
four 16-column tiles per band, and 16 u32 payload words per K2 tile.

## `escha-mma-cache-v1` binary layout

All integers are little-endian.  The file is:

```text
[256-byte header]
[512 x 16-byte shared lane descriptors]
[136 CTA][320 K-stage][2 band][272-byte warp publication record]
```

The order is therefore exactly
`[projection][output-CTA][K-stage][64-col band][warp publication record]`.
One band record contains four tile-local rings.  Canonical tile words
`p[0..15]` become the 17 contiguous words
`[p0,p15,p14,...,p1,p0]`.  For every canonical pair `(dw0,dw1)`, where
`dw1=(dw0-1) mod 16`, the two funnel operands are now adjacent.  The repeated
terminal `p0` is the only per-tile payload expansion.

The shared descriptor index is `[fragment_j][half2_slot][lane]`, or
`j*64 + slot*32 + lane`.  One 16-byte descriptor contains:

```text
row0,row1,fragment_col,half2_slot,dep_pi0,dep_pi1,
dw0_0,dw1_0,dw0_1,dw1_1,dsh0,dsh1,runtime_access_u32
```

`runtime_access_u32` packs `word_offset0:7, dsh0:5, word_offset1:7,
dsh1:5`.  The offsets already include the tile-within-band and descending-ring
mapping.  Thus each fragment lane loads one descriptor and two adjacent-word
pairs, performs the unchanged two funnel shifts and two codebook evaluations,
and forms one `half2`.  `fragment_col`, `row*`, and `dep_pi*` are explicit
oracle fields; no runtime reconstruction of `escha_dep_pi`, `sp`, `g0`,
`dw0`, `dw1`, `dsh`, tile, fragment row, or fragment column is required.
The table is stored once per overlay, not once per record or weight.

The header records dimensions, offsets, record counts, source tensor hashes,
and descriptor hash.  The manifest records the full overlay/payload hashes and
the exact GGUF byte offsets.

### Size forecast

| component | bytes |
| --- | ---: |
| header | 256 |
| descriptors | 8,192 |
| records: `136*320*2*4*17*4` | 23,674,880 |
| existing rin/rout reused by the op | 45,056 |
| **effective representation** | **23,728,384** |

Growth is `(23,728,384 / 22,327,296 - 1) * 100 = 6.276%`, below 25%.
The canonical code remains the fallback on disk but is not part of candidate
VRAM residency.  No decoded u16 codebook index or per-weight LUT index is
stored.

## Direct consumer and EXP-09 distinction

The control is the current 128x128, 256-thread, K-stage-16 packed mechanism:
cooperative canonical payload publication, one decode per weight to shared
`s_w[128][16]`, `ldmatrix` B consumption, and identical FP16/FP32 MMA/store
seams.  The candidate keeps the same CTA, A pipeline, warp output ownership,
K-stage order, MMA count, accumulation mode, and output layout.  It removes
shared-B materialization and reads fragment operands from the band record.

EXP-09 recomputed this chain for every direct-fragment weight in four row
warps:

```text
escha_dep_pi -> sp/mod -> g0 -> dw0/dw1 -> dsh -> payload address
```

That produced 310/450 decode-address ALU instructions and 145/176 registers.
Slice 1 replaces the entire chain with one coalesced descriptor lookup whose
hot word directly names both adjacent operand pairs and shifts.  The four row
warps still consume the same band, but they repeat only loads, funnel shifts,
and the exact codebook—not the index chain.  This is the discriminating
difference, not a scheduling tweak to EXP-09.

## Register forecast and residency

| live category / thread | FP16 | FP32 | discipline |
| --- | ---: | ---: | --- |
| accumulators | 32 | 64 | fixed `2x8`, never runtime-indexed |
| two A fragments | 8 | 8 | unchanged |
| one streamed B fragment | 2 | 2 | dies before next `j` |
| descriptor + two operand pairs + codebook | 7-9 | 7-9 | scoped inside one slot |
| lookahead / pointers / loop and store state | 38-51 | 38-45 | compiler forecast |
| **forecast ptxas total** | **87-102** | **119-128** | measured result decides |

Arrays of B fragments are forbidden in the candidate.  Accumulators remain
fully unrolled to avoid EXP-10 stack homing.  No `maxrregcount` is permitted.
PASS requires FP16 <=104, FP32 <=128, STACK=LOCAL=0, no `LDL`/`STL` spill
traffic, and runtime occupancy of at least two 256-thread CTAs/SM.  At the
register ceilings, `104*256*2=53,248` and `128*256*2=65,536`; candidate dynamic
shared memory is only 8,192 bytes/CTA.

## Harness and frozen mechanism gate

`repack_slice1.py` validates Escha v1 metadata and the exact tensor triplet,
hashes the source tensors, emits the overlay atomically, reopens it, validates
every header/descriptor/payload byte, reverses every record, and proves exact
canonical-code recovery.

`slice1_harness.cu` is standalone CUDA.  It reads the canonical code directly
at the manifest byte offset and the overlay payload, generates deterministic
M=2048 half activations, runs matched warmups and alternating timed launches,
checks bitwise FP16 output equality, reports CUDA function attributes and
occupancy, and exposes FP16 plus FP32 symbols for resource inspection.
`run_slice1.py` builds for sm_120a, captures ptxas/cuobjdump/SASS evidence,
counts `IMAD+LOP3+IADD3` in the actual harness control and candidate symbols,
executes the benchmark, and writes the gate reports.

All rows must pass:

| gate | threshold |
| --- | --- |
| representation | growth <=25% |
| resources | FP16 <=104, FP32 <=128, zero stack/local/spills |
| decode/address ALU | candidate at least 30% below actual control symbol |
| residency | >=2 CTAs/SM |
| direct op | candidate at least 15% faster at M=2048 |

Any failure produces `CONFIRM-REJECT`, stops before full-model work, and closes
the packed-exact +50% line.
