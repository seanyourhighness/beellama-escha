# A4 MMA sidecar V2 — collective-consumer Slice 1 plan

Date: 2026-09-02  
Scope: standalone `blk.0.ffn_gate`, K2 5120 -> 17408, M=2048. V1 overlay bytes are frozen; no loader or model-runtime source is in scope.

## Decision

Keep `escha-mma-cache-v1` unchanged and replace only its failed consumer. V2 is a 256-thread, 128-row x 128-column CTA split into two contiguous four-warp, 64-column bands. Each four-warp band collectively decodes its band record exactly once into one shared `[64][16]` half tile. The same four warps then load that single publication with `ldmatrix`; no row warp reloads or re-evaluates another row warp's weights.

This differs from both prior failures:

- EXP-09/A4 V1: every row warp decoded all 1024 weights in its band (4x decode). V2 decodes 1024 weights once across the four warps.
- EXP-10: all warps decoded every tile while overlapping a two-slot B ring, and runtime-indexed accumulators were homed to stack. V2 has one B publication, no next-B overlap/ring, and fully unrolled fixed-index MMA/accumulator access.

## Frozen representation

V2 consumes the exact A4 file: 256-byte header, one 512 x 16-byte descriptor table, and `[CTA][K-stage][band]` 272-byte records. Each record is four 17-word descending rings. `runtime_access_u32` still supplies two pre-resolved `(word_offset, shift)` pairs, so the decode path is adjacent loads + funnel shift + unchanged codebook. No repack or growth change is permitted.

Effective representation remains 23,728,384 bytes including reused rotations versus 22,327,296 canonical bytes: 6.275224729% growth.

## Ownership and cooperative decode

Thread geometry is rearranged only to make each band a contiguous cooperative group:

```text
lane       = threadIdx.x                 // 0..31
warp       = threadIdx.y                 // 0..7
band       = warp >> 2                   // warps 0..3 or 4..7
row_group  = warp & 3                    // 32 output rows/warp
band_tid   = row_group*32 + lane         // 0..127
```

Each band group covers the 512 descriptors with four descriptors/thread:

```text
desc(g) = band_tid + 128*g, g=0..3
```

For fixed `band_tid`, the descriptor sequence advances fragment `j` by two, so its publication coordinates are:

```text
base_j = band_tid >> 6
slot   = (band_tid >> 5) & 1
flane  = band_tid & 31
col(g) = 8*base_j + (flane >> 2) + 16*g
row0   = 8*slot + 2*(flane & 3)
row1   = row0 + 1
```

Each descriptor produces the two half values named by its pre-resolved runtime word and stores them at `[band*64 + col(g)][row0,row1]`. Thus 128 threads x 4 descriptors x 2 values = 1024 unique values = 64 x 16 per band. Across both band groups, the CTA performs exactly 2048 codebook evaluations, matching one decoded 128 x 16 B tile. A4 V1 performed 8192 evaluations because four row warps independently reconstructed each band.

The 136 record words for the two bands are first loaded once into shared memory (68 loaders/band). Decode reads the adjacent ring words from this shared record publication, not repeatedly from global memory.

## Shared layout

The launch uses the official kernel's measured 45,056-byte class:

| region | bytes | use |
| --- | ---: | --- |
| A4 descriptor staging arena | 8,192 | all 512 `runtime_access_u32` words are compacted into the first 2,048 B once/CTA; remaining bytes preserve the official-class layout/reserve |
| eight 128x16-half A-stage slots | 32,768 | slots 0/1 are the active cp.async ring; inactive arena begins with the 544-byte two-band record publication |
| one decoded 128x16-half B tile | 4,096 | two disjoint 64-column band publications |
| **total** | **45,056 B** | **44.0 KiB/CTA** |

The extra A-stage capacity deliberately preserves the 45,056-byte official-class allocation while Slice 1 changes only the consumer. Two CTAs require 90,112 bytes, below the RTX 5090 SM shared-memory limit; registers must also permit two CTAs.

## Collective and barriers

At kernel entry all threads copy two descriptors, issue A[0], then one CTA rendezvous makes the descriptor table visible. Per K16 tile:

1. Each band group loads its 68-word record; the A pipeline drains; one CTA barrier publishes records and A.
2. Each four-warp band collectively executes its disjoint four-descriptor/thread assignment into its own shared `[64][16]` tile.
3. One 128-thread cooperative-group barrier per band publishes decoded B. This is the semantic V2 mapping of the official WARPSYNC/ENDCOLLECTIVE pair: begin converged band work, then end/rendezvous before any band warp consumes the publication. It is not a per-row-warp decode.
4. All four row warps load the same band bytes with `ldmatrix` and issue the unchanged 16 MMAs/warp/tile.
5. One CTA barrier protects the shared A/B/record regions before reuse.

Publication cost is one CTA input-publication barrier plus one band-local decoded-B publication barrier per tile; the final CTA reuse barrier is not a publication. There is no B ring, producer warp, replay, cross-band exchange, or runtime-indexed MMA helper.

## Register forecast and residency

| live category/thread | official K2 FP16 | A4 V1 FP16 | V2 FP16 forecast | V2 FP32 forecast |
| --- | ---: | ---: | ---: | ---: |
| accumulators | 32 | 32 | 32 | 64 |
| two A fragments | 8 | 8 | 8 | 8 |
| one streamed B fragment | 2 | 2 | 2 | 2 |
| descriptor runtime word | not observed | repeated transient | 1 | 1 |
| one cooperative decode transient | bounded | repeated per fragment | 6-10 | 6-10 |
| pointers/loop/store/compiler state | remainder | large duplicated path | 30-44 | 38-46 |
| **total** | **80 measured** | **140 measured** | **82-100** | **118-128** |

The decode transient dies before `ldmatrix`/MMA, and the B fragment is streamed one `j` at a time. Fully unrolled `[i][j]` accesses forbid EXP-10 accumulator homing. PASS remains FP16 <=104, FP32 <=128, stack/local/spills zero. Register math at the ceilings is 104 x 256 x 2 = 53,248 and 128 x 256 x 2 = 65,536 registers, so two CTAs fit the 65,536-register SM pool.

## Frozen gate and protocol

`run_slice1_v2.py` reuses `repack_slice1.py`, compiles an sm_120a standalone harness, generates the same deterministic M=2048 activations, runs the current packed control and V2 FP16 candidate with two warmups and five alternating timed pairs, and performs bitwise FP32-output comparison. It also builds FP32 symbols for resource/SASS inspection.

All rows must pass:

| row | threshold |
| --- | --- |
| representation | growth <=25% and reverse byte exact |
| correctness | zero output bit mismatches |
| resources | FP16 <=104, FP32 <=128, stack/local/spills zero |
| decode/address ALU | V2 at least 30% below actual packed control in both modes |
| residency | at least 2 x 256-thread CTAs/SM in both modes |
| direct op | V2 at least 15% faster than packed control at M=2048 |

Every row is independently reported. Any failure yields `CONFIRM-REJECT`; no model integration follows.
