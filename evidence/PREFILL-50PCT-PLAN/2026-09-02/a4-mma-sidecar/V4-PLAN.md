# V4-PIPE — cooperative decode-once deep B-ring plan

Date: 2026-09-02  
Scope: standalone `blk.0.ffn_gate`, K2 5120 -> 17408, M=2048. V1/V2/V3 and runtime/loader sources remain unchanged.

## Decision

Test one mechanism at two compile-time depths: `D=2` (BK32 superstages) and
`D=4` (BK64 superstages). Each 128-thread, four-warp band reconstructs every
`[64,16]` B tile exactly once into a double-banked shared ring. The two bands
run opposite schedules: band 0 decodes the next superstage and then consumes
the current one, while band 1 consumes current and then decodes next. Because
the bands have disjoint B, payload, accumulator, and output ownership, their
warps may issue cooperative decode ALU/STS concurrently with the other band's
LDSM/HMMA stream. This is the smallest standalone test of decode-once plus a
deep ring; it changes neither GGUF loading nor BeeLlama runtime dispatch.

The byte representation is V3's descriptor-free, fixed-mapping payload without
change. `repack_slice1_v3.py` is reused rather than manufacturing another copy
of identical bytes.

## What binds at two CTAs/SM

The repeated-body SASS and direct timings separate the two candidate walls:

| body | HMMA | BAR | B STS | total LDSM | SHF | IMAD+LOP3+IADD3 | direct op |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared-B control, K16 | 16 | 3 | 10 STS total | 10 (2 A + 8 B) | 8 | 27 | 1.591085 ms |
| V2 collective, K16 | 16 | 2 in SASS | 5 STS total | 10 (2 A + 8 B) | 20 | 69 | 2.408634 ms |
| V3 direct, BK64 | 64 | 1 | 0 | 8 A-only | 130 | 410 | 3.456275 ms |

V2 has no four-row-warp reconstruction duplication, and its static body has
fewer BAR and STS instructions than control, yet it is 1.51x slower. Its
critical path is record publication -> descriptor LDS/address extraction -> 20
SHF plus 69 decode/address ALU -> decoded-B publication -> LDSM -> 16 HMMA.
The descriptor/record path expands the serial pre-MMA instruction stream from
the control's 27 ALU + 8 SHF to 69 + 20. The shared-B round trip is therefore
not independently binding: reducing its SASS count did not win.

V3 is the stronger attribution. It removes decoded-B STS, B LDSM, and decoded-B
publication barriers, and amortizes synchronization to one BAR per 64 HMMA,
but four row warps each reconstruct the same band. Its repeated BK64 body has
410 ALU + 130 SHF, or 102.5 + 32.5 per K16, and is 2.17x slower. Restoring
descriptors adds only 13.56%. At >=2 resident CTAs the binding issue stream is
the duplicated decode ALU, not shared-memory capacity or the B round trip.

The per-superstage V4 critical path is intentionally different:

```text
band 0:  load/decode B[next,D] ----------------> LDSM/HMMA B[current,D]
band 1:  LDSM/HMMA B[current,D] --------------> load/decode B[next,D]
          |<--- concurrent warp issue window -->|
end:      one band-local publication/reuse rendezvous
```

Payload publication adds one band-local rendezvous before each future decode.
Thus the steady body has two band-local barrier instructions per superstage:
one per 2 K16 for D=2 and one per 4 K16 for D=4, versus control's 3 per K16
and V3's one per 4 K16. The gate records emitted SASS rather than assuming the
compiler preserves the source count.

## Latency, issue, and occupancy math

Each K16 performs exactly the control's work: 2 bands x 64 x 16 = 2,048
codebook evaluations per CTA and 16 HMMA instructions per warp. V3 performs
8,192 evaluations per CTA/K16. V4 therefore executes 4,096 evaluations and 32
HMMA/warp per D=2 superstage, or 8,192 evaluations and 64 HMMA/warp per D=4
superstage; normalized decode is exactly 2,048 evaluations/K16.

Using the control SASS as the measured instruction proxy, a D-stage band decode
contains approximately `D*(27 ALU + 8 SHF + 8-10 STS)` per participating warp,
while its opposite band exposes `D*(16 HMMA + 8 B-LDSM + 2 A-LDSM)` plus pointer
work. D=2 exposes 32 HMMAs/warp; D=4 exposes 64. The same-warp dependency is not
claimed to disappear. Instead, four decode warps and four tensor-consumer warps
have independent PCs, so Blackwell's warp schedulers can fill tensor-pipe issue
slots while the opposite band issues integer/shift/store instructions. D=4
doubles that continuous issue window and cuts steady barrier density from
1/K16 to 0.5/K16. If D=4 does not improve over D=2, the measured limit is not
insufficient ring distance; it is remaining decode issue demand, payload
publication, or producer/consumer load imbalance.

The accumulator seam remains `Acc16[2][8]` or `Acc32[2][8]`, fully unrolled and
compile-time indexed. Ring depth changes shared addresses only; no B fragment
array or stage-indexed accumulator array is live. The resource forecast is:

| live category/thread | FP16 forecast | FP32 forecast |
| --- | ---: | ---: |
| accumulators | 32 regs | 64 regs |
| two A fragments + streamed B | 10 | 10 |
| two fixed decode mappings/transient | 8-12 | 8-12 |
| pointers, ring state, compiler state | 38-48 | 43-50 |
| **forecast total** | **88-102** | **121-128** |

FP32 uses `__launch_bounds__(256,2)` to prevent a >128-register launch shape;
FP16 uses the stricter `__maxnreg__(104)` contract. Both must compile without
spills. The frozen gate is measured FP16 <=104, FP32 <=128, and zero
stack/local/spill.
At the ceilings, two CTAs require 53,248 or 65,536 registers respectively.

## Exact shared layout and ownership

The harness launches both variants with the official 45,056-byte class so the
occupancy comparison is controlled. The D=4 variant genuinely addresses every
byte:

| region | D=4 bytes | layout / lifetime |
| --- | ---: | --- |
| decoded-B ring | 32,768 | `[bank=2][stage=4][band=2][col=64][k=16]`; current and next BK64 |
| warp-private A stage | 8,192 | `[warp=8][row=32][k=16]`; LDSM then overwrite with next K16 via async copy |
| payload ring | 4,096 | `[bank=2][stage=4][band=2][tile=4][word=16]` |
| **total** | **45,056** | **44.0 KiB/CTA; 90,112 B for two CTAs** |

D=2 uses the identical allocation/occupancy class but addresses two stages per
bank (16,384 B decoded B), one warp-private A stage (8,192 B), and two stages
per payload bank (2,048 B). Keeping the launch allocation fixed prevents a
shallower variant from winning merely through higher shared-memory residency.

Within a band, `band_tid = row_group*32 + lane` maps to `r=band_tid&15` and
two fixed `ccl` values `(band_tid>>4)` and `+8`. Four unrolled 16-column tiles
give eight unique weights/thread: 128 threads x 8 = 1,024 values/band/K16.
The K2 `(word, previous_word, shift)` mapping is computed once before the K
loop and uses V3's descriptor-free fixed payload ordering. Stores publish the
control-compatible `[n][k]` layout, and every row warp subsequently streams the
same bytes with `ldmatrix`.

The single warp-private A slot is safe because both A fragments are in registers
before its owner warp issues the async overwrite for K16+1. No other warp reads
that slot. B and payload banks are not overwritten until their owning band has
finished the prior superstage and crossed its end rendezvous. Bands never share
those regions, so no CTA-wide rendezvous is required in steady state.

## Frozen gate and protocol

`run_slice1_v4.py` compiles both D=2 and D=4 FP16/FP32 symbols for sm_120a,
runs the freshly compiled shared-B control and both candidates with two warmups
per timing call and five alternating pairs, and bit-compares each FP16 output
against all 35,651,584 control outputs. It retains the overlay manifest, binary,
ptxas diagnostics, resource dump, SASS, samples, and generated gate reports.

| row | threshold |
| --- | --- |
| correctness | byte-exact representation reverse and 0 / 35,651,584 mismatches for each depth |
| representation | <=25% growth; no u16/fp16-per-weight stream |
| decode evaluations | 2,048/CTA/K16, i.e. 2,048D/CTA/superstage; not V3's 8,192/K16 |
| resources | FP16 <=104, FP32 <=128, stack/local/spills 0, >=2 CTAs/SM |
| direct op mechanism | each depth <1.591085 ms to beat shared-B |
| direct op breakthrough | each depth <=1.352422 ms for >=15% throughput |
| diagnostics | emitted HMMA and barrier counts per D-stage repeated superstage |

The report gives an independent verdict per row and depth. `<1.591085 ms`
means the mechanism is real; `<=1.352422 ms` is the breakthrough. A miss is
banked as evidence and reported without unilaterally closing Sean's packed path.
