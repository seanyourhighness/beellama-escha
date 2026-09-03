# V5-PIPE — warp-specialized producer/consumer plan

Date: 2026-09-02  
Scope: standalone `blk.0.ffn_gate`, K2 5120 -> 17408, direct-op M=2048. No runtime, loader, `ggml/`, or prior sidecar file changes.

## Architecture decision

Use a **12-warp CTA: 4 producer warps + 8 consumer warps (1:2)**. The CTA
continues to own one 128x128 output tile. Producer warps decode only and write a
two-slot shared-B ring; consumer warps stage A and execute only LDSM/HMMA. This
chooses the directive's wider-CTA option rather than taking two MMA owners away
from the existing eight-warp geometry.

An eight-warp 2P:6C alternative would preserve the number of CTAs but would
force two consumers to own extra accumulator tiles. That makes the FP32
accumulator floor at least 85.3 registers/thread before A/B fragments, pointers,
and output state, so the <=128-register resource gate becomes fragile and the
ownership becomes asymmetric. An eight-warp 4P:4C alternative doubles every
consumer's accumulator state and is impossible for spill-free FP32. A 16-warp
8P:8C CTA preserves consumer ownership but two CTAs would have a 64-register
per-thread allocation ceiling, exactly the FP32 accumulator footprint before
any fragments. The 12-warp 4P:8C point is the only symmetric full-coverage
choice worth testing.

The cost of 12 warps is a stricter measured residency constraint. On the RTX
5090's 65,536-register SM pool, two 384-thread CTAs require:

```text
R <= floor(65,536 / (384 * 2)) = 85 registers/thread
threads = 384 * 2 = 768 threads/SM
shared = 17,408 * 2 = 34,816 bytes/SM
```

Thus the practical V5 ceiling is 85 registers for both symbols, stricter than
the frozen <=104 FP16 / <=128 FP32 limits. The consumer body streams one A and
one B fragment at a time around the fixed accumulator seam. FP16 has 32
accumulator registers; FP32 has 64. The forecast is 58-72 FP16 and 78-85 FP32.
The producer branch carries no accumulator or fragment arrays. Compilation and
runtime occupancy, rather than this forecast, decide the resource row.

## Exact ownership and coverage

Producer warps are `warp 0..3` (128 threads). For each K16 stage the producer
collective loads the two bands' 128 canonical payload words once, then each
producer thread reconstructs 16 values: two fixed `ccl` mappings x four
16-column tiles x two bands. This is exactly:

```text
128 producer threads * 16 values = 2,048 decoded fp16 values / CTA / K16
```

No producer path contains LDSM or MMA. The fixed mapping and V3 payload order
remain unchanged and descriptor-free.

Consumer warps are `warp 4..11`; their local index `cw=warp-4` maps to
`row_group=cw>>1` and `band=cw&1`. Each consumer owns 32 rows x 64 columns,
represented by 2 m16 x 8 n8 accumulator tiles. Per K16:

```text
16 HMMA / consumer warp * 8 consumers = 128 HMMA
8 disjoint (32x64) owners = 128x128 complete output tile
```

There is no nominal or executed MMA-coverage reduction versus the shared-B
control. The widened CTA adds producers instead of repurposing consumer warps.

## CTA-cooperative A staging

The eight consumer warps cooperatively copy one `[128][16]` fp16 A tile into a
shared double buffer. Consumer thread `ctid=0..255` copies one 16-byte vector,
so the CTA issues exactly 4,096 A bytes/K16. Both column bands load their A
fragments from that single shared copy. A consumer-only named barrier publishes
the copy before LDSM.

V4 staged `[warp=8][row=32][k=16]`, or 8,192 A bytes/K16. V5 therefore removes
4,096 bytes/K16 (**-50%**) and 1,310,720 bytes/CTA across 320 K16 stages. The
shared A footprint remains 8,192 bytes only because it is double-buffered;
traffic, not capacity, is the diagnostic.

## B ring and split arrive/wait handoff

Depth stays two; V5 changes ownership, not depth. Shared layout:

| region | bytes | layout |
| --- | ---: | --- |
| decoded-B ring | 8,192 | `[slot=2][n=128][k=16]` |
| shared A ring | 8,192 | `[slot=2][m=128][k=16]` |
| payload ring | 1,024 | `[slot=2][band=2][word=64]` |
| **total** | **17,408** | 34,816 bytes for two CTAs |

Six named barriers implement independent role progress:

- ready[0..1], participant count 384: producer threads execute nonblocking
  `arrive` after decoded-B STS; consumer threads execute `wait` (`bar.sync`)
  before B LDSM.
- free[0..1], participant count 384: consumer threads execute nonblocking
  `arrive` after the stage's final HMMA; producers execute `wait` before
  reusing that B/payload slot.
- payload, participant count 128: producers rendezvous after the cooperative
  global-to-shared payload load and before decode.
- A-ready, participant count 256: consumers rendezvous after the single
  cooperative A copy and before A LDSM.

In a steady BK32 superstage (two K16 stages), the B handoff has two ready
arrive/wait pairs and two free arrive/wait pairs: **4 split handoff phases / 8
role endpoints per superstage**. The role-local payload and A publication each
occur twice per superstage. Producers can move to the other ring slot after
ready-arrive; consumers can stage A while producers decode, and can move to the
other A slot after free-arrive. This is the mechanism intended to hide the
same 4,096-byte decoded-B publication and LDSM round trip that V2 exposed on a
single collective critical path.

## Frozen gate and run protocol

`run_slice1_v5.py` will create a fresh V3-format overlay, compile a fresh
shared-B control plus V5 FP16/FP32 symbols for sm_120a, retain ptxas/resource/
SASS evidence, and run one correctness launch plus five alternating
control/candidate timing pairs with two warmups per timing call. Every timing
sample is the mean of five launches; medians decide the gate.

| row | threshold |
| --- | --- |
| correctness | 0 bit mismatches over 35,651,584 FP16-path outputs |
| representation | <=25% growth; byte-exact; V3 fixed mapping |
| consumer coverage | full 128x128 tile; 8 consumers x 16 HMMA/K16 |
| producer ownership | 4 producers; zero MMA instructions executed |
| resources | <=104/128 registers, zero stack/local/spills, >=2 CTAs/SM (12-warp register math additionally requires <=85) |
| barriers | split ready/free arrive/wait recorded per BK32 superstage |
| A traffic | 4,096 B/K16, -50% versus V4 |
| mechanism | median <1.591085 ms |
| breakthrough | median <=1.352422 ms |

If the mechanism row passes, the final verdict is `CONTINUE`. If it is
>=1.591085 ms, the final verdict is `FINAL-CONFIRM-REJECT`, the seven-experiment
set is closed, and delivery returns to the banked hybrid path (P-ARCH-23
8.599GB/3023 or 23G 8.808GB/3228).
