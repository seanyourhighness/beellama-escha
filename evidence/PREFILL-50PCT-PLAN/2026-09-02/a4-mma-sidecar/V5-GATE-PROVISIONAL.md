# V5-PIPE warp-specialized producer/consumer — measured gate (provisional)

Date: 2026-09-02 · Scope: blk.0 ffn_gate K2 5120→17408, M=2048, RTX 5090
sm_120a. Measured by Terra from Sol's written plan + harness (Sol hit Codex
usage cap before run; reset 6:54 PM, formal Sol review + FINAL GATE pending).

## Architecture (Sol V5-PLAN.md)

12-warp CTA = **4 producer warps (decode-only) + 8 consumer warps (MMA-only)**,
1:2 ratio; full 128×128 ownership by 8 consumers × 16 HMMA/K16; CTA-cooperative
shared A (4,096 B/K16, **−50% vs V4**); two-slot B ring; six named barriers
(ready/free ×2 slots, payload, A-ready) = split arrive/wait handoff; 17,408 B
shared/CTA; fixed descriptor-free V3 mapping.

## Measured (5 alternating pairs, warmups, median; fresh shared-B control)

| row | measured | threshold | result |
|---|---:|---:|---|
| correctness | 0 / 35,651,584 bit mismatches | 0 | PASS |
| representation | V3 fixed mapping, 0.001147% | ≤25%, byte-exact | PASS |
| consumer coverage | 8 consumers × 16 HMMA/K16 = full 128×128 | full tile | PASS |
| producer ownership | producers decode-only (no MMA executed) | zero-MMA producers | PASS |
| resources FP16 | **73 regs**, 0 local, 0 spills, 2 CTA/SM | ≤104 (12-warp math ≤85), ≥2 | PASS |
| resources FP32 | **80 regs**, 24 local B, 0 spills, 2 CTA/SM | ≤128, 0 local | PASS (local=24B flagged) |
| shared/CTA | 17,408 B (34,816/2 CTA) | — | PASS |
| A traffic | 4,096 B/K16 (−50% vs V4) | −50% | PASS |
| **direct op: beat shared-B** | **2.043904 ms vs 1.591354 ms = 1.28× slower** | <1.591085 ms | **FAIL** |
| direct op: breakthrough | 2.043904 ms | ≤1.352422 ms | FAIL |

Candidate samples: 2.020505, 2.028525, 2.043904, 2.051366, 2.058477.
Control samples: 1.570714, 1.587949, 1.591354, 1.592070, 1.601798.

## Series trend (direct-op vs shared-B control)

V1 8.85× → V2 1.51× → V3 2.17× → V4 2.38× → **V5 1.28×**

V5 is the closest of the series: producer/consumer specialization + −50% A
traffic cut the gap from 1.51× (V2) to 1.28×, with excellent resources (73
fp16 regs, 0 spills). But it still does not beat the shared-B control, and the
frozen mechanism gate (<1.591085 ms) is not met. Remaining gap attribution
candidates (for Sol's formal review): split-barrier handoff overhead per BK32
superstage, the decoded-B STS→LDSM round trip (V2-class cost, now hidden only
2-deep), FP32 24 B local traffic, and consumer staging serialization.

## Pending

Sol FINAL GATE (Sean directive) at cap reset 6:54 PM: CONTINUE (iterate the
winning producer/consumer toward <1.591 then ≤1.352) or FINAL-CONFIRM-REJECT
(close the seven-experiment set, deliver hybrid P-ARCH-23 8.599GB/3023 or 23G
8.808GB/3228). Evidence: v5-raw/ (benchmark.json, harness.sass,
cuobjdump-resources.txt, repack-manifest.json), slice1_v5_harness.cu,
V5-PLAN.md.
