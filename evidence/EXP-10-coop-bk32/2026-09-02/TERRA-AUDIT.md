# EXP-10 — Terra Math & Design Audit (Gate 1 approval)

Date: 2026-09-02 · Auditor: Terra (Hermes) · Plan: `REWRITE-PLAN.md` (Sol,
Gate 1, 29,971 B). Control: promoted Stage 2 at `969db62df`.

## Independent verification (not taken from the plan)

| claim | independent check | result |
|---|---|---|
| Control smem 13,824 B = s_pay 1,536 + s_u 8,192 + s_w 4,096 | direct arithmetic | **CONFIRMED** |
| BK32 = 17,920 B (2-slot B), BK64 = 26,112 B (4-slot) | direct arithmetic | **CONFIRMED** |
| 2-CTA residency at 128 regs = 65,536 exactly | 128×256×2 | **CONFIRMED** |
| smem does not displace 2-CTA occupancy | BK64 worst = 54,272 B/2-CTA = 23.2% of 228 KiB | **CONFIRMED** — regs are the binding resource, exactly as claimed |
| Control = 3 CTA barriers per K tile | 3 `__syncthreads` in control kernel K-tile region | **CONFIRMED** |
| BK32 = 2 barriers/tile (P + D) | P(n+1) publishes payload[n+1]/A[n+1] + frees A[n]; D(n+1) publishes B[n+1] + frees B[n]/payload | **CONFIRMED by schedule proof** |
| Decode coverage: 256 × 8 = 2048 = 16×128 B tile | direct | **CONFIRMED** |
| No row-warp decode duplication | decode formula uses only `tid%16`/`tid/16` + k, no `wm/wn` | **CONFIRMED** (differs from EXP-09's warp-local build) |
| B regs 16 → 2 (streamed one `tile_b`) | control `tile_b B[NTT=8]` = 16 regs; candidate one `B` at a time | **CONFIRMED** |
| Bit-compat argument | same decode formula, same ldmatrix mapping, same K-tile order per accumulator, j-outer vs i-outer changes no accumulator's contribution sequence | **CONFIRMED** |

## Audit of the critical schedule (P/D proof read carefully)

- **A-ring**: A[n] loaded before P(n+1); A[n+2] issued into the recycled
  A[n&1] slot only after P(n+1) — no ldmatrix race. Relative `(ti-lo)&1`
  phases handle T=1 and uneven split-K slices. Sound.
- **B-ring**: B[n+1] written to `s_w[(n+1)&1]` while MMA(n) reads
  `s_w[n&1]` — different slots, no WAR race even with even/odd warp
  reordering. D(n+1) gates B-slot reuse for B[n+2]. Sound.
- **Payload**: single `s_pay`; D(n) proves all decode readers of payload n
  finished before payload n+1 is published at the top of the next iteration;
  P(n+1) proves publication visible before decode B[n+1] begins. Sound.
- **Even/odd warp parity** is a scheduling hint, not a correctness partition:
  all 8 warps decode AND consume every tile. Correctness does not depend on
  scheduler behavior — only the performance hypothesis does. This is the right
  way to expose decode||MMA without EXP-08's ownership serialization.
- **Last tile**: decoded by iteration T-2's D(T-1), consumed in epilogue with
  no next decode. No final reuse barrier needed. Sound.

## Review flags (Sol §11) — decisions

1. **BK32 over BK64: APPROVED.** BK64 (4-slot) removes no barrier (all-256
   decode still needs full-CTA publication), wastes 8 KiB/CTA if shallow, and
   +3–7 regs if deep → breaks ≤104/≤128. BK32 is the only admissible stage.
2. **Two full-CTA barriers/tile (2T vs 3T): APPROVED.** Named/subset barrier
   cannot publish a full B tile (all warps contribute). P/D phase proof holds.
3. **Symmetric warp phasing (no dedicated producer warps): APPROVED.** Any
   warp removed from MMA leaves its 32×64 ownership uncovered or forces a
   replay design. Out of scope.
4. **Streamed ldmatrix B with j-outer MMA: APPROVED** — it is fragment/order
   preserving, NOT a direct-fragment rewrite (which EXP-09 proved fatal).
5. **A-ring seam: CONFIRMED** per the proof above.
6. **Register forecast 97–104 FP16 / ≤128 FP32: CONFIRMED as the only
   admissible range.** Measured ptxas is decisive; no forced cap, no spills.
   ≤97 preferred; 98–104 requires occupancy+SASS review before timing.
7. **Payload timing: CONFIRMED** by the D-before-overwrite / P-before-decode
   proof.

## Non-negotiables (must survive implementation)

- Decode formula, payload publication, ldmatrix B mapping, store seams,
  split-K lo/hi, mixed-acc IC≤6144, separate `escha_finalize_dense` — all
  literal from control.
- Host smem exactly 17,920 B; same grid/block; tags `mma-fp16-bk32` /
  `mma-fp32-bk32` only when candidate launches; macro-off path byte-identical.
- No dedicated producer warps, no direct-fragment construction, no finalize
  fusion, no tile-aspect change, no BK64, no `maxrregcount`, no spill.

## Verdict

**PLAN=READY.** Authorization to implement the macro-guarded candidate per
§8, then run §9 gates in order: builds → resources/SASS (≤104 fp16 / ≤128
fp32, smem 17,920 B, 2 barriers/tile vs 3, no spills, LDGSTS retained) →
fragment oracle (bit-exact vs control) → route 800/800 + family gate (K2
5120→17408 matmul ≥10% faster, no family >5% regress) → tails/split-K/
sanitizer → P2/P7 + decode → matched 9-pair campaign (≥5% full 2K, ≥4/5
samples, no depth >5% regress) → Sol final verdict. Any resource or
correctness failure stops before timing; <2% full gain = REJECT + revert +
stop incremental kernel work per Decision B.
