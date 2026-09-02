# EXP-09 — Terra Math & Design Audit (Gate 1 approval)

Date: 2026-09-02 · Auditor: Terra (Hermes) · Plan under audit:
`REWRITE-PLAN.md` (Sol Gate 1, 24,107 B). Control = promoted Stage 2 at
`1c193ad4c`.

## Independent verification (not taken from the plan)

Verified against `ggml/src/ggml-cuda/escha-moe.cu` control kernel
`escha_matmul_dense_tiled_mma` (line 958) by direct arithmetic:

| claim in plan | independent check | result |
|---|---|---|
| Control warp footprint 32 rows × 64 cols (MT=2, NTT=8) | NW=8, WN=2, WM=4, MT=BM/16/WM=2, NTT=BN/8/WN=8 → 16*2=32 rows, 8*8=64 cols | **CONFIRMED** |
| Candidate per-warp footprint = control footprint | band=warp&1 (≡ control wn), row_group=warp>>1 (≡ control wm): identical 32×64 ownership, so partial-store coordinates map 1:1 to control | **CONFIRMED** |
| acc fp16 = 32 regs, fp32 = 64 regs | 2*8 tiles × tile_ah::ne=2 (fp16) / tile_c::ne=4 (fp32) | **CONFIRMED** |
| Control holds B[NTT=8] = 16 half2 regs live | kernel declares `tile_b B[NTT]` (line 1117), all 8 filled before MMA (lines 1122-1125) | **CONFIRMED** |
| Forecast 89/120 vs ceilings 97/128 | 97−16(B array)+2(streamed B)+temps ≈ 89; same shape for fp32 | **PLAUSIBLE — measured gate decisive** |
| Control = 3 CTA barriers per K tile | 3 `__syncthreads` in the K-tile body (payload/A publish, B-decode publish, post-MMA read-protect) | **CONFIRMED** |
| Candidate = 2 barriers (no decode-publish) | decoded B never enters smem → only payload/A publish + read-protect remain | **CONFIRMED by design** |
| smem 9,728 B vs control 13,824 | 13,824 − s_w[BN][16] 4,096 B = 9,728 | **CONFIRMED** |
| Decode duplication: 4 row-warps per band decode same B | ownership math forces it; same trade official `escham_code_gemm` makes (warp-local decode, 0 STS.U16, 80 regs) | **ACCEPTED as designed trade** |
| 16 MMA/warp/K-tile logical work unchanged | MT*NTT = 2*8 = 16 m16n8k16 | **CONFIRMED** |

## Review flags (Sol §9) — decisions

1. **Duplication trade (4 row-warps decode same band): APPROVED.** It is what
   bounds ownership to 32×64 and avoids EXP-07's full-tile register lifetimes
   (124–154 regs). It mirrors the official structure that demonstrated 80 regs
   and 0 STS.U16. Cross-warp B sharing is explicitly out of scope.
2. **Direct fragment construction primary, ldmatrix only as oracle:
   APPROVED**, conditioned on the mandated K2/K3 fragment oracle gate before
   any model parity run. Fragment order (funnel operands, ring predecessor,
   half2 lane mapping) is the top correctness risk and must be proven by
   capture, not assumed.
3. **Register forecast hard-gate treatment: CONFIRMED.** 89/120 are planning
   estimates. Measured ≤97 fp16 / ≤128 fp32 with zero spills and no
   STACK/LOCAL is the pass bar. Do not solve pressure with `maxrregcount`.

## Non-negotiables (must survive implementation)

- Partial→finalize contract unchanged; `escha_finalize_dense` untouched.
- FP16 store seam (tile_ah x[0].{x,y}→l=0,1; x[1].{x,y}→l=2,3) copied
  literally from control lines 1153-1192.
- Split-K supported (n_slices>1), no silent single-slice restriction, no
  occupancy/slice-policy change, no K-tile reorder.
- Mixed-acc threshold IC≤6144 frozen; route tags `mma-fp16-mlw` /
  `mma-fp32-mlw` only when the candidate actually launches.
- smem exactly 9,728 B; s_w must not survive under another name.
- Default path byte-identical with macro absent; rollback = delete guarded
  block + candidate build dir.

## Verdict

**PLAN=READY.** Authorization to implement the macro-guarded candidate per
§6, then run §7 gates in order: build → resources/SASS → route proof + smoke
→ fragment oracle + numerical/tails/split-K → P2/P7 + decode → matched 9-pair
ABBA campaign. Any resource-gate failure or >5% family regression stops the
experiment before the final campaign. Sol code review of the diff is required
before timing. Evidence stays under `evidence/EXP-09-mainloop-rewrite/2026-09-02/`.
