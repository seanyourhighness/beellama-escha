# EXP-08 — fused GEMM finalize (rebased from EXP-04-nextvar READY plan; 2026-09-01)

## Pivot context
Selected as T2 after EXP-07 (register-B mainloop) failed its pre-timing
resource gate (124-154 regs vs 97/128 control; rejected + reverted, commit
50b8f0bf2). Sol verdict (b) mandated pivot to T2 immediately. The EXP-04-nextvar
plan below was Sol-PLAN-READY at `b12c6b963`; it is rebased onto the current
control: **HEAD `50b8f0bf2`** (promoted Stage 2 kernel, `ace024e72` = Stage-2
promotion; `4501b3ee1` = Stage-1 attribution only). BASE-01 canonical control:
ESCHA 2326.77 tok/s / 880.19 ms; Stage-1 attribution rotate 4.6% / matmul
88.6% / finalize 6.7%; finalize is ~10–15 ms wall -> credible **+2–4%**.

## Value framing (Sol REVISE item 2)
EXP-08 is an independent, lower-risk SMALLER-POSITIVE candidate. BASE-01 proved
the FFN packed-GEMM mainloop (not finalize) is 94.6% of the gap, so EXP-08
alone cannot reach any 3000+ target; it is worth one bounded run as +2–4% and
must NOT delay the mainloop redesign (T1). Classification:
- >=5% median wall = SMALLER POSITIVE, promote.
- +2–5% median wall = SMALLER-POSITIVE-2 (informative; decide promote/reject
  on resource/parity/regression gates and decode neutrality).
- <2% = REJECT + revert (informative negative).

## Control, bound, and ONE variable

Control is HEAD `50b8f0bf2` (promoted Stage 2, kernel unchanged since
`ace024e72`): FP16 MMA accumulation for `IC<=6144`, FP32 above, writes FP32
partials, then runs `escha_finalize_dense`. Stage 1 assigns finalize 6.7% of
projection time; BASE-01 puts finalize at ~10–15 ms wall (~+2–4% opportunity).

The sole variable is finalize placement. For Escha MMA prefill with `n_slices==1`, perform Hadamard-128, normalization, and `rout` scaling in the GEMM epilogue and write `dst` directly. Input rotation, decode, BM/BN, grid, A overlap, HMMA family, accumulator threshold, and generation do not change.

Implement a guarded, isolated `escha_matmul_dense_tiled_mma_fusedfin<K,BM,BN,FP16_ACC>` derived from the promoted kernel; do not reconnect `_ff` unchanged. Preserve dispatch exactly: `FP16_ACC=true` iff `IC<=6144`, for K2 and K3. For FP16, use Stage 2's corrected two-`half2` `tile_ah` mapping, convert each lane to FP32 at the logical partial-store seam, and never index it with `tile_c::ne`.

Preserve butterfly order `len=1,2,4,8,16,32,64`, then `rsqrtf(128)`, then FP16 `rout` converted to FP32. Run `len<=32` in-place within each 64-column owner warp using exact-order register/shuffle butterflies; use `16*BN*sizeof(float)` staging only for the `len=64` cross-half exchange and ordered store. Sol must verify lane ownership and pair-scoped/minimal synchronization.

- `n_slices==1`: no `p_buf` allocation/write; fused symbol writes `dst`; skip separate finalize.
- `n_slices>1`: unchanged Stage 2 kernel writes FP32 `p_buf[slice][row][OC]`; `escha_finalize_dense` sums slices in fixed order and finalizes. Never tag this fused.

P-ARCH-14 was neutral because its FP32-only `_ff` replayed the full Hadamard in CTA shared memory, adding 72 barriers/CTA. This design differs by retaining mixed accumulation, repairing the FP16 seam, removing the partial allocation/lifetime, and using a warp-owned epilogue with only the required cross-half exchange. A CTA-wide clone is not this experiment.

## Numeric contract

- FP32-acc families: `dst` bitwise identical to separate-finalize control, including butterfly order, normalization, `rout` multiply, and addressing.
- FP16-acc families: target bitwise identity to Stage 2 control. Against the FP32-acc reference, rel-RMS `<=1.5e-3` and no worse than the banked Stage 2 family value.
- Zero NaN/Inf in every family. P2 factual and P7 tool-call greedy gates: 16/16 in both arms.

## Resource, route, and SASS contract

Base MMA dynamic shared memory is 13,824 B at BM=BN=128. Fused staging adds exactly 8,192 B, totaling 22,016 B/CTA. Report compiler static/dynamic shared memory and calculated active CTAs/warps per SM beside control; the candidate may not cross a shared-memory occupancy boundary.

For isolated K2/K3 FP16/FP32 fused symbols: REG `<=128`, STACK/LOCAL 0, and no spill loads/stores. FP16 should retain headroom versus its 97-register control; FP32 may not exceed its 128-register control. Any spill or occupancy loss stops work before timing.

Tags are predicate-exact: `mma-fp16-fusedfin` for `IC<=6144 && n_slices==1`; `mma-fp32-fusedfin` for `IC>6144 && n_slices==1`. Split-K/noncandidate routes keep existing tags. Per-symbol SASS must show Stage 2's split: FP16 fused symbols contain `.F16` HMMA and no `.F32`; FP32 contain `.F32` and no `.F16`. HMMA counts/GEMM body must match the corresponding promoted specialization.

## Ordered gates

Use frozen binaries and the approved model, commands, shared prompt IDs, and 9-pair noise protocol. Never authorize fallback after observing data.

1. **Sol review before benchmarking:** verify single-variable isolation, fragment mapping/order, synchronization, `n_slices` dispatch, allocation lifetime, and metadata gating.
2. **Route/SASS/resources:** 800/800 profile records, zero predicate mismatch/fallback; isolated symbols and HMMA split; `<=128` registers, no spills or occupancy loss.
3. **Numerics:** all FP32 families bitwise identical; FP16 families within contract; zero NaN/Inf.
4. **Parity:** P2 16/16 and P7 16/16, both arms.
5. **Decode:** generation remains unfused; throughput regression `<=2%`.
6. **Families:** compare total projection time (`rotate+matmul+epilogue`, avoiding embedded-epilogue accounting bias) for every K2/K3 family; no regression `>5%`.
7. **Matched 2K:** nine adjacent pairs in frozen `AB BA BA AB AB BA BA AB AB` order. Primary: each arm CV `<=2%`. Otherwise use only the pre-approved paired-log fallback: candidate/control `G>=1.05`, 95% t-CI lower `>1.0`, candidate faster in `>=8/9`. Outcome bands (consistent with Value framing): median gain `>=20%` = BREAKTHROUGH; `>=5%,<20%` = SMALLER POSITIVE, promote; `+2%,<5%` = SMALLER-POSITIVE-2 (informative; decide promote/reject on resource/parity/regression/decode gates); `<2%` = REJECT+revert.

## Isolation, risks, rollback

Only metadata-gated Escha dense MMA prefill enters the candidate. Split-K stays separate-finalize; fusing its cross-CTA reduction would be another variable. Generation, input rotation, fallbacks, MOE, and standard Qwen/GGUF are untouched. Risks: fragment ownership, rounding-order drift, cross-warp races, register growth, and shared-memory occupancy.

Rollback removes/reverts only the guarded specialization and dispatch. The promoted Stage 2 mixed-acc default and ordinary FP32-partial/finalize path remain unchanged.

## Recommendation

Implement after Sol confirms the warp-owned design and static resource bound.
The credible window is +2–4% wall (finalize ~10–15 ms at 880 ms baseline), with
+5% as the strict SMALLER-POSITIVE promotion threshold and +2–5% classified
SMALLER-POSITIVE-2. EXP-08 is independent and lower-risk but must not delay the
mainloop redesign (EXP-07 negative; T1 re-planning is the primary path to
3000+). Rotate fusion has only a 4.6% ceiling; consider it only under a lower
gate. Consider MLP up+gate activation sharing only after independent
attribution shows >=5%.

NEXTVAR PLAN READY
