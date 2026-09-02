# EXP-07/EXP-08 Series Status (2026-09-01/02)

Series: ESCHA-W2 PREFILL post-BASE-01 optimization series targeting LowGPU
prefill parity. Sol gates via Codex. Branch escha-w2-prefill.

## EXP-07 — register-decoded B mainloop: REJECTED (pre-timing)
- v1 (all-rows MT=8): compiled, removed shared-B STS.U16, but 124/154 regs vs
  control 97/128 -> occupancy loss. Rejected pre-benchmark.
- v2 (Sol VERDICT=b revision): disjoint 8-col fragments MT=8/NTT=1, two 64-col
  passes; STS.U16=0, no spills, but 125/141 regs vs 97/128; fp32 HMMA 24 vs 32.
  Rejected pre-benchmark. Candidate reverted.
- Conclusion: shared-B elimination costs more registers than it saves at this
  geometry on SM120.

## EXP-08 — fused-finalize (warp-owned, n_slices==1): REJECTED (matched campaign)
- Sol Gate 1 PLAN=READY (3 rounds); Sol code review CONFIRM (3 rounds: transpose
  fix, barrier fix).
- Pre-timing gates PASSED: regs 96/128, no spills, smem 22,016 B, route 736
  fused + 64 split-K 0 fallback, P2/P7 16/16 parity, decode +0.75%.
- Canonical 9-pair campaign: control 2319.22 vs candidate 1895.11 tok/s,
  geometric latency ratio 1.2129 CI[1.1830,1.2436], 0/9 candidate-faster,
  **-18.29% tok/s** -> REJECT (Sol REJECTION=CONFIRM). Candidate reverted.
- Interpretation: warp-pair batch epilogue serializes (8 batches x 2 named
  barriers, 2-warp owners) vs full-CTA separate finalize. Retry only with a
  full-CTA-parallel epilogue design.

## Series outcome
Both T1 (mainloop register-B) and T2 (finalize fusion) are negative at this
geometry. This strengthens BASE-01's classification: the packed-GEMM mainloop
deficit is structural and not recoverable by the two highest-ranked
single-variable kernel changes tried so far (shared-B removal, finalize fusion).
Remaining levers ranked: (T3) fused input rotation ~4.6% (below 5% rule unless
combined); (T4) shape-specific ffn_down decode structure; mainloop redesign
re-planning with a genuinely different B-decode or occupancy structure (per
EXP-05 official 80-reg/45 KiB structure). Promoted Stage 2 remains the default.

## Commits
- 50b8f0bf2 EXP-07 reject + revert records
- 8a87e3d40 EXP-08 implementation (guarded) + pre-timing gates
- (next) EXP-08 rejection records; docs/ledger updates
