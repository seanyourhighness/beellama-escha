# EXP-04 ESCHA-W2 PREFILL — Phase closure report (2026-09-01)

## Starting and ending commits
- Protected pre-EXP-04 checkpoint (remote before this phase): `4501b3ee1`
- Ending local == remote: **`ff002814c`** (19 commits past checkpoint)
- Branch `escha-w2-prefill`; worktree clean (untracked = `.escha-venv/` +
  `weights/escha` + `results/prompts.json` local symlinks only).

## Phase 0 — State reconciled
- Checkpoint `4501b3ee1` protected; rollback = revert guarded operator or
  build without the gate; local-aid files documented, never committed.
- Worker preflight (one DeepSeek V4 Flash via Nous Portal): **WORKER_BLOCKED,
  HTTP 401** — recorded once, no retries, primary-agent-only thereafter.

## Phase 1 — P-ARCH-20 contamination audit
- The FP16 fragment store (`acc16[i][j].x[l].x` over `l<tile_c::ne=4` when
  `tile_ah::ne=2`) was introduced at `f61c9cc2f`, fixed at `7b1880f41`.
- **P-ARCH-20's fp16 builds compiled and executed the faulty kernel** (SASS
  confirms `.F16`); correctness never validated. **The −44.22% result is
  INVALID** as evidence against fp16 MMA accumulation.
- EXP-02/03, Stage 1, Stage 2 control/candidate never executed the faulty
  path → conclusions stand. Corrections appended (amend, not rewrite) to the
  ARCH-01 audit + ledger row.

## Phase 2 — Stage 2 formal closure → **CONFIRMED + PROMOTED**
- Sol pre-authorized the noise protocol BEFORE data (`PROTOCOL=APPROVED`):
  9 matched ABBA pairs, primary CV≤2%, fallback paired-log rule.
- Noise run (frozen binaries, GPU 2887 MHz const): control CV 3.24% /
  candidate CV 2.15% (primary not met on this host); **fallback PASS:
  G=1.0973, 95% CI [1.0774, 1.1175], 9/9**; median **+9.31%**
  (2355.9 vs 2155.2 tok/s).
- Reconfirmed: route 800/800 0-mismatch; SASS `.F16`/`.F32` split; REG
  97/128, no spills, smem unchanged; P2/P7 16/16; decode +1.63% (≤2%); no
  family regressions; binaries unchanged since `7b1880f41`.
- **Sol VERIFY: VERDICT=CONFIRM, Stage 2 gate PASS, SMALLER POSITIVE.**
  **Stage 2 (mixed acc, IC≤6144 fp16) promoted as default prefill control**
  (`ace024e72`).

## Stage 3 — bounded-K FP16 for 17408→5120 → **REJECTED + REVERTED**
- Attribution: 23.3% of matmul (~209 ms wall); projected +4.4/+5.4/+6.4%.
- Sol PLAN READY → implemented (`03f648a3b`, `12110c78a`) → Sol code review
  CONFIRM. Route 128/128 `mma-fp16-boundedk`; SASS fp16-only REG 97 no
  spills; numerical rel-RMS 1.08e-3 (benign, finite); P2/P7 16/16; decode
  −1.31%.
- **Matched noise protocol: median +2.76%, paired-log G=1.0272
  [1.0005,1.0546], 8/9 → below the ≥5% gate → REJECT.** 4-slice FP32 twin
  was −4% (split-K + 4× partial/finalize traffic); FP16 recovered +8.7% at
  the same topology; net <5%.
- **Guarded operator reverted** (`da1bf72b9`, `3e3c5221d`); promoted Stage 2
  retained. **Sol VERIFY: VERDICT=CONFIRM, Stage 3 gate PASS (reject+revert)**.
- Next variable per Stage 1 profile (finalize 6.7% > rotate 4.6%): **fuse
  output rotation/scale into the GEMM epilogue** — **Sol PLAN READY**
  (warp-owned `_fusedfin`, n_slices==1 only, +8 KiB staging, tags
  `mma-fp16/32-fusedfin`). Implementation pending (Sol code review first).

## Resource usage (frozen binaries, cuobjdump per-symbol)
| kernel | REG | STACK | LOCAL | static SHARED | dynamic SHARED |
| --- | ---: | ---: | ---: | ---: | ---: |
| control/candidate fp32 K2/K3 | 128 | 0 | 0 | 1024 | 13824 |
| candidate fp16 K2/K3 | 97 | 0 | 0 | 1024 | 13824 |
| bounded-K fp16 K3 (reverted) | 97 | 0 | 0 | 1024 | 13824 |

No spills, no local/stack anywhere.

## Correctness / parity / decode
- P2/P7: 16/16 on control, Stage 2 candidate, bounded-K candidate.
- Decode: Stage 2 +1.63%, bounded-K −1.31% (both within ≤2%).
- Numerical: Stage 2 fp16 rel-RMS within contract; bounded-K 1.08e-3; all
  outputs finite, 0 NaN/Inf.

## SHA equality / clean worktree
- **local == remote `ff002814c`** (verified via `git ls-remote`).
- Worktree clean (only documented local aids untracked).
- GBrain ledger page synced and verified retrievable; wiki committed.

## Standing recommendation
Promoted Stage 2 (+9.31% median, Sol-verified) is the current default prefill
control. The next structural variable (fused GEMM finalize) has a Sol-approved
plan with a credible 5.0–6.7% window; implement only after the Sol code review
of the warp-owned design (the plan's own stop condition if the P-ARCH-14
barrier cost cannot be shown removed).
