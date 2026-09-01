# EXP-06 — Down-projection BM64-equivalent tile geometry
## Phase 0 state and change control — 2026-09-01

## Starting point
- Branch: `escha-w2-prefill`
- Starting SHA: `4bc1afc1ddb57f1210cba5e52630dc3d57e57d90`
- Remote `github/escha-w2-prefill`: same SHA (verified 2026-09-01)
- Merge base with remote: same SHA
- Local-only commits: 0; remote-only commits: 0
- Protected ARCH-01 rollback SHA: `4501b3ee10dfce3451df7442c7baa2b03021a105`
- Promoted implementation control: EXP-04 Stage 2 mixed accumulator (`ace024e72`)
- Stage 3 implementation SHA: `03f648a3b` (present in ancestry, reverted; no current bounded-K symbols)

## Current routing
`ggml/src/ggml-cuda/escha-moe.cu` selects:
- `mma-fp16-mixedacc` for `IC <= 6144`
- `mma-fp32-mixedacc` for `IC > 6144`

The EXP-06 target is only the long-IC down projection: K3, `IC=17408`,
`OC=5120`, prefill/non-generation MMA path, and only row ranges approved by
Sol Gate 1.

## Scope lock
Exactly one structural variable may change: the BeeLlama tile/CTA dimension
proven equivalent to official `ESCHAM_GEMM_BM=64` for this target. It must not
change accumulator policy, input/output transforms, packed-code format/decoding,
split-K policy except mandatory indexing mechanics, or any other family.

## Local aids excluded from git
At reconciliation, `weights/`, `results/`, and `.escha-venv/` exist locally
and remain untracked/ignored execution aids. No weights, GGUFs, venvs, binaries,
or symlink targets may be committed.

## Rollback procedure
1. The experiment-off route must compile to the current promoted Stage 2 behavior.
2. If any decisive gate fails, `git revert <EXP-06 implementation SHA>` (or
   restore the target hunk exactly to starting SHA `4bc1afc1d`), retain only
   evidence/docs commits, and prove `git diff 4bc1afc1d -- ggml/src/ggml-cuda/escha-moe.cu`
   is empty.
3. The protected recovery point remains `4501b3ee1`; normal EXP-06 rejection
   restores the promoted Stage 2 control at `4bc1afc1d`, not ARCH-01.

## Gate sequence
Phase 1 accounting/BM semantics/revalidation → Sol `PLAN=READY` → isolated
implementation commit → Sol implementation `VERDICT=CONFIRM` → build/SASS/
route/numerical proof → benchmark gates → Sol `VERIFY=CONFIRM` → promotion or
revert.
