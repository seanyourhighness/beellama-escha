# EXP-11 Attempt 2 — byte-equality IS reproducible: pairwise-tree evidence

Date: 2026-09-02 · Follow-up to ROOT-CAUSE-NONPORTABLE.md

## New evidence (contradicts "non-portable" conclusion)

Probing NumPy's actual float32 summation tree on the exact sizes the Q2_K
oracle uses:

- **n=256**: `np.sum(x, dtype=f32)` == recursive split-halves pairwise
  exactly (−31.969685). Reproducible in C++ with a balanced binary tree.
- **n=16**: `np.sum(x, dtype=f32)` == 4×4 pairwise tree AND == SIMD-8-lane
  accumulate-then-combine tree (−2.6911025), NOT sequential (−2.691102) and
  NOT 8+8-naive. A bounded 4/8-lane tree reproduces NumPy exactly.

Conclusion: the divergence is NOT "NumPy arithmetic cannot be ported" — it is
"the C++ port used sequential accumulation instead of NumPy's pairwise tree."
For the fixed reduction widths used by the oracle (16 and 256), the pairwise
tree is bounded and reproducible. **Byte-equality is achievable** by changing
the native `make_qkx3`/`make_qp`/`quantize_block` accumulations (and sigma2)
to the matching pairwise trees, plus verifying `np.rint` (ties-to-even,
already matched) and `np.clip`/`np.minimum`/`np.divide` ufunc semantics.

## Revised options

1. **Bounded fix (preferred, no contract change)**: reimplement the NumPy
   summation order in C++ for fixed 16/256 widths; retest microbench + layer-0
   SHAs (ea4cb733… / 041c0045… / e5a94a62…). If byte-equal → Attempt 2
   proceeds as designed. NOTE: the oracle is NumPy-version-sensitive, so the
   recipe must pin the exact NumPy version whose tree the C++ matches (report
   used 2.5.1 in one path, 2.4.3 in another — resolve this first).
2. Re-scope correctness (needs Sean): accept valid-GGML + MAE + per-platform
   determinism instead of byte-equality.
3. Literal rule → Attempt 3.

Evidence: this file; pairwise probes in session history.
