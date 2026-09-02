# EXP-11 Attempt 2 Slice 1 — byte-equality root cause: non-portable NumPy arithmetic

Date: 2026-09-02 · Root-cause by Terra (Sol usage-capped at time of diagnosis)

## Finding

The native C++ Q2_K port (`tools/escha_native_transcode/main.cpp`) is
algorithmically identical to the banked NumPy oracle
(`conversion/escha/transcode_oracle.py`) — same make_qkx3 37-trial structure,
same make_qp refinement, same packing. Yet byte-equality fails pervasively:

- 2048-value microbench (seed 42): 101/672 bytes differ, first diff at byte 2.
- 16,384-value probe (seed 7): **36/64 blocks (56%) differ**, 473/5376 bytes.
- Differing vs matching block spans overlap completely (13.9–19.0 vs
  14.4–22.8) — not an edge case or tie-breaking.

## Root cause

**NumPy pairwise summation vs C++ sequential float32 accumulation.** Even a
single 16-element `np.sum(w*x, dtype=np.float32)` differs from a sequential
float loop: NumPy `-31.24263` vs sequential `-31.2426376343` on the same
input. Every `np.sum(..., dtype=np.float32)` in the oracle (sum_w, sum_x,
sum_l, sum_l2, sum_xl, best_mad, mad over each trial; sigma2 over 256) uses
pairwise order, while the C++ port sums sequentially. The tiny divergences
propagate through the 37-trial scale/min search, changing which trial wins in
~56% of blocks — hence different d, dmin, and level bytes.

This is not a porting bug that a C++ correction can fix while remaining a
"native sequential C++ implementation": byte-equality would require
reproducing NumPy's exact pairwise summation tree (and np.rint/np.minimum/
np.divide ufunc rounding semantics) for every reduction, which is precisely
the "exact native arithmetic cannot be made portable" condition in
PROGRAM-PLAN §11 / revision §5 step 6.

## Program decision per the frozen rules

REVISION §5 step 6: "If the native path differs by one byte, stop before
timing/promotion and fix the implementation. If exact native arithmetic cannot
be made portable, PROGRAM-PLAN §11 requires rejection of Attempt 2 and pivot
to Attempt 3; approximate equality is forbidden."

Options that remain:
1. **Attempt 2 rejected → Attempt 3** (MMA-ready sidecar representation) per
   the literal rule — the NumPy oracle is not portable to native byte-equal
   C++ without re-implementing pairwise summation semantics, and the frozen
   contract forbids approximate equality.
2. **Re-scope the oracle contract**: redefine the cache correctness bar as
   "valid Q2_K + dequant-MAE ≤ oracle MAE + deterministic-per-platform" instead
   of byte-equality to the specific NumPy build. This WEAKENS a frozen gate and
   would require Sean's explicit decision (the entire cache-keying/validation
   design was built on byte equality). Note GGML itself does not require
   byte-identical quantization across builds — any valid Q2_K block with the
   correct layout loads and runs; the byte-equality bar was a verification
   convenience, not a runtime requirement.

Terra's read: option 1 follows the frozen rules literally; option 2 is the
engineering-sound path but changes the contract and needs Sean. Sol issues the
formal gate when available.
