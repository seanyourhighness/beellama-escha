# EXP-11 Attempt 2 Slice 1 — native byte-equality GATE FAILURE

Date: 2026-09-02 · Slice-1 gate: "native path differs by one byte = stop"
(revision `REVISION-AND-SLICE1-PLAN.md` §5 step 6).

## Result

The Sol-written native transcoder (`tools/escha_native_transcode/main.cpp`,
400 lines) **builds cleanly** and runs, but **FAILS the frozen byte-equality
gate**:

- Layer-0 `ffn_gate` prepare: expected oracle payload SHA `ea4cb733…`, native
  produced `2068eca7…` → tool correctly STOPPED (fail-closed behavior works).
- Raw quantizer microbench (2048 values, seed 42): native vs NumPy oracle
  `quantize_q2_k` → **101 of 672 bytes differ**, first divergence at byte 2
  (early in block 0), i.e. a systematic Q2_K algorithm mismatch, not rounding
  noise.

## Interpretation

Reconstruction reaching the payload stage and hashing means the GGUF read /
reconstruct / header / offset plumbing is likely sound; the divergence is in
the native Q2_K port (scale/min search, the 37 `make_qkx3_quants` trials, or
`make_qp_quants`/packing order) vs the banked NumPy oracle that the converter
byte-gate already proved. Per the revision's own rule, approximate equality is
forbidden — this slice cannot pass without a byte-exact native port.

## Options per the program (PROGRAM-PLAN §11 / revision §5)

1. **Fix within Attempt 2**: Sol diagnoses the exact native-vs-NumPy Q2_K
   divergence and corrects the C++ to byte-equality (if it is a porting bug,
   in-scope; does not consume Attempt 3).
2. **Reject Attempt 2 → Attempt 3**: only if exact native arithmetic is judged
   non-portable (e.g., NumPy reduction/rounding order cannot be reproduced in
   C++ without the oracle's exact vectorized op sequence).

Evidence: this file, native tool under `tools/escha_native_transcode/`,
probe report absent (tool stopped before writing), microbench in the shell
history above.
