# P-ARCH-18B — current-body vocabulary representation control

**Status: CLOSED — CASE A.** With every non-vocabulary tensor byte-identical,
dense F16 vocabulary representation is faster than the LowGPU-vocabulary
representation for the matched 2k prefill boundary.

## Control

- Fixed body: 2,052 common tensors byte-identical between artifacts, verified by
  `scripts/escha-compare/verify_parch18_f16vocab.py`.
- Dense `token_embd.weight` and `output.weight` are exact dequantizations of
  the LowGPU source tensors (`max_abs_diff = 0`).
- Same `build-cuda-parch10-async` binary, immutable 2,048 IDs, `-p 2048 -n 0
  -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on`, CUDA graphs on.
- Alternating two-capture A/B: F16 → LowGPU → LowGPU → F16; four samples per
  capture, eight samples/artifact, median wall time.

## Result

| Vocabulary form | median | tok/s | delta vs LowGPU vocabulary |
| --- | ---: | ---: | ---: |
| LowGPU sidecars (current hybrid) | 884.416 ms | 2315.65 | baseline |
| Exact dense F16 dequant | **842.835 ms** | **2429.89** | **-41.581 ms (-4.70% time)** |

Raw evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-18/2026-08-30/f16vocab-clean-001/` and `f16vocab-clean-002/`.

## Interpretation

This closes the originally proposed clean artifact control: P-ARCH-17's
model-level movement was not solely body lineage/order. The LowGPU vocabulary
operator/workspace path costs at least 41.581 ms under this matched boundary.
The `-n 0` graph still includes the one-token LM head, so this result includes
both embedding and output-projection representation effects. It does not by
itself identify their individual shares.

**NEXT:** Rebase the remaining budget from the F16-vocabulary control before
running accumulation or direct-fragment experiments; retain a generation
quality gate before any model-format default changes.
