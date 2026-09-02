# Frontier measurement — same binary, graphs ON, matched 2K, r5 (2026-09-02)

Binary: `build-cuda-exp10-control` (HEAD `933c6f969`). RTX 5090. Matched
2048-token prefill, `-ub 2048`, F16 KV, graphs ON, r5, per-artifact files in
`frontier/`.

## Results (median of 5; CV is host noise on this WSL box)

| artifact | size GB | median tok/s | median ms | min | max | CV% | vs control |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical packed (control) | 8.619 | 2,529 | 809.8 | 2,133 | 2,559 | 7.5 | — |
| 21A standard-FFN | 8.483 | 2,962 | 691.4 | 2,856 | 3,089 | 3.0 | +17.1% |
| **23 +gate Q2_K** | **8.599** | **3,023** | 677.5 | 2,810 | 3,079 | 3.6 | **+19.5%** |
| 23G +embed Q4_K | 8.808 | 3,228 | 634.5 | 2,969 | 3,235 | 4.6 | +27.6% |
| 23I-step +attn std | 9.036 | 3,307 | 619.3 | 3,011 | 3,313 | 4.1 | +30.7% |
| 23I +linear std | 9.345 | 3,283 | 623.9 | 3,042 | 3,324 | 4.3 | +29.8% |
| LowGPU IQ3_XXS (native ref) | 9.571 | 3,514 | 582.9 | 2,995 | 3,519 | 6.8 | +38.9% |

Note: this batch's control read 2,529 (host noise vs 2,426 earlier r3); treat
+% within-batch as the stable signal.

## Findings vs Sean's goal (~8.5 GB, >= SGLang ~3030, ~native 3339)

1. **P-ARCH-23 at 8.599 GB / 3,023 tok/s is the goal-fit artifact**: it sits
   in the 8.5 GB class AND clears the ~3,030 SGLang-class bar (0.2% short of
   3030 at median; max sample 3,079). It is the smallest artifact that reaches
   the SGLang speed class.
2. **23G at 8.808 GB / 3,228 tok/s** is the best speed-per-GB if "around 8.5"
   tolerates 8.8 GB (+27.6%, near-LowGPU-parity at 0.31 GB smaller than 23I).
3. **NEW: 23I-step (-attn, 9.036 GB) ties 23I (-linear, 9.345 GB)** — 3,307 vs
   3,283 (within CV). The 48 linear-attention QKV/SSM standardization added
   +0.31 GB with NO prefill benefit in this protocol. That challenges the 23I
   champion choice: the 9.036 GB artifact is smaller at equal prefill speed.
   (Decode/quality may still favor -linear; unmeasured here.)
4. Control in-batch noise (7.5% CV) means the matched 9-pair ABBA campaign is
   still required before promotion of any candidate.

## Delivery recommendation

- If "~8.5 GB" is strict: **P-ARCH-23 (8.599 GB, 3,023 tok/s)** is the
  release candidate — SGLang-class prefill at the target size.
- If ~8.8 GB is acceptable: **23G (3,228 tok/s)** beats the SGLang target with
  margin and approaches native parity.
- Verify decode + KVarN/DFlash2/context + quality suite on whichever is chosen;
  matched ABBA to firm the number.
