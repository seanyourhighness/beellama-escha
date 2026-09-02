# PREFILL +50% — A1 baseline measurements (graphs ON, same binary)

Date: 2026-09-02 · Binary: `build-cuda-exp10-control` (HEAD `bbbb98f6c`)
Host: RTX 5090 · matched 2048-token, `-ub 2048`, F16 KV, graphs ON, r3.

| artifact | tok/s | ms/2k | vs hist 2319.22 | vs same-bin 2426.1 |
|---|---:|---:|---:|---:|
| canonical full-Escha (control) | 2,426.1 | 849.6 | +4.6% | — |
| **P-ARCH-23I** | **3,242.8** | 632.8 | **+39.8%** | **+33.7%** |
| LowGPU IQ3_XXS reference | 3,338.8 (hist) | 613.2 | +44.0% | — |
| **+50% target** | **3,479** | 588.7 | +50.0% | +43.4% |

## Findings

1. 23I at **graphs ON = 3242.8**, slightly BELOW its graphs-OFF 3300.4 —
   CUDA graphs do NOT help the standard body (no escha decode to capture);
   the +42.3% historical claim was graphs-off. Honest same-protocol gain is
   **+39.8% over the certified control**.
2. 23I is still **2.9% below LowGPU parity** (3243 vs 3339) and **6.8% short
   of the +50% target** (3479). The A2/A3 window is real: ~+5–7% needed.
3. Because LowGPU (9.571 GB, IQ3_XXS-heavy) beats 23I (9.345 GB, Q2_K-heavy)
   despite being larger, **quant decode-speed, not size, is the residual
   lever** — supporting A2 (per-family quant microbench) as the next attempt.

Evidence: `23i-graphs-on.json`, `control-graphs-on.json`, `.stderr` in this dir.
