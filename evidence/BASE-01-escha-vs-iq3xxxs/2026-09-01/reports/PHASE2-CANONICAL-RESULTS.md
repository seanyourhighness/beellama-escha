# BASE-01 Phase 2 — Canonical Matched Campaign Results (2026-09-01)

Frozen binary: `build-cuda-base01` (HEAD be6bf478d), llama-bench sha `76485e11…`,
libggml-cuda sha `d1866388…`. Both arms fully GPU-resident (65/65 layers CUDA0;
vocab CPU_Mapped: A 7726.18 MiB / B 644.14 MiB — storage-only, kernels on CUDA0).
Contract: `-p 2048 -n 0 -ngl 99 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 1`,
graphs ON, shared-2048 IDs (sha `695c3609…`), throughput = 2048 / prompt_seconds.
Block 1: AB BA BA AB AB BA BA AB AB, fresh process per trial, warmup per arm.

## Raw trials (block 1)

| trial | arm | tok/s | ns |
|---|---|---|---|
| p1t1 | A | 2345.64 | 873107846 |
| p1t2 | B | 3204.40 | 639122070 |
| p2t1 | B | 3231.22 | 633815546 |
| p2t2 | A | 2316.72 | 884007330 |
| p3t1 | B | 3212.12 | 637584349 |
| p3t2 | A | 2326.77 | 880191684 |
| p4t1 | A | 2227.58 | 919382260 |
| p4t2 | B | 3192.42 | 641519743 |
| p5t1 | A | 2343.50 | 873908005 |
| p5t2 | B | 3237.58 | 632570375 |
| p6t1 | B | 3217.36 | 636546567 |
| p6t2 | A | 2336.91 | 876370630 |
| p7t1 | B | 3239.48 | 632201168 |
| p7t2 | A | 2334.03 | 877450687 |
| p8t1 | A | 2301.96 | 889674968 |
| p8t2 | B | 3164.46 | 647187782 |
| p9t1 | A | 2323.80 | 881315369 |
| p9t2 | B | 3212.63 | 637484029 |

## Statistics

- **ARM A (ESCHA):** median tok/s 2326.77; mean 2317.44; sample-SD 36.35; CV 1.57% (tok/s), 1.61% (latency). Median latency 880.192 ms.
- **ARM B (IQ3):** median tok/s 3212.63; mean 3212.41; sample-SD 23.74; CV 0.74%. Median latency 637.484 ms.
- CV contingency: NOT triggered (both arms < 2% sample-SD CV).
- Paired latency ratios (A/B): 1.3661, 1.3947, 1.3805, 1.4331, 1.3815, 1.3768, 1.3879, 1.3747, 1.3825
- **Paired-log G = 1.3863; 95% CI [1.3717, 1.4010]** (df=8, t=2.306); pairwise B-faster 9/9; CI excludes 1.0 → decisive.
- **Median latency gap (A−B): 242.708 ms. Median tok/s gap (B−A): 885.9 tok/s.**

## Historical comparison

| claim | result |
|---|---|
| ESCHA ~2356 tok/s | **REPRODUCED** (median 2326.8; −1.2% — consistent with Stage 2 promoted control) |
| IQ3 ~3339 tok/s (P-ARCH-23F, build 0b035b3a2) | **NOT REPRODUCED on this build** (3212.6; −3.8% vs 3339; different frozen binary) |
| IQ3 ~3600 tok/s | **NOT SUPPORTED** — no recorded measurement anywhere; canonical median 3212.6 |

Note: the mission premise framed the gap as ~300 ms / ~3600 tok/s. The canonical
same-runtime gap is **242.7 ms** and the IQ3 reference under this binary is
**3212.6 tok/s**. The 3600 figure appears only as a stretch target in
`docs/escha-w2-prefill-next-plan.md`; the recorded P-ARCH-23F reference (3339)
was measured on an older preview build (0b035b3a2). Attribution proceeds on the
measured 242.7 ms gap.
