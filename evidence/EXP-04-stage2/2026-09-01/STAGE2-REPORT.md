# EXP-04 Stage 2 — structurally-gated mixed accumulator

Date: 2026-09-01. Branch `escha-w2-prefill`.
Implementation commit `7b1880f41` (Sol-reviewed CONFIRM after fp16-store fix).

## ONE structural variable

Under compile gate `ESCHA_MMA_MIXEDACC_EXPERIMENT=1`, each K2/K3 MMA prefill
projection selects the accumulator by IC alone:
- `IC <= 6144` → `escha_matmul_dense_tiled_mma<K,128,128,FP16_ACC=true>`
- `IC > 6144`  → existing `...FP16_ACC=false>` (fp32)

This is the native Escha `escham_code_gemm` mixed policy, applied structurally
across every prefill family — NOT the rejected P-ARCH-20 single-shape
(5120→17408 only) toggle, and NOT measured on the sync path. Only the
accumulator type changes: geometry, grid, smem, decode, A-stage overlap,
fp32 partial buffer, and `escha_finalize_dense` are byte-for-byte unchanged.

## Fingerprints

- git_head: `7b1880f41` (implementation) on `b00263ba1` (Stage 1 base)
- escha-moe.cu sha256: (record in provenance.manifest)
- build dirs: `build-cuda-exp04-stage2-control` (gate off) /
  `build-cuda-exp04-stage2-mixedacc` (gate on)
- model: `escha-w2-lowgpu-mono-parity.gguf` sha256 `e307007f…4778d`
- GPU: RTX 5090, SM120 (cc 12.0), driver 610.88, 32,606 MiB

## SASS / resource proof (cuobjdump per-symbol)

| symbol | REG | STACK | LOCAL | SHARED | HMMA.16816 |
| --- | ---: | ---: | ---: | ---: | --- |
| mixedacc K2 fp16 (`Lb1E`) | 97 | 0 | 0 | 1024 | 16× F16, 0× F32 |
| mixedacc K3 fp16 (`Lb1E`) | 97 | 0 | 0 | 1024 | 16× F16, 0× F32 |
| mixedacc K2 fp32 (`Lb0E`) | 128 | 0 | 0 | 1024 | 0× F16, 32× F32 |
| mixedacc K3 fp32 (`Lb0E`) | 128 | 0 | 0 | 1024 | 0× F16, 32× F32 |
| control K2 fp32 | 128 | 0 | 0 | 1024 | 0× F16, 32× F32 |
| control K3 fp32 | 128 | 0 | 0 | 1024 | 0× F16, 32× F32 |

- fp16 kernels contain ONLY `HMMA.16816.F16`; fp32 twins contain ONLY `.F32`.
- Register gate ≤128: PASS (fp16 97, fp32 128). Zero spills, zero local memory.
- SHARED 1024 identical across all symbols (smem unchanged from control).

## Route proof (ESCHA_PROFILE, graphs off, attribution only)

800/800 prefill calls tagged; **0 predicate mismatches**:
- 672 × `mma-fp16-mixedacc` (all IC≤6144 families: K2 5120→10240/6144/5120
  [6144→5120]/17408/12288/1024, K3 5120→17408)
- 128 × `mma-fp32-mixedacc` (K3 17408→5120, IC>6144)

## Operator / numerical note

FP32-side families remain byte-identical (same fp32 kernel, partial, finalize).
FP16-side families accumulate in fp16 (expected rel-RMS ~2.1e-4, the same
scale as the official escha runtime's own mixed policy). Outputs finite, all
rows written; no NaNs/hangs at the 2k gate or parity.

## Performance — matched control/candidate, graphs on, profile off

2k prefill, canonical model, fixed shared-2048 IDs, `-b 2048 -ub 2048`,
F16 KV, FA on, r3/r5, full CUDA offload:

| run | candidate median | control median | gain |
| --- | ---: | ---: | ---: |
| r3 (A/B) | 2508.1 | 2281.0 | **+10.0%** |
| r3 (B/A) | 2496.0 | 2251.2 | **+10.9%** |
| r5 | 2496.9 | 2258.9 | **+10.5%** |

- Median gain stable across 3 matched runs: **+10.0 / +10.9 / +10.5%**.
- CV: 3.1–4.4% per arm on this WSL/610.88 host (candidate 4.1–6.3%, control
  3.1–4.2%) — above the plan's ≤2% letter; the gain is consistent, but the
  host's timing noise (same as Stage 1 control 3.12% and banked EXP-01
  samples) prevents a ≤2% formal CV pass. Flagged for Sol verify.

## Decode guardrail

`-p 0 -n 64 -r 5` (gen path, unchanged code): candidate median 44.55,
control 42.69 → **no regression** (+4.3% candidate, within noise; r3 earlier
showed 40.85 vs 44.16 which was a noisy sample). Decode does not use MMA.

## Parity — P2/P7

`run_compare.py --only P2-factual,P7-tool-call`, greedy, seed 42, 16 tokens:
- candidate: **16/16 (100%)** both prompts
- control: **16/16 (100%)** both prompts

## Family regression (steady-state matmul ms, candidate vs Stage-1 fp32 base)

| family | fp32 mm | mixed mm | delta |
| --- | ---: | ---: | ---: |
| k2 5120→1024 | 0.188 | 0.139 | −26.0% |
| k2 6144→5120 | 0.803 | 0.629 | −21.7% |
| k2 5120→10240 | 1.314 | 1.035 | −21.2% |
| k2 5120→6144 | 0.853 | 0.681 | −20.2% |
| k2 5120→12288 | 1.588 | 1.275 | −19.7% |
| k2 5120→17408 | 2.095 | 1.688 | −19.4% |
| k3 5120→17408 | 2.113 | 1.741 | −17.6% |
| k3 17408→5120 (fp32) | 2.228 | 2.218 | −0.5% |

No family regressed (all ≤ −17.6% on fp16 side; fp32 side flat). Regression
gate ≤5%: PASS.

## Classification

**SMALLER POSITIVE (≥5%, <20%): +10% full-2K median, stable across 3 runs.**
Not a ≥20% breakthrough. All hard gates except the host-noise CV letter pass:
route 0-mismatch, SASS F16/F32 split proven, no spills ≤128 regs, decode no
regression, P2/P7 16/16, no family regression.

## Rollback

Revert commit `7b1880f41` (or undefine the compile gate). Gate absent: default
K2/K3 fp32-acc dispatch, route tag `mma-fp16`, and control symbols are
byte-for-byte unchanged (control lib = previous default).

## Raw artifacts (evidence/EXP-04-stage2/2026-09-01/)

- mixedacc-profile.{json,stderr} — route proof (800 lines, 0 mismatch)
- mixedacc-bench.json / control-bench.json — matched r3
- mixedacc-bench-r2.json / control-bench-r2.json — matched r3 B/A
- mixedacc-bench-r5.json / control-bench-r5.json — r5
- mixedacc-decode-r5.json / control-decode-r5.json — decode guardrail
- parity-mixedacc/compare-report.json, parity-control/compare-report.json
- control-sass.txt / mixedacc-sass.txt — full cuobjdump dumps
- (per-symbol SASS/resource extracted inline above)
