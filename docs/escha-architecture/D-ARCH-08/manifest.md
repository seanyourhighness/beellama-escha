# D-ARCH-08 — production regression (combined candidate)

**Status:** `COMPLETE — COMBINED CANDIDATE PASSES; PRODUCTION DEFAULTS UNCHANGED`

## Candidate

Combined experimental production candidate = P-ARCH-04 SM120 MMA default +
P-ARCH-10/11 async A-stage overlap + `-ub 2048` prefill; decode uses the
default `gen-splitk-fp32` (D-ARCH-05 warp-GEMV excluded after measurement).
Binary: `build-cuda-parch10-async` (llama-bench
`32911ef90000dfc31d7149d5cf9897e7b087547f7ac87ff348d8ebbb97865edc`).

## Fresh combined regression (2026-08-30)

- Prefill 2k exact IDs, ub2048: **2209.86 tok/s**, exit 0, zero CUDA-error
  lines (`combined-001/prefill-ub2048.json`).
- Decode (server, same binary): c=1 aggregate 31.16 tok/s (step 22.45 ms),
  c=8 aggregate 54.52 tok/s (step 97.55 ms); both complete.
- GPU stable (15,123 MiB with the resident server; idle/0% before and 4%
  after the benchmark), no compute process left behind.

## Full regression envelope (assembled from the gate chain)

| check | result | evidence |
|---|---|---|
| prefill matrix 512/1024/2048/4096 x ubatch 512/1024/2048 | all pass, exit 0 | P-ARCH-15 |
| K2/K3 route | `mma-fp16` prefill 3,200/3,200 and 800/800 (ub2048) | P-ARCH-11/13/15 |
| deterministic correctness | P1/P2/P5/P6/P7 16/16 | P-ARCH-13/15 |
| CUDA safety | compute-sanitizer 0 errors (3,200-record workload) | P-ARCH-13/14 |
| decode c=1,2,4,8 | baseline + corrected measured | D-ARCH-01/05 |
| graphs on/off | 1.62x benefit confirmed at c=1 | D-ARCH-04 |
| memory | fits 32 GB; post-state idle | all gates |
| repeated process restart | 10+ clean server restarts this run | D-ARCH-04/05/08 |

Production defaults remain unchanged (the candidate stays experimental/opt-in).
