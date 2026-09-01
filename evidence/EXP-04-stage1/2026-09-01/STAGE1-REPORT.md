# EXP-04 Stage 1 — fuseable rotate/GEMM/finalize bound (attribution)

Date: 2026-09-01. Branch `escha-w2-prefill`, HEAD `4501b3ee1`
(ARCH-01 commit; `escha-moe.cu` byte-identical to EXP-01 promotion `215aa4ac3`).

## Purpose

Measure the per-stage cost of BeeLlama's separate rotate / packed-GEMM /
finalize launches on the current default route, to quantify the fuseable upper
bound before any structural change (EXP-04 Stage 2). Measurement-only: no code
changes. Per audit: `ESCHA_PROFILE` per-stage ms per projection.

## Fingerprints

- git_head: `4501b3ee10dfce3451df7442c7baa2b03021a105`
- escha-moe.cu sha256: `bfe0e43d135220cc2d62033c12ac4b43896cf07b4ae5dcbf81bc98dc215b43c2`
  (identical to EXP-01 consolidated provenance — kernel unchanged)
- llama-bench sha256: `bb174036b76a91aa457cb6ac32b6c41db33a8ef29a86135ce80ebe9dfe908bf1`
- llama-server sha256: `15f41fd706242c79ca3ccdc4a3211217af5b0811cf03654011320f3398bbde27`
- libggml-cuda.so.0.19.0 sha256: `29cdcc2e367148678049b0adfd200bdba936cd6c2537af3fe4414747e6dcf52d`
- model: `escha-w2-lowgpu-mono-parity.gguf` sha256 `e307007f…4778d`
  (canonical full-Escha control), size 8,619,127,360 B
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32,606 MiB,
  driver 610.88
- Build: fresh `build-cuda-exp04-stage1` (cmake + ninja, Release,
  `-DCMAKE_CUDA_ARCHITECTURES=120`, `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`,
  `GGML_NATIVE=OFF`, `GGML_CUDA_GRAPHS=ON`)

## Contract

- `llama-bench -p 2048 -n 0 -ngl 99 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on`
  with the fixed shared-2048 prompt IDs
  (`escha-w2-lowgpu/evidence/P-ARCH-05/…/shared-2048.ids`).
- Attribution run: `ESCHA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1` (profile forces
  per-op sync, so graphs disabled — attribution only, not a timed score).
- Timed control (graphs on, no profile): median 2284.7 tok/s
  (samples 2193.6 / 2333.3 / 2284.7; avg 2270.5 ± 70.9) — reproduces the banked
  EXP-01 2k baseline (~2302 tok/s) within CV. Decode 64: samples 39.3 / 44.2 /
  40.6 tok/s (median 40.6) — no regression vs EXP-01 decode guardrail.

## Route proof

800/800 `ESCHA_PROFILE` lines on the attribution run report
`route=mma-fp16 rows=2048 gen=0`. No fallback, no tiled-fma call.

## Per-stage attribution (steady state, excluding the first cold call per family)

| family | n | mean total ms | rotate ms (%) | matmul ms (%) | epilogue ms (%) |
| --- | --- | --- | --- | --- | --- |
| k2 5120→10240 rows=2048 | 95 | 1.503 | 0.059 (3.9%) | 1.314 (87.5%) | 0.129 (8.6%) |
| k2 5120→6144 rows=2048 | 95 | 0.977 | 0.062 (6.4%) | 0.853 (87.3%) | 0.062 (6.3%) |
| k2 6144→5120 rows=2048 | 127 | 0.909 | 0.058 (6.4%) | 0.803 (88.3%) | 0.049 (5.3%) |
| k2 5120→17408 rows=2048 | 127 | 2.356 | 0.046 (2.0%) | 2.095 (88.9%) | 0.216 (9.1%) |
| k3 5120→17408 rows=2048 | 127 | 2.389 | 0.064 (2.7%) | 2.113 (88.4%) | 0.213 (8.9%) |
| k3 17408→5120 rows=2048 | 127 | 2.468 | 0.191 (7.7%) | 2.228 (90.3%) | 0.049 (2.0%) |
| k2 5120→12288 rows=2048 | 31 | 1.772 | 0.041 (2.3%) | 1.588 (89.6%) | 0.143 (8.1%) |
| k2 5120→1024 rows=2048 | 63 | 0.256 | 0.056 (21.8%) | 0.188 (73.2%) | 0.013 (5.0%) |

Aggregate (792 steady-state calls): **rotate 4.6% · matmul 88.6% · epilogue 6.7%**
(61.9 / 1186.1 / 90.2 ms of 1338.3 ms measured projection time).

## Findings

1. **The packed GEMM body dominates: 88.6% aggregate, ≥73% in every family.**
   Matmul is the wall, exactly as the audit expected.
2. **The fuseable launch bound is small.** rotate (4.6%) + epilogue (6.7%) =
   ~11.3% of measured projection time in the best case where fusion fully
   eliminates both stages and their launch overhead. P-ARCH-14 already found
   fused finalize neutral, so the realistic recoverable portion is lower
   (rotate-only fusion ≈ 4.6%; finalize-only ≈ 6.7% but shown neutral).
3. **Cold first-call artifacts are large but non-recurring.** The first call of
   each family shows inflated rotate/epilogue (e.g. 5120→10240 first call:
   rotate 3.40 ms, epilogue 8.47 ms vs 0.059/0.129 steady state) — warmup /
   allocator / first-launch overhead, not steady-state cost. Excluded from
   attribution; reported for completeness.
4. **The 5120→1024 family is the exception** (rotate 21.8%): small projection
   where rotate launch dominates, but it is only 63/800 calls and 0.256 ms
   mean — immaterial to the 2k wall.

## Stage 1 conclusion

A rotate/GEMM/finalize fusion candidate cannot plausibly deliver the ≥20%
breakthrough gate by itself: the fuseable bound is ~11% of projection time and
matmul is 88.6%. Stage 2 should target the packed-GEMM body (structural mixed
accumulator or B-decode/launch structure with SASS proof), not launch fusion,
unless a fusion experiment is justified purely as a small positive (≥5% gate).
This is attribution, not a wall-speed claim: profile-mode totals include per-op
sync bubbles and are NOT the timed result (median timed 2k = 2284.7 tok/s).

## Raw artifacts

- `stage1-profile.json` / `stage1-profile.stderr` (ESCHA_PROFILE=1 run)
- `stage1-bench.json` / `stage1-bench.stderr` (timed control, graphs on)
- `stage1-decode.json` / `stage1-decode.stderr` (decode guardrail)
- Aggregation: python heredoc in this session (see ledger entry).
