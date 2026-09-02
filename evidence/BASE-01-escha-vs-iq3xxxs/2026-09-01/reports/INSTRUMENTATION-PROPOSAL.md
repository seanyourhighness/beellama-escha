# BASE-01 — Profiling-only instrumentation proposal (for Sol review)

## Problem
Phase 3 requires symmetric per-operator attribution for the IQ3 arm (standard
GGML ops). The ESCHA arm has the existing `ESCHA_PROFILE` CUDA-event hook.
The IQ3 arm has no equivalent hook, and nsys cannot capture CUDA kernel
durations under WSL (CUPTI limitation: only RUNTIME/API activity captured;
verified twice, retry with --trace=cuda --cuda-graph-trace=node also empty).

## Proposed change (minimal, env-gated, profiling-only)
Add one env-gated CUDA-event timer around the standard `mul_mat` dispatch in
`ggml/src/ggml-cuda/ggml-cuda.cu` (the `ggml_cuda_mul_mat` entry that every
standard quantized projection goes through), mirroring the existing
`ESCHA_PROFILE` pattern in `escha-moe.cu`:

- Gate: `GGML_OP_PROFILE` env var (unset = zero behavior change, no events, no
  sync, no output).
- Behavior when set: record `cudaEventRecord(start)` before the op, record
  `cudaEventRecord(stop)` after, `cudaEventSynchronize(stop)`, print one line:
  `GGML_OP_PROFILE op=mul_mat name=<tensor> ne=<dims> total_ms=<x>`
- Per-call, per-thread static events (like ESCHA_PROFILE) to avoid
  create/destroy churn.
- Same timing boundary as ESCHA_PROFILE: full logical operator inclusive of its
  kernels, measured on the op's stream.

## Isolation & safety
- Built ONLY into a separate build dir `build-cuda-base01-profile` (never the
  canonical `build-cuda-base01`).
- Profiling runs use graphs OFF (`GGML_CUDA_DISABLE_GRAPHS=1`) — relative
  attribution only; never canonical numbers.
- After evidence collection, source is reverted; `git diff` vs `be6bf478d` on
  `ggml/src/ggml-cuda/ggml-cuda.cu` must be empty and the canonical build
  binary SHAs must be unchanged.
- No kernel hot-path change: the event records wrap the existing dispatch call;
  with the env unset the added code is a single `getenv` check.

## Sol review request
Return CONFIRM or REVISE (with exact items) on this instrumentation design:
env gate, event reuse, boundary equivalence with ESCHA_PROFILE, isolation
build, revert discipline.
