# EXP-retrospective — External research findings (2026-09-02)

Compiled via `research-run` (web + Reddit + X lanes, 20 evidence rows) +
targeted GitHub API repo discovery. Compiler run:
`~/.hermes/research-runs/escha-w2-prefill-breakthrough/`.

## Highest-value external find: independent ESCHAM → llama.cpp port

**`YanissAmz/escha-port`** (GitHub, master; kernel branch
`YanissAmz/llama.cpp` `feature/escha-format`,
`ggml/src/ggml-cuda/escham.cu`) — a working QTIP/ESCHAM port of the same
Qwen3.8-27B W2 format, with a detailed measurement write-up
(`docs/WRITEUP.md`). It independently hit the SAME wall as our series and its
conclusions corroborate Sol's retrospective:

- **"Tensor cores did not win on arithmetic."** Swapping FFMA→mma in isolation
  = 1.05×. The ×3 prefill win came from the smaller f16 accumulator freeing
  registers that make **band blocking** possible (blocking = reuse of the x
  tile across output bands, stopping L2 re-reads). "The instruction was the
  enabler, not the cause."
- **Stage progression (pp512, RTX 3090):** one-tile-per-warp 14.69 → inline
  codec (LUT removed) 72.90 → half2 decode 110.86 → shared-x batched 200.29 →
  tensor-core + band blocking **610.20 t/s**. ×33 total.
- **LUT trap:** a decode lookup table generated 13× more L2 traffic than the
  weights it replaced; computing the codec inline was the single largest win
  (Bee already does this — no LUT in the MMA path).
- **Their ceiling:** ncu showed both pipes saturated, **33% occupancy limited
  by registers (115/thread)**, and six tuning variants failed to beat the kept
  kernel. "Going further requires removing the round trip through shared
  memory: decoding into fragments and accumulating on the tensor cores. That
  is a rewrite, not a tuning." — i.e. exactly EXP-09's territory, and they
  stopped there too.
- **Bottom line:** escha (9.66 GiB) beats Q2_K_XL on disk but loses prefill
  2.27× vs Q4_K_XL on the same 3090 (610 vs 1383 t/s pp512). They concluded
  the format wins on size, loses on prefill; the deficit is kernel-level, not
  format-level — consistent with Sol's Decision A framing.

## Other external signal

- **`EschaLabs/escha-mlx`** (official, Apple Silicon MLX runtime, bit-exact
  Metal kernels) — only public EschaLabs repo; CUDA runtime (SGLang fork) is
  not public. Community TP-fix forks reference `escha-runtime-qwen3dense`
  (private).
- **X/community:** `sudoingX/qwen38-mtp` (llama.cpp MTP flag, +33–145% decode
  on Qwen3.8-27B dense; prefill not the lever), Blackwell 27B-dense community
  bench repo (23 entries), Qwen3.8-Flash-Next llama.cpp work (PR #27742) —
  all decode-side or standard-GGUF-side, none touching packed Escha prefill.
- **Reddit lane:** no signal this run (LocalLLaMA queries zeroed); niche
  ESCHAM kernel topic has no community thread.
- **CUTS/supporting:** NVIDIA CUTLASS Blackwell exists as reference for
  tcgen05-style MMA if a native kernel project (Sol direction 5) is funded.

## Implication for the breakthrough decision

External evidence independently validates Sol's retrospective: (1) removing
shared-B to chase the official two-band structure requires a *full
representation/kernel* change, not a local mainloop edit; (2) the only
demonstrated prefill-parity paths are representation-level (P-ARCH-23I
internally; standard-quant on disk externally); (3) decode ALU and MMA are
coupled through registers/occupancy on SM120. Both internal and external
analysis point to Decision A (certify P-ARCH-23I) unless exact packed-sidecar
execution is a hard product requirement — or, if it is, Sol direction 3
(cooperative BK32 double-buffered B) is the single authorized next kernel
experiment.
