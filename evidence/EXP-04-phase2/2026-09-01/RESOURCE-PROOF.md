# EXP-04 Phase 2 — frozen binaries + resource proof

Date: 2026-09-01. Binaries frozen; no performance-code changes during
verification.

## Frozen binaries (copied to /tmp/exp04-freeze/{control,candidate} on ext4)

| file | control sha256 (16) | candidate sha256 (16) |
| --- | --- | --- |
| llama-bench | 34df37036f2ca1fe | 570bcc57695989fe |
| llama-server | 5dae9cf0e8ed12e7 | e5a8ad5155871e7f |
| libggml-cuda.so.0.19.0 | 5bea9eb9d9f36254 | 4f575fba83aff444 |

These match the hashes recorded in
`evidence/EXP-04-stage2/2026-09-01/provenance.manifest` and the git-committed
build outputs. Candidate built with `-DESCHA_MMA_MIXEDACC_EXPERIMENT=1` at
commit `7b1880f41`; control is the gate-off build of the same source.

## Per-kernel resource usage (cuobjdump --dump-resource-usage -fun <symbol>)

| kernel | REG | STACK | LOCAL | static SHARED | dynamic SHARED | spills |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| control K2 fp32 (`Lb0E`) | 128 | 0 | 0 | 1024 | 13824 | none |
| control K3 fp32 (`Lb0E`) | 128 | 0 | 0 | 1024 | 13824 | none |
| candidate K2 fp32 (`Lb0E`) | 128 | 0 | 0 | 1024 | 13824 | none |
| candidate K3 fp32 (`Lb0E`) | 128 | 0 | 0 | 1024 | 13824 | none |
| candidate K2 fp16 (`Lb1E`) | 97 | 0 | 0 | 1024 | 13824 | none |
| candidate K3 fp16 (`Lb1E`) | 97 | 0 | 0 | 1024 | 13824 | none |

- No local memory, no stack frame, no spill stores/loads in any kernel.
- Dynamic shared memory = `NTJ*ESCHA_MAX_W*sizeof(uint2) + 2*BM*TILE*sizeof(half)
  + BN*TILE*sizeof(half)` = 8*24*8 + 2*128*16*2 + 128*16*2 = 1536 + 8192 +
  4096 = **13,824 B** (14 KiB). Identical for control and candidate because
  the launch site passes the same `smem` for both template instantiations.
- Static shared 1024 B identical.

## Occupancy implications (SM120, 65,536 regs/SM, ~227 KiB smem/SM)

- fp32 kernels (128 regs × 256 thr = 32,768 regs/CTA) → **2 CTAs/SM** (register
  bound).
- fp16 kernels (97 regs × 256 thr = 24,832 regs/CTA) → 2.64 by registers, but
  **2 CTAs/SM** effective (same shared-memory budget 14 KiB/CTA; 2 CTAs is the
  same residency as the control). Lower register pressure leaves headroom and
  removes spill risk.
- Neither path changes tile/grid geometry, so CTA count and split-K policy are
  identical.

## Verification of freeze integrity

- Control `libggml-cuda.so` sha `5bea9eb9d9f36254` = the same library used for
  the Stage 2 matched control runs (evidence/EXP-04-stage2 JSONs reference the
  build dir; binary hash matches provenance.manifest).
- Candidate `libggml-cuda.so` sha `4f575fba83aff444` = the Stage 2 candidate.
- No rebuild, no source edit since `7b1880f41`; `git status` shows only
  evidence/docs commits (kernel untouched).
