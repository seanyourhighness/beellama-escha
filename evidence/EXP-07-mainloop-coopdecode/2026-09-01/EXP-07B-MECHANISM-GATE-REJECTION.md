# EXP-07b — Mechanism-Gate Rejection (2026-09-02)

## Decision

**FAIL; reverted before timing.** No benchmark, route run, numerical/parity run,
or performance campaign was executed. The promoted Stage 2 source remains the
checkout implementation and no WIP commit was created.

## Layout tested

The proposed `MT=4, NTT=2` two-row-warp layout is not a valid decode-once
register-B design: both row-warps consume the same 16-column B band, so each
must decode its own copy because registers cannot cross a warp boundary. That
would recreate EXP-02's duplicate decode.

EXP-07b therefore used the authorized disjoint-column alternative. Each of the
eight warps owned all 128 rows and one 8-column fragment (`MT=8, NTT=1`). Two
sequential, disjoint 64-column passes covered the original 128-column CTA. The
accumulator scope was reused between passes; each B column was decoded once.
The second pass replayed A to avoid keeping both column passes live.

## Build

- Fresh build: `build-cuda-exp07b-regb`
- CUDA: 13.0.88
- Architecture: `sm_120a`
- Gate: `-DESCHA_MMA_REGISTERB_EXPERIMENT=1`
- Target: `ggml-cuda`
- Compile result: PASS

## Resource gate

`cuobjdump --dump-resource-usage build-cuda-exp07b-regb/bin/libggml-cuda.so`

| kernel | regs | control | stack | local | result |
|---|---:|---:|---:|---:|---|
| K2 fp16 | 125 | 97 | 0 | 0 | **FAIL (+28)** |
| K2 fp32 | 141 | 128 | 0 | 0 | **FAIL (+13)** |
| K3 fp16 | 125 | 97 | 0 | 0 | **FAIL (+28)** |
| K3 fp32 | 141 | 128 | 0 | 0 | **FAIL (+13)** |

An intermediate single-pass 64-column probe compiled at 92 registers for fp16
and 108 for fp32 (K2 and K3; stack/local 0), but it represented only half the
control output tile and therefore could not meet the full-tile HMMA gate.

## Focused SASS gate

Counts are static instruction occurrences in each final full-tile candidate
symbol. K2 and K3 were identical within each accumulator mode.

| mode | STS.U16 | LDS.64 | HMMA.16816 | BAR.SYNC | LDL | STL | control HMMA / BAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 0 | 8 | 16 | 4 | 0 | 0 | 16 / 3 |
| fp32 | 0 | 12 | 24 | 6 | 0 | 0 | 32 / 6 |

- Shared-B elimination succeeded: `STS.U16 == 0`; the remaining `LDS.64`
  instructions are A-fragment loads.
- No spills were emitted (`STACK=LOCAL=0`, `LDL=STL=0`).
- Register ceilings failed for both accumulator modes.
- fp16 preserved control HMMA count but increased static barriers 3 -> 4.
- fp32 lost HMMA work 32 -> 24 and did not reduce barriers (6 -> 6).

The final candidate therefore fails multiple pre-timing gates. Per the approved
fail-fast rule, source was reverted and no benchmark was run.

