# Path A forensic — how the official kernel actually feeds MMA (2026-09-02)

## The discovery (from hot-k2-sass.txt, official escham_code_gemm)

The official mainloop does **NOT** decode → shared-B → ldmatrix like Bee. Its
real structure, traced from SASS:

1. **B fragments are register-resident and reused across A fragments.** In a
   14-HMMA cluster, only **2 B fragments** (R46/R48) feed **5 distinct A
   row-fragments** (R8×4, R4×4, R28×2, R16×2, R12×2) producing **14 distinct
   accumulator tiles**. That is: B decoded once into registers, then reused
   across the full row span — decode amortized over many MMAs.

2. **A is streamed from shared via plain `LDS`** (116 LDS in the MMA window),
   NOT ldmatrix. A fragments load right before their HMMA; B stays live.

3. **The codebook transform is `SHF`+`PRMT`+`HADD2`** (24 SHF, 14 PRMT,
   26 HADD2 in the window) on payload words loaded by LDS — no
   multiply-based `escha_codebook_h`, no `LOP3`-heavy path in the hot loop
   (only 48 LOP3 in-window vs Bee's 41 per 16 HMMA).

4. **64 HMMAs per mainloop region** (our control: 16 per K16 loop body) =
   four-K16-deep unrolled region with B fragments held across the whole region.

## Why this differs from every Bee attempt

| | Bee shared-B control | Official |
|---|---|---|
| B path | decode → shared → LDSM each warp | decode → **registers, reused across rows** |
| A path | LDSM (ldmatrix) from shared | plain LDS streamed per-HMMA |
| decode ALU | arithmetic codebook (mul+lop3) | SHF/PRMT/HADD2 on LDS payload |
| HMMA/region | 16 | 64 (4-K16) |
| B reuse | re-decoded/reloaded per K16 | register-resident across region |

V1–V5 all kept Bee's *arithmetic decode* and varied the *pipeline around it*.
The official instead (a) makes B register-resident so decode is amortized
across the whole row span, and (b) uses SHF/PRMT/HADD2 — cheap bit
manipulation — rather than the multiply codebook. **Both differences are in
the decode path itself, which we never changed.**

## Path A target (revised)

Not "beat control by 15%". Reach official's structure: register-resident B
reused across the row span + SHF/PRMT/HADD2 decode + 4-K16 region → ~4.75
cyc/HMMA = ~3030 tok/s on the packed artifact = SGLang parity. The measured
official-vs-control gap is exactly 21%, matching the audit's 0.78–0.90×.

## Next step (blocked on Sol, capped to Sep 6 19:58)

Sol is hard-capped until Sep 6 7:58 PM. Path A implementation needs Sol's
high-reasoning pass to transcribe this decode structure into a working kernel
with the frozen gate. This file is the head-start so no time is lost when the
cap clears.
