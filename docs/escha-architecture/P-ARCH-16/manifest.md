# P-ARCH-16 — final prefill attribution

**Status:** `COMPLETE — PREFILL ARCHITECTURE INVESTIGATION CLOSED`
**NEXT_GATE:** `D-ARCH-01 — matched decode baseline`

## Answer

With the best proven experimental configuration (SM120 MMA default + async
A-stage overlap + `-ub 2048`), Bee's matched 2k prefill is **919.410 ms /
2228.75 tok/s = 72.8% of Escha** (669.554 ms / 3058.75 tok/s), up from the
P-ARCH-05 baseline ratio of 40.7%. The remaining 249.856 ms delta splits into
K2 W2 (165.031 ms, 66.2%), K3 W2 (-30.463 ms, Bee faster), and the unsplit
non-W2 residual (114.799 ms, 46.0%).

## Attribution table

| Stage | Bee ms | Escha ms | Delta ms | % of delta |
|---|---:|---:|---:|---:|
| K2 W2 | 350.756 | 185.725 | 165.031 | 66.2% |
| K3 W2 | 300.377 | 330.840 | -30.463 | -12.2% |
| all W2 | 651.132 | 516.565 | 134.567 | 54.0% |
| non-W2 residual | 267.788 | 152.989 | 114.799 | 46.0% |
| **full prefill** | **919.410** | **669.554** | **249.856** | **100.0%** |

## Close rationale

- Remaining differences are fragmented/non-dominant: no single stage carries a
  majority of the residual once K3 is net-negative and the non-W2 bucket is
  unsplit.
- The single largest item (K2 MMA body) was tested with the two bounded
  corrections permitted by the gate family (geometry, boundary fusion) and
  neither moved it; the remaining Escha-style decode/staging change is a
  broad rewrite, outside the bounded-experiment scope of this run.
- Decode has never been attributed on this hybrid; it is now the larger
  unknown.

Evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-16/2026-08-29/final-attribution-001/`
plus the retained P-ARCH-15 matrix, P-ARCH-06 trace, and P-ARCH-05 wall data.

**CLOSE PREFILL ARCHITECTURE INVESTIGATION. NEXT_GATE: D-ARCH-01.**
