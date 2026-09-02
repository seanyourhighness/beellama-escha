# Sol Gate 1 — Verdict Record (EXP-07 series, 2026-09-01)

## Verdict: PLAN=READY

- Round 1: PLAN=REVISE — 6 items: (1) v1 "CTA-cooperative decode-once"
  restated the control (control already decodes into shared `s_w` once per
  CTA); (2) candidate must be precisely specified; (3) pre-implementation
  mechanism gate required (SASS counts + direct measurements); (4) route
  arithmetic fix (FFN = 384 records of 800, not 384x3); (5) family-regression
  gate basis explicit; (6) ranking/parity math sound.
- Round 2 (v2): **PLAN=READY** — verified register-decoded B mechanism
  (removes Bee's 8 STS.U16 + 8 LDS.64 shared-B round trip; official has 0
  B stores/reloads), disjoint column-warp slices + fragment exchange, no
  per-M-warp duplicate decode (EXP-02 fix), frozen A-stage/geometry/acc/
  split-K/rotate/finalize; pre-timing gates (0 B STS/LDS, regs <= control,
  fewer barriers, >= HMMA, route proof, M=512/1024/2048 direct measures);
  corrected route arithmetic; explicit family-regression convention.

Series targets (T1..T5) and EXP-07 v2 detail:
- evidence/EXP-07-mainloop-coopdecode/2026-09-01/PLAN.md (v2)
- evidence/EXP-07-mainloop-coopdecode/2026-09-01/SERIES-PLAN.md

Dispatched via Codex CLI (read-only reviews).
