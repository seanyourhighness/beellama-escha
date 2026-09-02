# Sol Gate 1 — Verdict Record (2026-09-01)

## Verdict: PLAN=READY

- Round 1: PLAN=REVISE — 5 items (residency proof, operator boundaries, graphs-off accounting, provenance stratification, sample-SD CV + contingency).
- Round 2: PLAN=REVISE — 2 items (residency telemetry completeness; CV contingency operationalization: block-aware output, combined 18-pair analysis, df, inconclusive rule).
- Round 3: PLAN=REVISE — 1 item (analyzer must enforce triggered second-block completion before a definitive combined decision; no mispairing on missing trials).
- Round 4: PLAN=REVISE — 2 items (triggered block-2 absence must force INCONCLUSIVE; incomplete blocks must emit combined INCONCLUSIVE, not NO DECISION).
- Round 5: **PLAN=READY** — verified HEAD be6bf478dd; cv_triggered enforced; combined analysis runs with any arm data; incomplete/required-missing block forces INCONCLUSIVE.

All REVISE items resolved. Authorized to collect authoritative data per the plan:
- Canonical campaign: 9 matched pairs (AB BA BA AB AB BA BA AB AB), fresh process/trial, one unrecorded warm-up per arm, graphs ON, `-p 2048 -n 0 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -ngl 99`, shared-2048 IDs, throughput = 2048 / measured_prompt_seconds.
- Residency proofs (per-arm -v stderr + json + residency.md), VRAM telemetry per trial.
- CV>2% contingency: run BLOCK=2 (second 9-pair block) and combine with df=17; CI spanning 1.0 or incomplete blocks => INCONCLUSIVE.
- Attribution (Phase 3+): graphs-off only, symmetric full-operator boundaries, accounting closed inside graphs-off totals; per-arm graphs-on/off delta reported separately; no ratio scaling.

Dispatched via Codex CLI (`codex exec`, sandbox danger-full-access), read-only reviews.
