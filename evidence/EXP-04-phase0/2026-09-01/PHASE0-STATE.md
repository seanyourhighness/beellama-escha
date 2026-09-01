# Phase 0 — state reconciliation record (2026-09-01)

## Git state
- Local HEAD: `6ef86bfbd370dd7f1df4bb169ba0cf81b0fc3119` (branch `escha-w2-prefill`)
- Remote `github/escha-w2-prefill`: `4501b3ee1` — the certified pre-EXP-04
  checkpoint (ARCH-01). Local is ahead by 7 commits.
- Ancestry: `4501b3ee1` (ARCH-01) and `7b1880f41` (Stage 2 impl) are both
  ancestors of HEAD. Confirmed.
- **Protected rollback checkpoint:** `4501b3ee1`
  (docs: ARCH-01 audit; kernel = EXP-01 default route, no Stage-2 gate).
  Rollback of the Stage 2 guarded operator = revert `7b1880f41` (or build
  without `-DESCHA_MMA_MIXEDACC_EXPERIMENT=1`).

## Untracked files / symlinks (local execution aids only)
- `.escha-venv/` — Python venv (transformers+jinja2 for run_compare.py);
  275 MB; added `/.escha-venv` to .gitignore.
- `weights/escha` → `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha`
  (symlink; tokenizer source for run_compare.py). NOT to be committed.
- `results/prompts.json` → `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/results/prompts.json`
  (symlink; prompt corpus for run_compare.py). NOT to be committed.

## Do-not-commit policy
Per goal + AGENTS.md: no venvs, weights, model symlinks, or temporary prompt
links. Evidence JSONs/hashes/commits are explicitly requested by the goal and
are committed normally.
