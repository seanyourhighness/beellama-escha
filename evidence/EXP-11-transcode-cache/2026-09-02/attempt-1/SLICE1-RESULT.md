# EXP-11 Attempt 1 — Slice 1 result: oracle proven; operational trigger fired

Date: 2026-09-02 · Author: Sol (implementation) + Terra (review)

## What was delivered (files)

- `conversion/escha/transcode_oracle.py` — shared deterministic oracle
  (reconstruction identical to `escham_cpu.reconstruct_deploy_weight` reading
  GGUF-stored code/rin/rout; pure-NumPy q2_k/q4_k/q6_k quantizers) used by
  BOTH the converter and the generator.
- `conversion/escha/transcode_cache_schema.py` — frozen cache/manifest schema
  (PROGRAM-PLAN §6: root layout, recipe-id, per-entry key, manifest, overlay
  sha, complete marker, atomic tmp+rename).
- `tools/escha-transcode-cache.py` — `prepare` / `verify` (deep) / `compare`
  CLI. One-layer probe run with `/usr/bin/time -v`.
- Evidence under `evidence/EXP-11-transcode-cache/2026-09-02/attempt-1/`
  (`prepare-a/b.json`, `determinism.json`, `oracle-equality.json`,
  `cache-verify.json`, `corruption-rejection.json`, `.time.txt` files, cache
  trees run-a/run-b with `complete` markers).

## Correctness gates — ALL PASS

| gate | result |
|---|---|
| Oracle payload byte-equal (blk.0 ffn_gate/up/down) | True ×3 (SHA matched) |
| Converter regression (layer-0 --standard-gdn-gate-quant q2_k) | byte_equal=True (refactor changed nothing) |
| Determinism (prepare twice) | overlay SHA `f3911881…` identical; entry SHAs identical |
| Deep verify | status ok, no errors |
| Dequantized MAE vs fp32 source | 0.00217–0.00233 (Q2_K-expected class) |
| Source identity | canonical SHA `e307007f…` confirmed |

## Operational gate — ATTEMPT-2 TRIGGER FIRED (frozen plan §8/§11)

One layer (3 FFN tensors): wall 150–169 s incl. one-time source hash;
per-tensor reconstruction ~5 s, quantization ~28 s, validation ~0.6 s.
**Projected all-64 cold build: 6,663–6,860 s (~111–114 min) vs the 120 s
frozen budget — `wall_budget_holds=False`.** RSS 2.79 GB: under the 3.0 GB
hard gate but over the 2.5 GB target. `attempt2_trigger=True` (computed by the
tool itself from the frozen budget).

Per PROGRAM-PLAN §12 step 3 ("measure one layer and project all 64 before
launching the full build") and §11 trigger row 3, the full 64-layer monolithic
build must NOT proceed. Attempt 1's NumPy-authoritative architecture is
consumed on its frozen operational gate; correctness artifacts are durable and
reused by Attempt 2 unchanged (byte-equal oracle is the contract).

## Attempt accounting

- Attempt 1 (NumPy-authoritative monolithic overlay): **exhausted on the
  operational gate** (correctness proven; 120 s cold budget missed by ~55×).
- Attempt 2 (native streaming, layer-sharded, resumable cache): next.
- Attempt 3 (new sidecar representation): reserved.

## Required next step

Sol writes the written revision (PROGRAM-PLAN §11): name the failed gate
(cold wall), attach raw evidence (this file + `.time.txt`), state why the
native streaming/shared shard architecture addresses it (bounded per-layer
RSS, resumable, parallelizable, deterministic byte-equality to the proven
NumPy oracle), freeze the Attempt-2 recipe and budgets, and confirm attempts
remaining = 1 (Attempt 3 still reserved).
