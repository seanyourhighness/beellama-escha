# EXP-04 Phase 2 — noise-resolution measurement (Sol-approved protocol)

Date: 2026-09-01. Protocol pre-registered by Sol: /tmp/exp04-noise-protocol.md
(**PROTOCOL=APPROVED**). Frozen binaries (/tmp/exp04-freeze/, hashes verified:
control lib `5bea9eb9…`, candidate lib `4f575fba…`). No code changes.

## Measurement (per protocol)

- 9 adjacent matched A/B pairs, order `AB BA BA AB AB BA BA AB AB` (18 trials),
  one unrecorded warmup per arm first, graphs ON, ESCHA_PROFILE OFF.
- Contract: `llama-bench -p 2048 -n 0 -ngl 99 -b 2048 -ub 2048 -ctk f16 -ctv f16
  -fa on`, canonical `escha-w2-lowgpu-mono-parity.gguf` (sha `e307007f…`),
  fixed shared-2048.ids, RTX 5090 (SM120, driver 610.88), GPU clock 2887 MHz
  constant across all trials, temp 46 °C, no competing GPU processes,
  all exit=0, all n_prompt=2048 n_gen=0.

## Results (raw)

| arm | samples (tok/s) | mean | sd | CV |
| --- | --- | --- | --- | --- |
| control | 2171.3, 2138.9, 2106.6, 2155.2, 1964.4, 2167.2, 2189.7, 2182.5, 2154.2 | 2136.7 | 69.2 | **3.24%** |
| candidate | 2344.6, 2370.8, 2374.4, 2395.8, 2231.6, 2304.2, 2355.9, 2334.9, 2382.3 | 2343.8 | 50.3 | **2.15%** |

- Median: control 2155.2, candidate 2355.9 → **median ratio gain +9.31%**
- Per-pair ratios: 1.0798, 1.1084, 1.1271, 1.1116, 1.1360, 1.0632, 1.0759,
  1.0698, 1.1059 — mean 1.0975, median 1.1059; candidate strictly faster 9/9.

## Protocol decision (pre-registered, no post-hoc statistics)

1. **Primary rule (per-arm CV ≤ 2%): NOT met** — control CV 3.24%, candidate
   CV 2.15%. The host cannot reach ≤2% per-arm (same WSL/610.88 noise seen in
   Stage 1 control 3.12% and every prior run). Per protocol, the fallback
   decides this block.
2. **Fallback paired-noise rule (pre-authorized): PASS**
   - Paired log-ratios: m=0.09282, s_y=0.02381
   - G = exp(m) = **1.0973 ≥ 1.05** ✓
   - 95% t CI = exp(m ± 2.306·s_y/√9) = **[1.0774, 1.1175]** — lower bound
     **> 1.00** ✓
   - Candidate strictly faster in **9/9** pairs (≥8/9 required) ✓
   - All other gates pass (below) ✓
   - **CONCLUSION: the pre-authorized paired-noise rule PASSES.** The
     candidate is faster with 95% confidence (CI entirely above 1.0) and the
     effect (≈+9.7% geometric mean) is comfortably above the ≥5% threshold.

## Same-session reconfirmations (frozen binaries)

| gate | result |
| --- | --- |
| Route proof (ESCHA_PROFILE=1, graphs off sidecar) | 800/800 tagged; 672 `mma-fp16-mixedacc` (IC≤6144) + 128 `mma-fp32-mixedacc` (IC>6144); **0 predicate mismatches**; all rows=2048 gen=0 |
| SASS .F16/.F32 split | resource-usage.txt + per-symbol proof: fp16 kernels only `HMMA.16816.F16` (16×, 0 F32); fp32 twins/control only `.F32` (32×, 0 F16); REG 97/128, no spills |
| P2/P7 | candidate 16/16, control 16/16 (100%) both prompts |
| Decode (r5) | candidate median 43.78 vs control 43.08 → **+1.63% (no regression ≤2%)** |
| Per-family regression | none (all 7 fp16 families improved; fp32-side family flat) — from Stage 2 evidence, unchanged binaries |
| Resource regression | none (REG 97 ≤ 128, STACK/LOCAL 0, static SHARED 1024, dynamic 13824 — identical to control) |
| Server fallback / route mismatch | none (all route lines from the Escha MMA path) |

## Classification (per ordered gates)

Full matched-2K improvement: median +9.31% (paired-log G +9.73%, 95% CI
[+7.74%, +11.75%]) → **≥5% and <20%: SMALLER POSITIVE**.

## Raw artifacts

- `evidence/EXP-04-phase2/2026-09-01/noise-run/trial-*.{json,stderr}` (18)
- `noise-run/run.log` (order, commands, telemetry, exit codes)
- `phase2-routeproof.{json,stderr}`, `phase2-decode-{candidate,control}.json`,
  `parity-{candidate,control}-phase2/compare-report.json`
- `RESOURCE-PROOF.md`
