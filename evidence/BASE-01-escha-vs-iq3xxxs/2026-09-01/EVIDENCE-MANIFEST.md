# BASE-01 — Raw Evidence Manifest (2026-09-01)

Evidence root: `evidence/BASE-01-escha-vs-iq3xxxs/2026-09-01/`

## Repository state
- HEAD local == remote: be6bf478dd6f2e63bea81a646bc8b613dbca9935
- Working tree: only `ggml-cuda.cu` modified (profiling-only GGML_OP_PROFILE hook,
  Sol INSTRUMENTATION=CONFIRM; isolated in build-cuda-base01-profile); canonical
  build-cuda-base01 binaries unchanged. Reverted before commit.

## Frozen binary
- build-cuda-base01 (HEAD be6bf478d): llama-bench sha 76485e115bc7c9212966c577906b8bbe7cfcb1cf39db44dbf5a48a80f7a6d82f, libggml-cuda sha d18663888721680a403ef063a7cc1c051b35f4d1ff7e6d4db4c293d1dc6f8534
- Profile build: build-cuda-base01-profile (llama-bench ea7a288a…, lib 3aac2a8c…) — hook only

## Artifacts
- Arm A: escha-w2-lowgpu-mono-parity.gguf sha256 e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d (8,619,127,360 B)
- Arm B: Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf sha256 ad85e40a28aafd907eebb6ff6b21786b897dd750b0918427f1243d6d84ebcc72 (9,570,663,040 B)
- Shared IDs sha256 695c3609bc35a32003a23be3ba1fbacc16cc94955548c2e855e91661c3f62350

## Phase 2 canonical (bench/)
- run.log (telemetry per trial), noise-run/p<t>t<1|2>-<A|B>.json + .stderr (18 trials)
- residency.md + residency-A/B.stderr + .json (65/65 layers CUDA0; CPU_Mapped vocab A 7726.18 MiB / B 644.14 MiB)
- Results: A 2326.77 tok/s median / 880.192 ms; B 3212.63 / 637.484 ms; G=1.3863 CI[1.3717,1.4010]; gap 242.708 ms

## Phase 3 attribution (profile/)
- A-escha-profile-{1,2,3}.stderr (ESCHA_PROFILE, 800 lines each) + ESCHA-AGGREGATE.json
- B-iq3-op-profile-{1,2,3}.stderr (GGML_OP_PROFILE, 994 lines per run; 2982 executed calls across 3 runs in the aggregate) + IQ3-OPPROFILE-AGGREGATE.json
- A-escha-graphsoff-{1,2,3}.json / B-iq3-graphsoff-{1,2,3}.json (816.2 / 590.3 ms avg)
- nsys retry artifacts documenting the WSL CUPTI kernel-duration limitation

## Phase 4 depth (operators/)
- ids-{128,512,1024,2048,4096}.txt (prefixes of shared IDs; 4096 doubled)
- A-escha-M<M>.{json,total.json,profile.stderr,total.stderr} and B-iq3-M<M>.* per depth
- DEPTH-MATRIX.json + clocks.log
- Whole-run gaps: 79.0 / 103.7 / 150.1 / 211.5 / 439.4 ms at M=128/512/1024/2048/4096

## Phase 5 SASS (sass/)
- resource-usage-full.txt (25,268 lines), RESOURCES.json
- Key: ESCHA tiled MMA fp32 128 regs / fp16 97 regs, no spills, smem 1024; MMQ 242–255 regs BK=128

## Scripts (scripts/)
- run-matched-campaign.sh, analyze-campaign.py, run-profile-attribution.sh,
  aggregate-escha-profile.py, aggregate-iq3-opprofile.py, account-family-gap.py,
  run-operator-depth.sh, analyze-operator-depth.py, extract-sass.sh, parse-resources.py,
  correlate_artifacts.py, correlate_qkv_split.py, correlate_permute.py, correlate_probes*.py,
  correlate_fix*.py, correlate_ffn_gate_k2.py

## Reports (reports/)
- STARTING-STATE.md, SOL-GATE1-VERDICT.md, PLAN-SOL-GATE1.md, INSTRUMENTATION-PROPOSAL.md,
  PHASE2-CANONICAL-RESULTS.md, PHASE3-COMPONENT-BREAKDOWN.md, PHASE3B-FAMILY-GAP.md,
  PHASE4-OPERATOR-DEPTH.md, PHASE5-SASS-RESOURCES.md, BASE-01-REPORT.md

## Manifests (manifests/)
- ARTIFACT-MANIFESTS.md, CORRELATION-SUMMARY.md, CORRELATION.json, CORRELATION-QKVSPLIT.json,
  CORRELATION-PERMUTE.json, CORRELATION-PROBES*.json, CORRELATION-FIX*.json,
  CORRELATION-FFN-GATE-K2.json
