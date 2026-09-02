# BASE-01 — Starting State Record (2026-09-01)

Mission: BASE-01 — ESCHA W2 versus IQ3_XXS LowGPU Same-Runtime Breakdown.
Read-only attribution investigation. No optimization implementation in this task.

Evidence root: `evidence/BASE-01-escha-vs-iq3xxxs/2026-09-01/`

## 1. Repository state (Phase 0)

- Repo: `/mnt/d/CODEX WORKSPACE/beellama-escha`
- Branch: `escha-w2-prefill`
- Local HEAD: `be6bf478dd6f2e63bea81a646bc8b613dbca9935`
- Remote HEAD (github): `be6bf478dd6f2e63bea81a646bc8b613dbca9935` — **local == remote** (verified via `git ls-remote github escha-w2-prefill` and `git rev-parse remotes/github/escha-w2-prefill`)
- Merge-base with `dflash2`: `ba27edad2a84ff045a556df06661e821285c2fab`
- Worktree status: clean except untracked local aids `results/` and `weights/` (documented; never committed)
- .gitignore covers `*.gguf`, `*.so`, `*.log`, build dirs (build-*); local aids stay untracked

## 2. Promoted implementation (EXP-04 Stage 2 mixed accumulator)

- Promotion commit: `ace024e72` ("PROMOTE Stage 2 mixed accumulator as default prefill control")
- Current `escha-moe.cu` at HEAD: template `FP16_ACC` (`ggml/src/ggml-cuda/escha-moe.cu:958`), applied at lines 1130/1157
- Diff `ace024e72..be6bf478d` on `escha-moe.cu`: only the env-gated `ESCHA_CAPTURE_DST_DIR` debug-capture hook (Stage-3 leftover, `getenv`-gated, inert unless env set; never in timed runs). No functional kernel change.
- `ESCHA_MMA_MIXEDACC_EXPERIMENT` gate removed at promotion (policy is now the default path); no occurrences of the gate string in tracked source.
- EXP-06 (BM64 down-proj candidate): **rejected and reverted** (`eb6679159` "exp06: reject BM64 down-proj tile candidate and restore promoted Stage 2"); `git grep` for exp06/BM64 in tracked CUDA/C++ source: **empty** — no candidate route remains.

## 3. CUDA build configuration (canonical control build)

- Build dir: `build-cuda-exp04-stage2-control` (frozen reference; EXP-04 Phase 2 noise-run control binary sha `34df37036f2ca1fe…` / lib `5bea9eb9d9f36254…`)
- CMakeCache: `CMAKE_BUILD_TYPE=Release`, `CMAKE_CUDA_ARCHITECTURES=120`, `CMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc`, `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, `GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_KVARN=ON`, `GGML_NATIVE=OFF`, `GGML_CUDA_FORCE_CUBLAS=OFF`, `GGML_CUDA_FORCE_MMQ=OFF`, `GGML_CUDA_FA_ALL_QUANTS=OFF`
- Toolchain: nvcc 13.0.88 (cuda_13.0.r13.0), cmake 3.28.3, ninja 1.11.1, gcc 13.3.0
- Driver: 610.88; GPU: NVIDIA GeForce RTX 5090, 32607 MiB (SM 12.0)
- New frozen build for BASE-01: `build-cuda-base01` from HEAD `be6bf478d` (same flags), built 2026-09-01 — see `bench/FROZEN-BINARY.md`

## 4. Historical canonical contract (recorded, from EXP-04 Phase 2 run.log)

- Command: `llama-bench -m <model> --prompt-tokens-file <shared-2048.ids> -p 2048 -n 0 -ngl 99 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json`
- Shared IDs: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids` (sha256 `695c3609bc35a32003a23be3ba1fbacc16cc94955548c2e855e91661c3f62350`)
- Throughput convention: `2048 / measured_prompt_seconds`
- Host: Linux WSL2 6.6.87.2-microsoft-standard; GPU RTX 5090 610.88 32607 MiB; clocks 2887 MHz GPU / ~13801-14001 MHz mem; temp 46C
- Historical ESCHA (promoted Stage 2 candidate median): **2355.9 tok/s** (paired noise run; 2155.2 control)
- Historical IQ3 LowGPU (P-ARCH-23F): avg **614.736 ms / 3338.76 tok/s** (samples 593.039/595.284/655.886 ms) on build `0b035b3a2` (pre-promotion preview build) — the "3600 tok/s" figure exists only as a stretch target in `docs/escha-w2-prefill-next-plan.md` (§2), with no recorded measurement.

## 5. Notes / risks

- GPU: `nvidia-smi` reports 21232 MiB used with no compute processes — stale `vmwp` (WSL VM) reservation (~19.7 GB by `vmwp`, `dwm` 0.65 GB, webview2 0.2 GB). Expected to be reclaimable when a new CUDA context allocates; verify at benchmark start that both models are fully GPU-resident with headroom.
- WSL profiling limitation (prior evidence): occupancy counters unavailable (`ERR_NVGPUCTRPERM`); use CUDA-event / ESCHA_PROFILE symmetric hooks for attribution.
- EXP-04 Phase 2 noise protocol precedent: 9 paired runs, paired-log G analysis; host CV >2% documented; paired-log was the pre-authorized fallback.
