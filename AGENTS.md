# AGENTS.md

This file gives code assistants local context for BeeLlama.cpp. The local tree is
the source of truth for behavior; use `tmp/upstream-llama.cpp` only as the
architectural reference when rebasing fork features.

## What This Is

BeeLlama.cpp is Anbeeld's llama.cpp fork. The v0.4.0 fork surface is intentionally
small:

- Upstream speculative decoding, including `draft-dflash`, `draft-mtp`,
  EAGLE3, and n-gram modes.
- KVarN target KV-cache compression for Qwen3.6 and Gemma 4, selected with
  `kvarn2`, `kvarn3`, `kvarn4`, `kvarn5`, `kvarn6`, or `kvarn8`.
- Standard low-bit KV cache formats `q2_0`, `q2_1`, `q3_0`, `q3_1`,
  `q6_0`, and `q6_1`. Bee's cache-facing `q2_0` uses the internal enum
  `GGML_TYPE_Q2_0S` so it cannot collide with upstream's serialized Q2_0 weight
  format.
- A profit-only adaptive draft-max controller for DFlash.
- Reasoning-loop detection and the opted-in realtime
  `/v1/chat/completions/control` endpoint.
- INI presets and KLD measurement support in `llama-perplexity`.

DFlash GGUFs must use upstream's `dflash` architecture, metadata, and tensor
names.

TurboQuant/TCQ, TQ3_1S/TQ4_1S, DDTree, CopySpec, the fork DFlash ring/tape and
reduced-verifier paths, the fringe controller, and their arguments and
environment variables were removed in v0.4.0. Do not reintroduce those systems
as compatibility code. The old cache names redirect to same-width KVarN presets.
Use upstream's `draft-dflash` name for the DFlash speculative type; the bare
`dflash` alias was removed in v0.4.0 and now errors.

## Build

```bash
# Linux CUDA
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Windows MSVC + CUDA
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON ^
  -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

# macOS Metal
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The default CUDA FlashAttention build contains 50 standard vector pairs and 15
balanced KVarN fast-decode pairs. The standard quant matrix follows the same
bit-pair rules as KVarN and adds homogeneous F16/F16 and BF16/BF16 tail pairs.
`GGML_CUDA_FA_ALL_QUANTS=ON` expands those to
169 standard pairs and all 36 ordered KVarN bit pairs.
`GGML_CUDA_KVARN=OFF` omits all dedicated CUDA KVarN kernels and templates.
`GGML_CUDA_FA_HALF_QUANTS` no longer exists. Valid KVarN pairs outside the fast
matrix use descriptor-native MMA fallback.

Use `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX 3090 and `89` for RTX 4090 when
the build host cannot detect the target GPU.

On Windows hosts matching CUDA 13.1 and compute capability 8.6, prefer:

```powershell
powershell -File scripts/build-win-cuda-13.1-sm_86.ps1 -AllTests
powershell -File scripts/build-win-cuda-13.1-sm_86-default.ps1 -AllTests
powershell -File scripts/build-win-vulkan.ps1 -AllTests
```

The first CUDA script compiles the expanded quant matrix; the `-default`
variant compiles the default pair matrix. The Vulkan script requires a Vulkan
SDK. For other hardware or toolkits, adapt the architecture, toolkit, and
build-name parameters instead of reusing the `sm_86` artifact names.

Key binaries are `llama-server`, `llama-cli`, `llama-bench`, and
`llama-perplexity` under the configured build directory's `bin` folder.

## Architecture

### Main Directories

- `ggml/` - tensor library, quantization, and CPU/GPU backends.
- `src/` - model loading, contexts, graphs, and memory.
- `src/models/` - model-specific graph builders.
- `common/` - arguments, sampling, presets, and upstream speculative decoding.
- `tools/server/` - HTTP API, slots, speculative scheduling, and Bee server
  extensions.
- `include/llama.h` - public C API.

### Fork-Specific Files

- `src/llama-kvarn.cpp` / `.h` - KVarN descriptors, presets, and validation.
- `src/llama-kv-cache-kvarn.cpp` / `.h` - KVarN memory and state handling.
- `ggml/src/ggml-cuda/kvarn.cu` / `.cuh` - shared CUDA/HIP KVarN store and
  materialization operations.
- `ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu` and
  `fattn-kvarn-portable.cuh` - optimized CUDA and portable CUDA/HIP direct
  KVarN attention.
- `ggml/src/ggml-vulkan/vulkan-shaders/kvarn_store.comp` and
  `kvarn_materialize.comp`, `kvarn_wht.comp`, and `kvarn_flash_attn.comp` -
  Vulkan KVarN storage, fallback materialization, transforms, and direct
  attention shaders.
- `tools/server/server-adaptive-dm.h` - profit adaptive draft-max controller.
- `tools/server/server-loop-guard.cpp` / `.h` - reasoning loop detection.
### Key Docs

- `docs/beellama-features.md` - fork feature and compatibility matrix.
- `docs/beellama-args.md` - Bee arguments, aliases, and removals.
- `docs/quickstart-qwen36-dflash.md` - Qwen3.6 DFlash guide.
- `docs/quickstart-gemma-4-31b-dflash.md` - Gemma 4 DFlash guide.
- `docs/preset.md` - INI preset format.

### Invariants

- KVarN is target-context only. Draft and auxiliary contexts use normal cache
  types.
- CUDA, CPU, Vulkan, and HIP/ROCm consume KVarN records directly in native
  attention paths. Vulkan native attention requires shader Int64 and
  buffer-device-address support. Materialization is an explicit fallback, not
  the normal route for these backends.
- Unsupported KVarN placements fail closed or use the explicit
  bit-width-matched fallback path; they must not silently reinterpret records.
- Custom CUDA helpers are resolved through
  `ggml_backend_cuda_reg_get_proc_address`.
- DFlash scheduling, checkpoints, verification, and multi-GPU behavior belong
  to upstream. Bee extensions must use upstream task, sampler, and checkpoint
  APIs rather than restoring fork-private verifier state.
- Benchmark claims require the exact model files, command, prompt, sampling
  settings, hardware, and commit ID.

## Test and Benchmark

```bash
# Unit and regression tests
ctest --test-dir build --output-on-failure

# KVarN quality at the intended serving cadence
build/bin/llama-perplexity -m model.gguf -f test.txt -c 4096 -b 512 -ub 256

# Decode speed
build/bin/llama-bench -m model.gguf -p 0 -n 64 -t 1

# Upstream DFlash with recommended standard q cache
build/bin/llama-server -m target.gguf \
  --spec-type draft-dflash \
  --spec-draft-model drafter.gguf \
  --spec-draft-n-max 8 \
  --flash-attn on --cache-type-k q5_0 --cache-type-v q4_1 \
  --port 8080
```

KLD comparisons use matching `-b` and `-ub` values for the baseline and
candidate. Record both values with every result.

## Git Conventions

- Keep fork-specific changes small and aligned with current upstream
  abstractions.
- Do not treat old benchmark notes as current evidence without rerunning them.
- Do not commit unless the user explicitly asks.


## Escha Support Operating Rules

This project extends BeeLlama with **first-class Escha support**. It is not a
private artifact-specific runtime, and it must stay a general-purpose Qwen/GGUF
engine.

- **Preserve standard BeeLlama/Qwen behavior.** Standard Qwen3.x GGUFs must keep
  original BeeLlama semantics exactly. Never let Escha-specific decoding leak
  into standard models.
- **All Escha-specific behavior must be narrowly metadata-gated.** Gate on
  reliable Escha metadata such as `qwen35.escha.version` (i.e. `escha_version`),
  never on filenames, model names, or content heuristics.
- **No filename hacks.** **No artifact-name conditionals.**
  **No benchmark-specific runtime behavior.** Runtime behavior must not change
  based on which benchmark is being run or which artifact file is loaded.
- **Use fresh builds after semantic/runtime changes.** Rebuild (ideally in a new
  build dir) and re-verify before scoring; never reuse a stale binary for a
  certified result.
- **Separate semantic correctness gates from performance-development gates.**
  A semantic/quality change must pass its correctness gate before being
  evaluated for performance.
- **Full 75-case certification is for milestones, not ordinary iterations.**
  Use lightweight probes during iteration; run the full medium 5-pack (or full
  8-pack) certification only at milestones.
- **Record decisive experiments in the experiment ledger.**
  (`docs/escha-prefill-experiment-ledger.md` — the canonical ledger; a GBrain
  mirror is kept in sync.)
- **Update `docs/current-state.md` whenever project state materially changes.**
  Keep it concise and current, not a full history.
- **Raw evidence should remain available and traceable.** Keep result JSONs,
  hashes, runtime/commit identity, and launch config with every certified run.
- **Parallel investigation:** DeepSeek V4 Flash workers via Hermes/Nous Portal
  may be used for parallel investigation when useful. The primary agent must
  verify decisive findings before integrating them.
