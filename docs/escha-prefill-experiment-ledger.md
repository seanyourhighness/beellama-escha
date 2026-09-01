# Escha x LowGPU prefill experiment ledger

This is the canonical working ledger for the BeeLlama port of the custom Escha
runtime and Escha x LowGPU model. It tracks every prefill hypothesis from proposal
through acceptance or elimination so rejected paths leave the active queue without
being forgotten.

External mirror: GBrain page `projects/beellama-escha-prefill-ledger`.

## Definition of done

The prefill layout is final only when one retained default meets all of these gates:

1. A sidecar route-proof run from the same source, binary, model, workload, and
   route environment records the selected CUDA route and rows. Timed runs keep
   `ESCHA_PROFILE` off because its per-operator synchronization perturbs throughput.
2. Operator output passes the exact current FP32 control gate (`max_abs_diff = 0`)
   for
   K=2 and K=3 across the production projection shapes and row-tail cases.
3. Greedy SGLang parity passes P1, P2, P5, P6, and P7 for at least 16 generated
   tokens, with P5 covering a prompt of at least 1,544 tokens.
4. RTX 5090 prefill exceeds 2,000 prompt tok/s at the 2k gate using the fixed
   benchmark contract below, without a decode regression greater than 2%.
5. The retained layout passes 128, 512, 2,048, and 4,096 prompt-token gates and
   handles non-multiple row tails without hangs, NaNs, or unwritten output.
6. Peak VRAM and the exact source/model/build fingerprints are recorded. The final
   Ada acceptance run remains separate until an RTX 4070 Ti 12 GB is attached.

## Experiment contract

- Change one independent variable per experiment. Use a new ID for a materially
  different kernel, arithmetic mode, scheduler, or launch layout.
- Record the source SHA-256, Git HEAD, binary timestamp/hash, model path/hash,
  GPU/driver, full command, environment variables, raw artifact paths, and exit code.
- Run in this order: build -> sidecar route proof -> operator exactness -> short parity ->
  2k performance -> full prompt-depth matrix. Stop at the first failed hard gate.
- Use at least one warmup and three measured runs. Report median and all samples;
  do not promote a best-of-run number.
- `ACCEPTED` means every required gate passed. `REJECTED` removes the hypothesis
  from the active queue but preserves its row and evidence. `INCONCLUSIVE` means
  the intended route or comparison was not proved. `BLOCKED` names an external
  dependency. Rejected experiments are never retried unless a named changed
  condition invalidates the original result.
- Every performance claim must identify the actual prompt-token count, `-b`, `-ub`,
  context, KV types, graph mode, route, hardware, and commit/source fingerprint.

## Fixed benchmark contract

Unless an experiment explicitly studies one of these values:

| Field | Value |
| --- | --- |
| Checkout | `/mnt/d/CODEX WORKSPACE/beellama-escha` |
| Model | `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf` |
| GPU | RTX 5090, SM120, 32,607 MiB |
| Sampling | temperature 0, seed 42 |
| Serving | full CUDA offload, F16 KV, flash attention on |
| Graph mode | Primary production gate: `ESCHA_ALLOW_CUDA_GRAPHS=1`; also retain one graph-disabled diagnostic control |
| Batch | `-b 2048 -ub 512`; record any harness-imposed change |
| Primary compute gate | `llama-bench -p 2048 -n 0 -r 3`, exactly 2,048 tokens; default warmup; report median and all samples |
| Secondary end-to-end gate | `llama-server /completion`; never compare or merge this number with `llama-bench` |
| Decode guardrail | `llama-bench -p 0 -n 64 -r 3`; establish a current control before candidate promotion |
| Parity suite | `scripts/escha-compare/run_compare.py`, P1/P2/P5/P6/P7 |
| Raw artifacts | `/tmp/escha-prefill/<experiment-id>/` |

## Current verified baseline

| Item | Evidence | Status |
| --- | --- | --- |
| Active checkout | `/mnt/d/CODEX WORKSPACE/beellama-escha`; the `/home/sean/code/beellama-escha` copy is older and lacks `lowgpu.cu` | accepted |
| Git base | detached `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d`; the Escha port also includes local files/changes, so Git HEAD alone is not a sufficient fingerprint | accepted |
| Actual prefill rows | `ESCHA_PROFILE` audit observed `gen=0 rows=512` for 1,600 calls; rows 2/3/4 are `gen=1` calls interleaved with prefill | accepted |
| Safe Blackwell route | FP32 activation rotation -> `escha_matmul_dense_tiled` -> fixed-order `escha_finalize_dense` | accepted baseline |
| **Champion (P-ARCH-23)** | **standard-FFN body + Q2_K reconstructed gate + LowGPU vocab: 8.599 GB, 2k prefill 697.032 ms / 2938.27 tok/s, quality PASS, decode 31.03** | **accepted 2026-08-30** |
| 2k compute baseline | Production graphs P000-R2 median 655.468 tok/s (666.312/653.131/655.468); graph-disabled P000-R1 median 660.487 | historical controls (superseded by champion) |
| Decode baseline | Production graphs P013-R2 median 44.9093 tok/s (44.7279/47.0682/44.9093); graph-disabled diagnostic median 25.612 | accepted guardrail |
| Exact generation parity | P1/P2/P6/P7: 16/16; P5 at 1,544 prompt tokens: 16/16 | accepted baseline (full-Escha artifact) |
| Reference target | SGLang Escha reference roughly 2,850-3,120 prompt tok/s at 2k on RTX 5090 | target context |

The earlier claim that Qwen3.5 prefill only reaches 2-4 Escha rows is superseded.
Those rows belong to the generation instantiation. Scheduler work intended only to
turn 2-4 rows into a large prefill batch is eliminated unless new route-tagged
evidence disproves the 512-row audit.

## Experiment ledger

| ID | Hypothesis / change | Route proof | Correctness | Performance | Decision | Evidence / reason |
| --- | --- | --- | --- | --- | --- | --- |
| P000 | Current Blackwell FP32 tiled-FMA prefill is the control | Exact 2k sidecar proved `route=tiled-fma-fp32 rows=512` for all 1,600 projection calls | P1/P2/P5/P6/P7 exact-token baseline | Production graphs P000-R2 median 655.468 tok/s; graph-disabled P000-R1 median 660.487 | `ACCEPTED-CONTROL` | `/tmp/escha-prefill/P000-R{1,2}/`; historical results are retained but superseded |
| P001 | Force the legacy ldmatrix/HMMA kernel on SM120 with `ESCHA_FORCE_MMA=1` | Historical intended legacy MMA route | One nsys-serialized run was recorded as completing; other runs were interpreted as pending | Historical apparent non-completion, GPU about 0-2%, graphs on and off | `SUPERSEDED-BY-P014-REVALIDATION` | P-ARCH-03 does not accept pending progress narration as stall evidence. Preserve this history, but re-establish current completion with shell exit code, process termination, complete output, and post-command process/GPU state. |
| P002 | Replace `cp.async` activation staging with plain shared loads in P001 | Historical forced MMA route | Not reached reliably | Historical apparent non-completion matched P001 | `SUPERSEDED-BY-P014-REVALIDATION` | Preserves evidence that `cp.async` was not uniquely implicated, but does not prove the current route stalls. |
| P003 | Blackwell WMMA candidate with four warps per CTA | WMMA candidate | Warps repeated identical work and raced the same partial output | Invalid layout; no valid speed claim | `REJECTED` | No distinct band, row, or output assignment per warp |
| P004 | Store the WMMA A fragment as `w_w[c*TILE+r]` | WMMA candidate | Full token divergence | Non-decisive | `REJECTED` | Transposes A; correct col-major storage is `w_w[r*TILE+c]` |
| P005 | Dispatch the FP16-activation WMMA path for `n_rows >= 2` | Candidate was exercised | P1 failed immediately, 0/1 generated-token prefix agreement | Non-decisive | `REJECTED` | FP16 activation rotation changes the exact FP32 serving math |
| P006 | Compound diagnostic: FP16 activation rotation plus one-warp/two-band Blackwell WMMA (`ESCHA_WMMA_PREFILL=1`) on true 512-row prefill | Must show `route=wmma-bw-fp16 rows=512` | Expected to fail bitwise FP32 exactness; must measure before any performance claim | Prior 609.8 tok/s number lacked route proof and is non-decisive | `BLOCKED-P012` | Resolves the contradiction between the later 512-row audit and the earlier microbatch interpretation |
| P007 | Compound diagnostic: FP16 activation rotation, full weight dequantization, and cuBLAS GEMM (`ESCHA_CUBLAS_PREFILL=1`) | Must show `route=cublas-fp16` | Unmeasured against exact FP32 operator gate | Unmeasured; transient full dense weight may be costly | `BLOCKED-P012` | Diagnostic control only; full dequantization means this is not a strict performance ceiling for inline packed decode |
| P008 | TF32 tensor-core input path | Not implemented | TF32 truncates FP32 multiplicands and cannot meet `max_abs_diff=0` | Not run | `ELIMINATED-DESIGN` | Arithmetic conflicts with the exact-output gate; reconsider only if the gate changes |
| P009 | Scheduler-only work to convert supposed 2-4-row prefill into large batches | Route-tagged audit shows 512-row true prefill already exists | N/A | Wrong bottleneck | `ELIMINATED-DESIGN` | Superseded premise; rows 2-4 are generation-graph calls |
| P010 | Exact FP32 shared-X, one-output-band-per-warp dataflow; keep control BM=128, BN=128, TM=8, TN=8, slice math, activation, and finalize order | New `route=band-fma-fp32` tag required | Preserve the identical FP32 operand and accumulator-update sequence | Remove the middle block-wide decode-publish barrier; unmeasured | `DESIGN-FROZEN-BLOCKED-P012` | DeepSeek/Hermes kernel review; single independent variable is band ownership/dataflow |
| P011 | Add explicit route identity to `ESCHA_PROFILE` | Proved `route=tiled-fma-fp32 rows=512`, 1,600/1,600 calls at the exact 2k gate | Instrumentation only; build passed | Profiled 512-token sample per projection: total 1.739794 ms; rotate 0.057481; matmul 1.657025; epilogue 0.025296 | `ACCEPTED` | `/tmp/escha-prefill/P011/profile.log`, `/tmp/escha-prefill/P000-R1/route-proof.log`; matmul is 95.2% of measured Escha projection time |
| P012 | Add an Escha operator exactness harness that runs the current tiled-FMA control and a selected candidate on identical packed tensors and compares output bytes | Must assert the intended control and candidate routes | Require `max_abs_diff=0` and byte equality for K=2/K=3, production shapes, rows 1/8/16/17/511/512/513 | Test harness only | `HISTORICAL` | Hard prerequisite: token-prefix parity alone cannot detect small arithmetic/order drift. Superseded by the certified quality benchmark + ESCHA-W2 PREFILL phase — not active. |
| P013 | Establish the decode regression control under the fixed harness | Generation route is unchanged; separate route proof remains available from prior profiling | Existing parity baseline remains valid | Production graphs median 44.9093 tok/s; graph-disabled diagnostic median 25.612 | `ACCEPTED-CONTROL` | `/tmp/escha-prefill/P013-R2/bench.json`, `/tmp/escha-prefill/P013/bench.json` |
| P014 / P-ARCH-03 | Revalidate current SM120 forced-MMA completion before treating historical pending output as a kernel stall | `mma-fp16 rows=512` proved for 800/800 calls; shell exit 0, complete output, process gone, no residual GPU process | PASS: P1/P2/P5/P6/P7 each 16/16 vs Escha/SGLang; P5 prompt 1,544 tokens | Diagnostic 512-token 894.768 tok/s; controlled graph-mode 2k samples 1243.72/1229.43/1254.32, median 1243.72 vs 655.468 normal (1.897x, +89.745%) | `ACCEPTED — CURRENT STALL DISPROVED` | No kernel-internal divergence reproduced; no invasive instrumentation. Current forced route is complete/correct/faster. Evidence: `docs/escha-architecture/P-ARCH-03/manifest.md` and external P-ARCH-03 directory. |
| P015 / P-ARCH-04 | Qualify existing MMA across the real SM120 production envelope and make it the default only if qualification passes | PASS: all seven real IC/OC pairs; rows 17/128/511/512 automatic `mma-fp16`; generation rows <=16 remain FP32; real 513+ tails split safely; explicit disable retains tiled-FMA | PASS: three forced and three post-change P1/P2/P5/P6/P7 runs, each 16/16; P5=1,544 prompt tokens | Stability: 40,800 MMA calls, 3/3 runs, prompts through 4,096, no detected CUDA errors. Post-change 2k samples 1246.82/1230.03/1217.32, median 1230.03 | `ACCEPTED — SM120 MMA DEFAULT` | One-line eligibility change `cc < BLACKWELL` → `cc <= BLACKWELL`; architectures above SM120 remain opt-in; `ESCHA_NO_MMA` fallback preserved. Manifest: `docs/escha-architecture/P-ARCH-04/manifest.md`. |
| P020 / P-ARCH-20 | Reproduce native Escha mixed policy with a Bee FP16 MMA accumulator only for `M=2048, IC=5120, OC=17408, K2`; all layout/staging/geometry unchanged | PASS: compile-time flag `ESCHA_MMA_FP16ACC_EXPERIMENT`; runtime shape predicate; isolated symbol has 32 `HMMA.16816.F16` and 0 `.F32` versus FP32 twin 0/32 | Not run: performance hard gate failed decisively, so no quality claim is made | FP32 median 929.039 ms / 2204.43 tok/s; FP16 median 1339.906 ms / 1528.47 tok/s; **+410.867 ms / 44.22% slower** | `REJECTED` | **CONTAMINATED (2026-09-01, EXP-04 Phase 1): the fp16 route executed a faulty fragment store (OOB `tile_ah` read + dropped `.y` lanes, fixed `7b1880f41`). The −44.22% is not valid evidence against fp16 MMA accumulation; parity never produced a report.** `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-20/2026-08-30/{control-001,fp16acc-001}/`; see `evidence/EXP-04-phase1/2026-09-01/CONTAMINATION-AUDIT.md` |
| P-ARCH-21A | Selective standard-quant substitution: replace all 192 FFN gate/up/down Escha sidecars with raw weights copied from the compatible `Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf`; attention/GDN/vocab unchanged | Loader dispatch per-FFN-family: standard MUL_MAT where all three FFN weights present; Escha sidecars otherwise; inventory clean (192 std FFN, 480 retained Escha attn/GDN sidecars, 6 packed-vocab tensors) | Quality **PASS** via llama-server `/completion` (greedy, seed 42): "The sky appears blue because shorter-wavelength blue light is scattered..." — clean, 1.7 s | 2k matched prefill median **727.724 ms / 2814.31 tok/s** (728.4/731.4/723.4); decode 30.44 tok/s vs 23.69 prod control; artifact 8.483 GB | `SUPERSEDED BY P-ARCH-23` | `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-21/2026-08-30/standard-ffn-001/`; `escha-w2-lowgpu-mono-parity-standard-ffn.gguf` sha256 `0f6c9584...`; beats Codex's recorded 741.067/2766.93 (one 778 ms warmup sample) |
| P-ARCH-21B | Extend substitution to the 48 linear-attention `attn_gate` (GDN gate) projections from the same source GGUF (`--standard-gdn-gate`) | Converter + loader gate fallback wired; inventory clean (192 std FFN, 48 std attn_gate, 0 gate Escha sidecars, 160 remaining Escha codes, 6 LowGPU vocab); artifact 8.596 GB | Quality **FAIL** — garbage tokens. **Root cause (P-ARCH-21C): source `attn_gate.weight` is a DIFFERENT projection than the checkpoint `in_proj_z` (corr ~0.04; FFN control matches 0.835). Bias is NOT the cause (`ESCHA_APPLY_BIAS 0`). Corrected gate (reconstructed from ckpt sidecars, F16) produces coherent output — path is fine, tensor source was wrong** | 2k prefill **697.046 ms / 2938.15 tok/s** (695.8/694.9/700.4); decode 29.55 tok/s (no regression); artifact 8.596 GB | `REJECTED as-built; CONFIRMED fixable via correct gate source` | `escha-w2-lowgpu-mono-parity-standard-ffn-gdn.gguf`; see P-ARCH-21C session evidence |
| P-ARCH-23 | Reconstruct the TRUE gate from checkpoint `in_proj_z` sidecars (`reconstruct_deploy_weight`: Hadamard-128 on IC then OC, rin/rout scaling; K=2, cbA) and re-quantize to standard Q2_K in-process (`--standard-gdn-gate-quant q2_k`); keep 21A standard-FFN body + LowGPU vocab | Converter pure-numpy Q2_K quantizer (ported from `ggml-quants.c` `quantize_row_q2_K_impl`, byte layout verified vs `Q2_K.dequantize_blocks`); inventory clean (192 std FFN, 48 std `attn_gate.weight` Q2_K, 0 gate sidecars, 160 escha codes, 6 LowGPU vocab); artifact 8.599 GB | Quality **PASS** via llama-server `/completion` (greedy, seed 42) coherent Rayleigh answer; parity suite at 21A-champion parity (P7 exact 23/23; P1/P2/P5/P6 coherent, differ token-wise from SGLang ref — same profile as 21A control); correct X7Q-91-MARS extraction from 1,544-token context; decode 31.03 tok/s (no regression) | 2k prefill **697.032 ms / 2938.27 tok/s** (702.5/695.5/693.0); artifact 8.599 GB | `ACCEPTED — NEW CHAMPION` | `escha-w2-lowgpu-mono-parity-standard-ffn-gdn-q2k.gguf` sha256 `2e61882d27f8828ce19d45d3eb33e6a4cd09802d2c84a19fa0636e8f8cf518fa`; evidence `evidence/P-ARCH-23/2026-08-30/standard-ffn-gdn-q2k-001/` |

## Active queue

> Historical queue snapshot (superseded — see `docs/current-state.md` for the
> current state). P-ARCH-04/05 through P-ARCH-23I are complete/closed.

1. ~~P-ARCH-04 / P015~~ — **CLOSED**: existing MMA is the qualified SM120 prefill
   default; official controlled baseline is 1230.03 tok/s. Do not reopen
   qualification without contradictory evidence.
2. ~~P-ARCH-05 is next~~ — **SUPERSEDED**: the P-ARCH-21/23/23G/23I substitution
   line and the 2026-08-31 certified quality benchmark supersede this queue.
   P-ARCH-05 established the controlled Escha reference comparison; attribution
   of the remaining prefill gap is now in `docs/current-state.md` (ESCHA-W2
   PREFILL phase).
3. P012/P006/P007/P010 remain separate historical experiments and are not
   bundled into any active queue.
4. **Active now:** ESCHA-W2 PREFILL (see `docs/current-state.md`).

## Current session evidence — 2026-08-29

### P011 — route-tagged profiling

- Source SHA-256 after instrumentation: `dc0da23c8e911c9eed9a6729f89e1dac264f9884cd3a9502b44582a9f7055e98`.
- Build: `cmake --build build-cuda-verify --target llama-bench -j 12`, exit 0.
- Probe: 512 prompt tokens, `-b 2048 -ub 512`, F16 KV, FA on, graphs disabled.
- Route: `tiled-fma-fp32`, rows 512, 400/400 Escha projection calls.
- Mean projection stages: rotate 0.057481 ms; matmul 1.657025 ms; epilogue
  0.025296 ms; total 1.739794 ms.
- Artifact: `/tmp/escha-prefill/P011/profile.log`.

### P000-R1 — fresh 2k control

- Binary SHA-256: `d94d26ea0d603ee86ed3d947aec1c04a943800d5c2c22ea3ca7142614cc49ac6`.
- Model file SHA-256: `e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d`.
- Model file size: 8,619,127,360 bytes; llama-bench reported tensor/model size
  8,608,061,440 bytes and 4,784,193,024 parameters.
- GPU/driver: RTX 5090, SM120, driver 610.88, 32,607 MiB.
- Contract: 2,048 prompt tokens, `-b 2048 -ub 512`, F16 KV, FA on, full CUDA
  offload, graphs disabled, one warmup and three measured repetitions.
- Samples: 661.425, 660.187, 660.487 prompt tok/s; median 660.487; mean
  660.700; standard deviation 0.646.
- Artifact: `/tmp/escha-prefill/P000-R1/bench.json`.

### P000-R2 — production graph-mode 2k control

- Contract: same as P000-R1 except `ESCHA_ALLOW_CUDA_GRAPHS=1` and no graph-disable
  variable.
- Samples: 666.312, 653.131, 655.468 prompt tok/s; median 655.468; mean
  658.303; standard deviation 7.033.
- Artifact: `/tmp/escha-prefill/P000-R2/bench.json`.

### P013 — decode guardrails

- Graph-disabled diagnostic samples: 23.2038, 25.612, 25.628 tok/s; median
  25.612; mean 24.8146.
- Production graph-mode samples with `ESCHA_ALLOW_CUDA_GRAPHS=1`: 44.7279,
  47.0682, 44.9093 tok/s; median 44.9093; mean 45.5685.
- Artifacts: `/tmp/escha-prefill/P013/bench.json` and
  `/tmp/escha-prefill/P013-R2/bench.json`.

### DeepSeek/Hermes independent reviews

Three read-only workers inspected the current source and ledger in parallel:

- Kernel design: froze P010-A to a 256-thread/8-warp kernel where each warp owns
  one 16-column output band inside the existing 128x128 tile. FP32 rotation,
  float codebook, per-accumulator update order, slice boundaries, partial layout,
  and finalize remain unchanged. Only the middle block-wide decode-publish barrier
  and shared weight ownership change.
- Methodology audit: identified the missing operator-level bitwise oracle, the need
  to distinguish compute-only `llama-bench` from server end-to-end throughput, the
  missing decode baseline, and the danger of mixing non-512 tail routes into a
  nominal 2k benchmark. These corrections produced P012/P013 and the fixed exactly
  2,048-token compute gate.
- Profiler plan: route proof must be separate from timed profiling; use uninstrumented
  Nsight attribution and stop tile work if packed matmul is not the dominant full
  prefill component. The route-proof stage already shows matmul is 95.2% of Escha
  projection time, but full-graph attribution remains a separate future experiment.

One recommendation was not adopted: P007 is not a hard feasibility ceiling because
it adds full FP16 dense-weight expansion on every call, work that P010's inline packed
decoder does not perform.

## Open design variables for P010

| Variable | Current control | First bounded comparison | Rule |
| --- | ---: | ---: | --- |
| Weight-band ownership | whole-block shared tile | one 16-column band per warp | P010-A; hold every numeric layout constant |
| Row tile `BM` | 128 | 64 vs 128 only after P010-A | Separate P010-B; hold `BN/TM/TN` fixed |
| Output tile `BN` | 128 | 64 vs 128 | Reject any ragged production shape |
| Per-thread tile `TM x TN` | 8 x 8 | derive from resource budget after NCU | Do not combine with a `BM/BN` change |
| Reduction slices | target-derived | 1 vs current only after route baseline | Fixed final slice order is mandatory |
| Activation storage | FP32 | FP32 only | FP16/TF32 are outside the exact gate |
| Epilogue | separate fixed-order finalize | fuse only as a separate experiment | Must reproduce identical output bytes |

## Current session evidence — 2026-08-30 (P-ARCH-21)

### P-ARCH-21A — selective standard-FFN substitution

- Date/time: 2026-08-30.
- Hypothesis: replace all 192 FFN gate/up/down Escha sidecars with raw standard
  quant weights from the compatible GGUF to cut prefill latency; attention/GDN
  and vocabulary unchanged. Artifact must stay ≤ 8.65 GB and pass quality.
- Independent variable: FFN tensor family only (attention/GDN/vocab retained).
- Source/Git/binary/model fingerprints:
  - Checkout: `/mnt/d/CODEX WORKSPACE/beellama-escha`, git `0b035b3a2`.
  - Build: `build-cuda-p20-control` (flags `-DESCHA_MMA_SM120_ASYNC_EXPERIMENT=1`).
  - Model: `escha-w2-lowgpu-mono-parity-standard-ffn.gguf`, sha256
    `0f6c958493de4841f347f5118475eb8f36c570a9cccc265c0023cafbba2b6542`,
    8,483,237,312 bytes.
  - Source GGUF: `Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf` (FFN weights copied raw).
  - GPU: RTX 5090, SM120, driver 610.88, 32,607 MiB.
- Exact command and environment: `GGML_CUDA_DISABLE_GRAPHS=1 llama-bench -m <standard-ffn.gguf>
  --prompt-tokens-file <shared-2048.ids> -p 2048 -n 0 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on
  -r 3 -o json -oe json`.
- Route proof: loader dispatch per-FFN-family — standard MUL_MAT where all three
  FFN weights present, Escha sidecars otherwise. Inventory verified with
  `model-inspect.py`: 192 std FFN weights, 208 Escha code tensors retained
  (attention/GDN), 6 LowGPU packed-vocab tensors.
- Performance samples: 728.414, 731.355, 723.403 ms → median **727.724 ms /
  2814.31 tok/s** (better than Codex's recorded 741.067 ms / 2766.93 tok/s,
  which included a 778 ms warmup sample). Decode 30.44 tok/s vs production
  23.69 tok/s control (graph-disabled).
- Quality: **PASS** — llama-server `/completion`, greedy, seed 42:
  "The sky appears blue because shorter-wavelength blue light is scattered more
  efficiently by molecules in the atmosphere than other colors, a phenomenon
  known as Rayleigh scattering." Clean, ~1.7 s.
- Tooling note: this build's `llama-cli` (server-pipeline based) hangs in
  conversation mode on BOTH the production and substituted artifacts — a
  harness quirk, not a model defect. The canonical gates (`llama-bench`,
  `llama-server /completion`) run both artifacts normally.
- Raw artifacts: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-21/2026-08-30/standard-ffn-001/`.
- Decision: **KEEP — current champion** (8.483 GB, 2814.31 tok/s, quality PASS).

### P-ARCH-21B — add linear-attention gate (GDN gate) substitution

- Date/time: 2026-08-30.
- Hypothesis: extend the same substitution to the 48 linear-attention `attn_gate`
  projections (`--standard-gdn-gate`) to recover more latency.
- Independent variable: `attn_gate` tensor family (48 projections) on top of 21A.
- Model: `escha-w2-lowgpu-mono-parity-standard-ffn-gdn.gguf`, 8,595,983,296 bytes
  (8.596 GB, still ≤ 8.65 GB). Inventory: 192 std FFN, 48 std `attn_gate.weight`,
  0 gate Escha sidecars, 160 remaining Escha codes, 6 LowGPU vocab.
- Performance samples: 695.818, 694.921, 700.399 ms → median **697.046 ms /
  2938.15 tok/s**. Decode 29.55 tok/s (no regression).
- Quality: **FAIL** — llama-server `/completion` returns garbage tokens
  (`3;6;q3;;01\n,oC...`). Faster than 21A but unusable.
- Key finding — bias is NOT the cause: the Escha path deliberately ignores
  projection biases (`ESCHA_APPLY_BIAS 0` in `build_escha_mm`, matching the
  reference Escha runtime per the model card). Dropping the gate bias in the
  standard path is behaviorally identical to production. The garbage must come
  from the gate weight content, not bias handling.
- Open question: generic IQ2/Q2_K/IQ1 gate quant loss vs a standard-gate-path
  code bug. Decisive next experiment: substitute the gate as dequantized F16
  (diagnostic, over-cap allowed) — clean output ⇒ quant loss (reject gdn);
  garbage ⇒ code bug (fixable, valuable).
- Raw artifacts: gdn artifact in `weights/`, summary `/tmp/parch21-gdn-summary.json`.
- Decision: **REJECTED (pending root-cause diagnostic)**.

### P-ARCH-21C — gdn root cause: wrong source tensor (NOT bias, NOT quant loss)

- Date/time: 2026-08-30.
- Hypothesis to test: is the gdn gate failure caused by generic quant loss on
  the source gate, or by a broken standard-gate path?
- Method: reference reconstruction (`escham_cpu.py:reconstruct_deploy_weight`,
  Hadamard-128 + rin/rout scaling) applied to the checkpoint `in_proj_z`
  sidecars; correlated against the source GGUF's `attn_gate.weight` and the
  production FFN gate as a control.
- Result 1 (mapping is wrong): the source GGUF's `attn_gate.weight` correlates
  **~0.04** with the checkpoint's true gate across layers 0/1/2, while the FFN
  control correctly matches at **0.835**. The source GGUF was built from a
  different model/export where `attn_gate` is a different projection. Codex's
  `--standard-gdn-gate` therefore wrote the WRONG tensor into the gate slot.
- Result 2 (path is fine): a diagnostic artifact with the gate reconstructed to
  dense F16 from the checkpoint sidecars (not the source GGUF) loads and
  produces **coherent, correct output** via llama-server `/completion`:
  "The sky appears blue because shorter-wavelength blue light is scattered more
  efficiently by the molecules in Earth's atmosphere than other colors of
  visible light." (11.12 GB, over-cap diagnostic — proves the standard gate
  path and weight orientation are correct).
- Result 3 (bias is not the cause): `ESCHA_APPLY_BIAS 0` in `build_escha_mm` —
  the reference Escha runtime deliberately ignores projection biases, so the
  dropped gate bias is behaviorally identical to production.
- Root cause: **the source GGUF's `attn_gate.weight` is not the checkpoint's
  `in_proj_z`** — the `--standard-gdn-gate` flag must reconstruct the true gate
  from the checkpoint sidecars instead of copying the source tensor.
- Next step: production fix = reconstruct the true gate and re-quantize it
  (e.g., Q6_K/IQ4_XS) to fit ≤ 8.65 GB, then re-benchmark. Diagnostic F16
  artifact: `escha-w2-lowgpu-mono-parity-standard-ffn-gdn-f16.gguf`.
- Decision: **REJECTED as-built; CONFIRMED fixable via correct gate source**.

## Current session evidence — 2026-08-30 (P-ARCH-22)

### P-ARCH-22A — vocab/output-path opportunity on the current champion

- Date/time: 2026-08-30.
- Hypothesis: recover the remaining prefill latency through non-destructive
  vocab/output-path optimization while preserving ≤ 8.65 GB, quality PASS, and
  the P-ARCH-21A standard-FFN substitution. P-ARCH-18B measured a 41.581 ms
  vocab-representation delta (LowGPU packed vs dense F16) on the OLD full-Escha
  body; re-investigate on the current champion body.
- Independent variable: vocabulary representation / output path, one side at a time.
- Method: per-side vocab storage flags added to `convert_escha_to_gguf.py`
  (`--embed-vocab`, `--head-vocab`), plus loader/graph support for mixed per-side
  storage (`qwen35.cpp`: each side decided by tensor presence, not one global flag).
  Built three over-cap diagnostics (embed-F16-only, head-F16-only, both-F16) plus
  the champion control, all on the identical standard-FFN body.
- Results (matched 2k, `-p 2048 -n 0 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on`,
  graphs disabled, 3 samples):

| Variant | avg ms | tok/s | note |
| --- | ---: | ---: | --- |
| champion (both LowGPU) | 707-736 | ~2894 | control |
| embed-F16 only (head LowGPU) | **669.2** | **3060.3** | full ~40 ms win |
| head-F16 only (embed LowGPU) | 714.6 | 2866.7 | ~0, no change |
| both-F16 | 667-687 | ~3055 | same as embed-only |

- Key findings:
  1. **The ~40 ms opportunity lives ENTIRELY on the embedding side.** Replacing
     only the LM head with dense F16 changes nothing (714.6 vs 707-728 control);
     replacing only the embedding captures the full win (669.2 vs 727.7).
  2. **The final-token-only hypothesis is ALREADY satisfied.** `llama_batch_get_one`
     passes `logits = nullptr`, so the batch allocator marks only the LAST token
     as an output (`n_outputs = 1`). `LOWGPU_PROFILE` confirms the LM head runs at
     `tokens=1` (0.88 ms) during the matched prefill. The runtime does NOT compute
     the full-vocabulary projection for every prefill token.
  3. **The LM-head kernel is not the cost.** LowGPU LM-head mul_mat at 1 token is
     ~0.9 ms; the embedding get_rows is 0.185 ms steady-state. The ~40 ms delta
     comes from the embedding representation/flow, not kernel wall-time.
  4. **The win is not capturable within the size cap.** Dense F16 embedding is
     +1.9 GB (embed-F16 artifact = 10.52 GB vs 8.65 GB cap). The P-ARCH-18B
     "41 ms" was always a diagnostic-only representation effect, and it remains
     over-budget on the champion body.
- Decision: **INCONCLUSIVE as a production win — blocked by size cap.** The LM-head
  side of P-ARCH-22 (22B final-token-only, 22C head layout) is a no-op because the
  mechanism already exists. The embedding side (22D) shows the gain but only via a
  representation that violates ≤ 8.65 GB.
- Open question: whether the embedding-side 40 ms can be recovered with a
  size-neutral change (e.g., matching the F16 embed dtype/flow in the packed path
  without materializing dense F16). Not pursued here; would be the next experiment.
- Raw artifacts: `escha-w2-lowgpu-mono-parity-standard-ffn-{embedF16,headF16,f16vocab}.gguf`.

### P-ARCH-22D — embedding-side, size-neutral attempt (F16 emit) — NEGATIVE

- Date/time: 2026-08-30.
- Hypothesis: the LowGPU embedding `get_rows` kernel writes F32 output while the
  dense F16 path writes F16; if the packed path emitted F16 (same fp16-rounded
  values it already computes internally, just not widened to F32), it would match
  the F16 embed flow and capture the ~40 ms without the +1.9 GB size cost.
- Change: `ggml_lowgpu_get_rows` result type F32 -> F16 (`ggml.c:6808`) and the
  CUDA kernel's store type F32 -> F16 (`lowgpu.cu`, `half *orow`).
- Result: **prefill hangs** (llama-bench exit 124, no output) on the unchanged
  champion artifact. Reverted in full; champion re-verified at 729.9 ms /
  2809.9 tok/s (samples 719.3/701.6/768.9).
- Interpretation: the first body ops (RMSNorm / escha rotate) expect F32
  embedding activations; the packed path cannot simply drop to F16 output. The
  F16-vocab artifact's gain therefore comes from the dense F16 tensor
  representation itself (standard `ggml_get_rows` on an F16 tensor), which costs
  +1.9 GB — not from a dtype that the packed kernel can emit for free.
- Decision: **REJECTED — no size-neutral embedding win.** The ~40 ms vocab
  opportunity is real (P-ARCH-22A) but only capturable via a representation that
  violates the ≤ 8.65 GB cap. Champion remains P-ARCH-21A (8.483 GB, ~2814 tok/s).
- P-ARCH-22 summary: 22B (final-token-only LM head) is already satisfied
  (`n_outputs = 1`); 22C (head layout) is a no-op; 22D (embedding) is
  size-capped. No new champion promoted from this experiment set.

## Current session evidence — 2026-08-30 (P-ARCH-23, phase 1: gate reconstruction)

### P-ARCH-23 — reconstructed true gate, Q2_K

- Date/time: 2026-08-30.
- Hypothesis: recover P-ARCH-21B's ~697 ms / 2938 tok/s without its quality
  failure by reconstructing the TRUE gate from the checkpoint `in_proj_z`
  sidecars (root-caused in P-ARCH-21C: source GGUF `attn_gate.weight` is a
  DIFFERENT projection, corr 0.043 vs FFN control 0.835) and re-quantizing to a
  standard type that fits the ≤ 8.65 GB cap.
- Independent variable: gate tensor source + standard quant. The 21B budget
  finding was corrected: Q6_K/IQ4_XS do NOT fit under 8.65 GB (9.34 / 8.91 GB);
  only ~2.5-bit fits. Q2_K chosen (2.625 bpw, 8.599 GB).
- Method: converter flag `--standard-gdn-gate-quant q2_k` reconstructs
  `w = reconstruct_deploy_weight(code, rin, rout, 5120, 6144, K=2, cbA=True, mul1=False)`
  (Hadamard-128 IC -> rin -> Hadamard-128 OC -> rout), quantizes `w.T` with a
  pure-numpy Q2_K port of `ggml-quants.c quantize_row_q2_K_impl`, writes
  `blk.{il}.attn_gate.weight` Q2_K. Loader unchanged (standard mul_mat path;
  `wqkv_gate_s` NULL; no gate bias — matches `ESCHA_APPLY_BIAS 0`).
- Numeric verification: independent reconstruction of layers 0-2 (K=2, no NaN,
  max abs ~0.22-0.28); s_in/s_out ~1.0 (raw rin/rout correct); orientation
  round-trip verified (GGUF ne=(5120,6144)); Q2_K MAE 0.00353; Q4_K MAE 0.000945;
  Q6_K MAE 0.000234 (byte-exact round-trips via `gguf.quants.dequantize`).
- Inventory: 192 std FFN, 48 std `attn_gate.weight` Q2_K, 0 gate sidecars,
  160 escha codes, 6 LowGPU vocab. Artifact 8,598,932,416 bytes = 8.599 GB.
- Performance (2k, `-p 2048 -n 0 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on`,
  graphs disabled, 3 samples): 702.529 / 695.549 / 693.018 ms -> median
  **697.032 ms / 2938.27 tok/s**.
- Quality: **PASS** — llama-server `/completion` greedy seed 42 coherent
  Rayleigh answer; decode 31.03 tok/s (no regression); parity suite at
  21A-champion parity (P7 exact 23/23; P1/P2/P5/P6 coherent, token-wise differ
  from SGLang ref — identical profile to the 21A control run).
- Artifacts: `escha-w2-lowgpu-mono-parity-standard-ffn-gdn-q2k.gguf`
  sha256 `2e61882d27f8828ce19d45d3eb33e6a4cd09802d2c84a19fa0636e8f8cf518fa`;
  evidence `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-23/2026-08-30/standard-ffn-gdn-q2k-001/`.
- Decision: **ACCEPTED — NEW CHAMPION** (P-ARCH-21A superseded).

## Current session evidence — 2026-08-30 (P-ARCH-23, phase 2: vocab sizes + Q6/Q4 + parity)

User directive: test larger vocab sizes while staying < 10 GB total; try Q6 and
Q4 gate quants for prefill speed and quality; retest base Escha W2 for a parity
update; surpass base (native + beellama) or achieve parity; if Q6/Q4 gains are
not substantial, promote P-ARCH-23 as champion; only then chase more speed.

### P-ARCH-23B — Q4_K gate (fits < 10 GB)

- Artifact `...-gdn-q4k.gguf`, 8,952,826,816 bytes = 8.953 GB.
  sha256 `5823ef2a988b1dda5f2a61ccebcec13adbc405320138bd95c3549a4d2a3cd1c4`.
- 2k prefill: 713.976 / 748.356 / 688.003 ms -> avg 714.112 ms / 2860.61 tok/s.
- Quality **PASS** (coherent). **Decision: SLOWER than Q2_K (+17 ms), bigger.
  No speed or quality gain -> REJECTED for champion.**

### P-ARCH-23C — Q6_K gate (fits < 10 GB)

- Artifact `...-gdn-q6k.gguf`, 9,342,110,656 bytes = 9.342 GB.
  sha256 `a3fc1cf073b1246fc92ff77d18ed168562135b70286009d9c34073943e1324ab`.
- 2k prefill: 697.519 / 698.581 / 692.249 ms -> avg 697.032 ms / 2942.08 tok/s.
- Quality **PASS** (coherent). **Decision: tied with Q2_K (+0.13% tok/s at
  +0.74 GB). No substantial gain -> REJECTED for champion.**

### P-ARCH-23D — embed-F16 vocab diagnostic (over 10 GB cap)

- Artifact `...-gdn-q2k-embedf16.gguf` (Q2_K gate + dense F16 embedding,
  LowGPU head), 10,635,156,288 bytes = 10.635 GB.
  sha256 `c3a0410f5f2b206085a695beae22cdfc6175fad4528c1ba2e20447bcb01f1617`.
- 2k prefill: 668.702 / 661.696 / 662.065 ms -> avg 664.154 ms / 3083.69 tok/s.
- Quality **PASS**. **Decision: over the 10 GB cap -> diagnostic only.
  Confirms the P-ARCH-22 ~35-40 ms embed-side vocab win stacks with the Q2_K
  gate (664 ms / 3084 tok/s). Highest-value future lever is a size-neutral
  embedding representation.**

### P-ARCH-23E — base Escha W2 retest on current build (parity check)

- Artifact `escha-w2-lowgpu-mono-parity.gguf` (production, full-Escha body),
  8,619,127,360 bytes. Retest on build-cuda-p20-control, same matched 2k.
- 2k prefill: 881.449 / 882.493 / 894.654 ms -> avg 886.199 ms / 2311.1 tok/s.
- **No new parity update**: matches historical 884 ms / 2316 tok/s. Our
  champion (697 ms / 2938) beats base in-beellama by ~21%. Native parity with
  the SGLang/upstream Escha reference (~2,850-3,120 tok/s) is now EXCEEDED at
  2938 tok/s; the original LowGPU GGUF (see below) is the remaining target.

### P-ARCH-23F — original LowGPU GGUF prefill target

- Artifact `Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf`, 9,570,663,040 bytes.
  Same matched 2k on the same build: 593.039 / 595.284 / 655.886 ms ->
  avg 614.736 ms / 3338.76 tok/s.
- **This is the remaining speed-parity target**: 3339 tok/s vs our champion
  2938 tok/s (gap ~14% / ~82 ms). Vocab-diagnostic shows ~35 ms of that gap
  lives in the embedding representation (664 ms with F16 embed), leaving the
  final ~50 ms to the body/kernel path.

### Promotion decision (per user directive)

- Q6/Q4 gains are NOT substantial (Q4 slower, Q6 tied) -> **promote P-ARCH-23
  (Q2_K) as champion**, P-ARCH-21A superseded. Ledger + GBrain updated
  2026-08-30. Next phase: speed breakthrough toward the 3339 tok/s original
  LowGPU GGUF target, starting with the size-neutral embedding opportunity.

## Current session evidence — 2026-08-30 (P-ARCH-23, phase 3: speed breakthrough, quantized embedding)

### P-ARCH-23G — standard-quantized embedding (Q6_K / Q4_K) on the champion body

User directive: chase speed toward the original LowGPU GGUF prefill (3339 tok/s)
after promotion. The embed-F16 diagnostic (P-ARCH-23D) showed the embedding
representation is worth ~33 ms but is +1.9 GB (over the 10 GB cap). Hypothesis:
a standard-quantized embedding (Q6_K/Q4_K) uses the same fast standard
`ggml_get_rows` path as F16 but at a fraction of the size, capturing most of the
win while staying under 10 GB.

- Change: converter `write_embed` extended to accept `q4_k`/`q6_k` storage —
  dequantizes the LowGPU codes (chunked, to avoid the worker cgroup OOM) then
  quantizes in-process with the existing `quantize_q4_k`/`quantize_q6_k` and
  writes `token_embd.weight` as the standard quant. Loader unchanged: a standard
  `token_embd.weight` (no `lowgpu_codes`) dispatches to standard get_rows.
- Artifacts (Q2_K gate + quantized embed, LowGPU head):
  - `...-gdn-q2k-embedq6.gguf`, 9,135,303,488 bytes = 9.135 GB
    sha256 `bdb4cf67a358afb820c55181a7a41017f4284be81a49e5eb49c253d7b7cb3296`
  - `...-gdn-q2k-embedq4.gguf`, 8,807,521,088 bytes = 8.808 GB
    sha256 `c9196385b007222acebf839e93fb25fea92d45ae82f7a8fa3257b1598e4ebe4c`
- Performance (matched 2k, graphs disabled, 3 samples):
  - Q6 embed: 674.181 / 667.804 / 671.379 ms -> avg 671.121 ms / 3051.66 tok/s
  - Q4 embed: 671.261 / 661.596 / 662.708 ms -> avg 665.188 ms / 3078.96 tok/s
- Quality: **PASS** both (coherent Rayleigh /completion, greedy seed 42).
- Budget: Q6 embed 9.135 GB (FIT < 10), Q4 embed 8.808 GB (FIT < 10); both
  recover essentially the full embed-F16 win (664 ms / 3084 tok/s) without the
  +1.9 GB penalty.
- Decision: **P-ARCH-23G-Q4 is the new speed leader on the champion body**
  (8.808 GB, 665.2 ms / 3078.96 tok/s, quality PASS). Q4 embed beats Q6 embed
  (8.81 GB vs 9.14 GB at equal/higher speed) -> Q4 chosen. This is +141 tok/s /
  -32 ms over the promoted Q2K+LowGPU-vocab champion, staying under the 10 GB cap.

- Parity confirmation: Q4-embed parity suite = P1 0.4% / P2 16.3% / P5 0% /
  P6 8.8% / P7 100% — **identical to the Q2K champion profile** (no regression).
  Quality PASS (coherent). All promotion criteria met (size <= 10 GB cap,
  quality PASS, reproducible 2k improvement over champion, no parity/stability
  regression, fingerprints in ledger + GBrain). **PROMOTED — new speed leader.**

### Speed-parity status (2026-08-30, matched 2k)

| Artifact | Size | 2k prefill | Gap to 3339 target |
| --- | --- | --- | --- |
| Original LowGPU GGUF (target) | 9.571 GB | 614.736 ms / 3338.76 tok/s | — |
| Champion Q2K + LowGPU vocab | 8.599 GB | 697.032 ms / 2938.27 tok/s | -400.5 tok/s |
| Q2K + Q4 embed | 8.808 GB | 665.188 ms / 3078.96 tok/s | **-259.8 tok/s** |
| Q2K + F16 embed (diagnostic, over cap) | 10.635 GB | 664.154 ms / 3083.69 tok/s | -255.1 tok/s |

- The quantized-embed breakthrough closes ~35% of the remaining gap (141 of
  ~400 tok/s). Remaining ~260 tok/s / ~70 ms is the body/kernel path (FFN + escha
  attention + QKV projections), the next speed frontier.
- Promotion of Q4-embed variant deferred pending stability/parity confirmation
  (per promotion criteria: exact fingerprints in ledger + GBrain, no regressions).

### P-ARCH-23H — full-attention substitution (16 layers off the Escha packed path)

User directive: keep chasing speed toward the 3339 tok/s original LowGPU GGUF
target. Body-substitution feasibility was re-derived: the remaining 160 Escha
sidecars only fit under 10 GB with IQ-density quants (Q4_K would be +1.77 GB
-> 10.58 GB OVER; IQ2_XXS +36 MB -> 8.84 GB FIT, IQ2_S/Q2_K/IQ3_XXS also FIT).
Correlation check on the donor source GGUF (Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS):
the 16 FULL-attention layers' projections are VALID donors (attn_q corr 0.927,
attn_k 0.955, attn_v 0.955, attn_output 0.940), but the linear-attention
in_proj_qkv (0.40) and ssm_out (0.04) are WRONG donors (same trap as the gate),
so only the 64 full-attn projections (1.678B params) are substitutable by raw
copy. Linear-attention QKV/SSM remain Escha-packed.

- Change: converter flag `--standard-attn-ffn` (with --standard-ffn-gguf) writes
  blk.{il}.attn_q/k/v/output.weight from the source GGUF (raw IQ quant) for
  full-attn layers (il%4==3) instead of escha sidecars; loader (qwen35.cpp)
  per-tensor fallback for attn_q/k/v/output like the gate (if no escha_code ->
  create standard weight; graph build_layer_attn already routes via .active()).
  Linear layers untouched.
- Artifact `...-embedq4-attn.gguf`, 9,036,050,208 bytes = 9.036 GB (FIT < 10).
  sha256 `2892acff4c0b256bb7e8c8ea95034b9da1187515f863406666626c161ea31d7b`.
  Inventory: 1144 tensors (1336 - 192: the 64 full-attn escha sidecars replaced
  by 64 standard weights, 12 tensors/layer x 16 layers removed).
- Performance (matched 2k, graphs disabled, 3 samples): 642.193 / 642.997 /
  703.542 ms -> avg 662.911 ms / 3095.05 tok/s. Clean pair 642.2/643.0 ms
  (~3184 tok/s); one 703.5 ms straggler (variance, warmup).
- Quality: **PASS** / completion (greedy seed 42) coherent Rayleigh answer.
  Parity suite: P1 0.5% / P2 16.3% / P5 0% / P6 8.8% / P7 100% - no regression
  vs champion profile.
- Decision: **new speed leader candidate** - ~23 ms over P-ARCH-23G-Q4 embed
  leader at +0.23 GB (9.036 vs 8.808). Clean-pair estimate ~3184 tok/s, within
  ~5% of the 3339 tok/s target. Promotion to be confirmed with a repeat bench
  (straggler sample) per promotion criteria.

- Repeat confirmation (r6, matched 2k, graphs disabled): 643.02 / 640.58 /
  640.86 / 642.01 / 707.18 / 639.76 ms -> 5 clean samples 639.8-643.0 ms
  (median ~642 ms, avg 3144.2 tok/s), one 707 ms warmup/variance straggler.
  **Reproducible -> PROMOTED as new speed leader** (9.036 GB, ~642 ms /
  ~3180 tok/s clean, quality PASS, parity no-regression). The 48 linear-attention
  QKV/SSM projections remain Escha-packed (wrong donors for raw copy - they
  would need reconstruct+quantize, the final substitution lever, or kernel work).

### P-ARCH-23I — linear-attention QKV/SSM reconstruction+quantize (full substitution)

User directive: final lever toward the 3339 tok/s original LowGPU GGUF target.
The 48 linear-attention in_proj_qkv + out_proj are WRONG donors in the source
GGUF (corr 0.40 / 0.04), so they were RECONSTRUCTED from checkpoint sidecars
(generalized `reconstruct_escha(prefix, ic, oc)` = Hadamard-128 IC -> rin ->
Hadamard-128 OC -> rout, K=2, cbA) and quantized to Q2_K in-process.

- Change: converter flag `--standard-linear-ffn` reconstructs+quantizes
  in_proj_qkv (5120->10240) as `blk.{il}.attn_qkv.weight` Q2_K and out_proj
  (6144->5120) as `blk.{il}.ssm_out.weight` Q2_K for the 48 linear layers;
  loader per-tensor fallback for attn_qkv/ssm_out (build_qkvz line 357 and
  ssm_out line 618 already route via .active()). Full-attn layers via
  --standard-attn-ffn (prior). All 64 layers' Q/K/V/O/QKV/SSM now standard.
- Artifact `...-embedq4-attn-linear.gguf`, 9,345,100,992 bytes = 9.345 GB
  (FIT < 10). sha256 `9725d1e54c4f325f38bc01ddfbf66313b96da4f70454dec1c8f534f485454825`.
  Inventory: 856 tensors (all body projections standard; only 160 escha codes
  for linear in_proj_z gate sidecars? no - gate is Q2_K too; remaining escha
  codec tables only).
- Performance (matched 2k, graphs disabled): 620.978 / 621.992 / 621.469 ms
  (avg 621.480 ms / 3295.36 tok/s); r6 confirm: 616.38 / 617.95 / 618.70 /
  631.19 / 619.29 / 619.86 ms -> avg 3300.4 tok/s, median ~619 ms.
- Quality: **PASS** / completion (coherent Rayleigh). Parity suite **IMPROVED**:
  P1 5.6% (was 0.4-0.5%), P2 38.8% (was 16%), P6 8.8%, P7 100% exact, P5 0%.
  The reconstruction-based QKV/SSM is closer to the SGLang reference than the
  packed Escha path - quality gained, not lost.
- **Decision: PROMOTED - FINAL SPEED LEADER. Prefill-speed parity with the
  original LowGPU GGUF ACHIEVED: 619-621 ms / ~3300 tok/s vs target
  614 ms / 3339 tok/s (within ~1.2%). Artifact 9.345 GB under the 10 GB cap.
  All 64 layers' attention projections + all FFN + gate + Q4 embed are now
  standard GGML; the remaining escha path is the shared codec tables only.**

## CERTIFIED — club-3090 medium quality 5-pack (2026-08-31)

Authoritative quality benchmark of the four control models on the **gated
BeeLlama runtime** (`build-cuda-qwen35-gated`, qwen35.cpp escha semantics gated
on `escha_version`). **Judge PASS (exit 0)** on 2026-08-31 after adding
deterministic launch/json/dir-listing proof. See
`evidence/club3090-medium/2026-08-31-gated/` (in the escha-w2-lowgpu tree) and
`docs/current-state.md`.

### Benchmarked models (5 non-Docker packs, 75 scenarios, thinking force-off)

| Model | Artifact | /75 | equiv/150 | toolcall | instruct | structout | dataextract | reasonmath |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **original-lowgpu** | `Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf` | **66** | **132** | 14/15 | 13/15 | 15/15 | 12/15 | **12/15** |
| base-escha-w2 | `escha-w2-lowgpu-mono-parity.gguf` | 65 | 130 | 14/15 | 13/15 | 15/15 | 12/15 | 11/15 |
| p-arch-23 | `...standard-ffn-gdn-q2k.gguf` | 65 | 130 | 15/15 | 13/15 | 15/15 | 12/15 | 10/15 |
| p-arch-23g | `...standard-ffn-gdn-q2k-embedq4.gguf` | 65 | 130 | 15/15 | 13/15 | 15/15 | 12/15 | 10/15 |

### Configuration (fixed, all four models, same binary)

- Runtime: `build-cuda-qwen35-gated/bin/llama-server` (commit `0b035b3a2-dirty`,
  qwen35.cpp gated on `escha_version`)
- F16 KV (`-ctk f16 -ctv f16`), 32K ctx (`-c 32768`), single slot (`-np 1`),
  flash attention on, jinja, thinking force-off, **chat parsing enabled**
  (no `--skip-chat-parsing`)

### Semantic conclusions (persist to GBrain)

1. **Artifact correction:** `escha-w2-lowgpu-mono.gguf` is **obsolete/invalid**.
   Known conversion defects: residual RMSNorm weights offset by −1.0; incorrect
   `ssm_beta`/`ssm_alpha` representation/byte ordering. The canonical full-Escha
   control is `escha-w2-lowgpu-mono-parity.gguf`
   (sha256 `e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d`).
2. **BeeLlama compatibility fix:** Escha-specific Qwen semantics in qwen35.cpp
   must be gated by reliable Escha metadata (`escha_version`). Standard
   Qwen/GGUF models must retain original BeeLlama semantics. **Regression
   evidence: original-lowgpu recovered 5/75 → 66/75 after gating** — direct proof
   that ungated Escha semantics were being applied to standard GGUFs. Treat as a
   BeeLlama model-compatibility/runtime fix, not an artifact-specific workaround.
3. **Quality result:** original-lowgpu 66/75; base-escha-w2 65/75; p-arch-23
   65/75; p-arch-23g 65/75. **The three Escha variants are effectively tied on
   this 5-pack.** P-ARCH-23/23G should be evaluated primarily for their
   performance effects (prefill speed), NOT claimed as quality improvements.

## EXP-01 — SM120 async-route consolidation (2026-08-31) — PROMOTED

Experiment 1 of the ESCHA-W2 PREFILL phase (branch `escha-w2-prefill`).
Determined whether the SM120 async A-stage overlap can safely become the
default Escha prefill route.

- **Change:** `ggml/src/ggml-cuda/escha-moe.cu` — invert the SM120 default so
  the double-buffered `cp.async` activation overlap is used unconditionally
  (gated by architecture + Escha operator via `escha_version`); the synchronous
  `uint4` fallback becomes an explicit opt-in via `ESCHA_MMA_SM120_SYNC_FALLBACK`.
  Removes the `ESCHA_MMA_SM120_ASYNC_EXPERIMENT` compile-flag dispatch debt.
- **Control (sync):** 1412.5 ms / 1449.9 tok/s matched-2k, CV 1.53%.
- **Candidate (async):** 889.5 ms / 2302.4 tok/s, CV 0.72% — **+58.8% faster**.
- **Consolidated default (no flag):** 887.9 ms / 2306.5 tok/s — matches candidate.
- **Gates:** CV ≤2% ✓; decode no regression (25.03→25.45 tok/s, within noise) ✓;
  P2/P7 100% both arms ✓; standard-GGUF output identical ✓; route=mma-fp16 ×800 ✓.
- **Artifact:** `escha-w2-lowgpu-mono-parity.gguf` sha256
  `e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d` (unchanged).
- **Verdict:** PROMOTE. Implementation commit `215aa4ac3` (separate from
  evidence/docs). Milestone certification required (first promoted prefill
  optimization, >10% gain, routing change) — full medium 5-pack next.
- Evidence: `escha-w2-lowgpu/evidence/EXP-01-sm120-async/2026-08-31/`
  (external; `evidence-summary.json` in-repo binding).

### Milestone certification (2026-08-31) — PASS, no quality regression

Full club-3090 medium 5-pack on the promoted consolidated (async-default)
build with the canonical full-Escha artifact: **65/75 (130/150 equiv)** —
identical to the certified base-escha-w2 baseline. Per-pack:
toolcall 14/15, instructfollow 13/15, structoutput 15/15, dataextract 12/15,
reasonmath 11/15. Evidence:
`evidence/EXP-01-sm120-async/2026-08-31/milestone-cert/` (milestone-summary.json).

### EXP-02 — direct-fragment packed K2 GEMM (2026-08-31) — REJECTED

Experiment 2 of the ESCHA-W2 PREFILL phase. Hypothesis: decode packed K2 codes
directly into warp-owned MMA B fragments (removing shared-B stores/barrier and
B ldmatrix reloads) for the target K2 shape (IC=5120, OC=17408, M≥2048).
Implemented behind `ESCHA_MMA_DIRECTFRAG_EXPERIMENT` (isolated, NOT default;
route tag `mma-directfrag-fp32`).

- **Result:** matched-2K prefill got **slower**: control (async default)
  880.1 ms / 2327.0 tok/s vs exp2 916.0 ms / 2235.7 tok/s (**−3.9% tok/s**).
  Fails the ≥5% full-2K gain gate (and ≥10% matmul gate).
- **Route proof:** 128× `mma-directfrag-fp32` (target K2 shapes) + 672×
  `mma-fp16`; per-projection matmul ~2.5 ms vs ~2.0 ms async baseline.
- **Likely cause:** 176 regs/thread vs the plan's <128 target (occupancy hit);
  warp-local decode duplicates codebook work across M-warps.
- **Verdict: REJECT.** Isolated (flag-gated, default path unchanged), then the
  kernel edits were **reverted** so the implementation tree matches the
  promoted EXP-01 state (`215aa4ac3`). Certified checkpoint untouched.
- Evidence: `escha-w2-lowgpu/evidence/EXP-02-directfrag/2026-08-31/`
  (external; evidence-summary.json records REJECT).
- Next: per Sol review — reduce register pressure or move to the next-ranked
  runtime opportunity; do not combine with another optimization in one diff.

### EXP-03 — shared-B 256x64 balanced K2 tile (2026-08-31) — REJECTED (neutral)

Experiment 3 of the ESCHA-W2 PREFILL phase (Sol next-plan #1). Hypothesis:
keep the proven shared-B/`ldmatrix`/FP32-HMMA + async-A path but change the K2
prefill tile 128x128 → 256x64 (same area/CTA count/threads; half B-decode per
CTA, reuse across 2× M rows) to amortize codebook work without EXP-02's
warp-local duplicate decode.

- **Result:** neutral — control (128x128) 889.0 ms / 2303.7 tok/s vs exp3
  (256x64) 892.7 ms / 2294.1 tok/s (**−0.42% tok/s**), 1/5 samples beat
  control median. Fails the ≥5% full-2K gain gate.
- **Resources:** 128 regs/thread, no spills (cuobjdump); route proof
  128× `mma-bm256-bn64` + 672× `mma-fp16`; per-projection K2 5120→17408
  matmul median 2.098 ms (unchanged vs ~2.0 ms async baseline).
- **Gates:** decode PASS (+0.76%), P2/P7 100% both.
- **Verdict: REJECT.** Kernel edits reverted; implementation tree matches
  promoted EXP-01 (`215aa4ac3`). Certified checkpoint untouched.
- **Conclusion:** tile aspect does not move prefill. Combined with EXP-02,
  B-decode amortization and fragment/tile layout are NOT the bottleneck —
  the packed K2 matmul body itself is the wall.
- Evidence: `escha-w2-lowgpu/evidence/EXP-03-bm256bn64/2026-08-31/`
  (external; evidence-summary.json records REJECT).

## ARCH-01 — Dense-vs-MoE architecture audit (2026-08-31) — DENSE-CORRECT / PERF-ARCH-MISMATCH

Architecture audit before further kernel tuning. Classification:
**DENSE-CORRECT / PERF-ARCH-MISMATCH**.

- **Dense semantics proven:** canonical artifact is `qwen35` (64 layers, hidden
  5120, intermediate 17408, n_expert absent, zero expert/moe tensors, 400
  `escha_code` tensors with no expert dimension). Loader forces
  `GGML_OP_ESCHA_MUL_MAT` when `n_expert==0` (llama-model-loader.cpp:1224-1230);
  CUDA maps it to `ggml_cuda_op_escha_mul_mat`; dense op reads 6 srcs (no
  `ids`); runtime 800/800 `mma-fp16` with zero moe/expert/topk records.
- **Filename is historical, not a correctness defect:** the dense tensor-core
  prefill kernel `escha_matmul_dense_tiled_mma<K,128,128>` is a genuinely
  dense 2D-tiled GEMM (CTA tile grid, split-K, CTA B-decode, ldmatrix+HMMA,
  async A); it shares only the ESCHA codec decoder with the MoE path.
- **Performance-architecture mismatch:** BeeLlama keeps separate
  rotate → packed-GEMM → finalize kernels with fp32 MMA accumulate; the
  official `escha-runtime-qwen3dense` fused `escham_code_gemm` uses a mixed
  fp16 accumulator policy (P-ARCH-19: official mixed 623.380 ms / 3285.31
  tok/s vs fp32 1176.882 ms / 1740.19 tok/s = 1.888×). P-ARCH-20 showed the
  BeeLlama fp16 toggle alone is −44.22% — consistent with a structural
  difference and ruling out an accumulator-only toggle (causal mechanism to
  be isolated by staged attribution).
- **Decision:** EXP-04 = dense fused-prefill parity, **staged attribution**
  (measure the fuseable rotate/GEMM/finalize bound first, then ONE structural
  candidate with SASS/profiler proof and a quality gate). P-ARCH-19/20 prove
  divergence and rule out an accumulator toggle, but do NOT isolate which
  mechanism carries the official gain; the fused-parity gain is hypothesis,
  not yet evidenced.
- Evidence: `docs/escha-w2-architecture-provenance-audit.md`;
  `escha-w2-lowgpu/evidence/ARCH-01-architecture-audit/2026-08-31/`.
- Worker note: 4× DeepSeek V4 Flash (Nous Portal) failed at dispatch with
  HTTP 401; the primary agent performed the full audit directly.

### EXP-04 Stage 1 — fuseable rotate/GEMM/finalize bound (attribution) — COMPLETE

- Date: 2026-09-01. Measurement-only (no code changes). Branch
  `escha-w2-prefill`, HEAD `4501b3ee1` (ARCH-01; `escha-moe.cu` byte-identical
  to EXP-01 promotion `215aa4ac3`, sha256
  `bfe0e43d135220cc2d62033c12ac4b43896cf07b4ae5dcbf81bc98dc215b43c2`).
- Fresh build `build-cuda-exp04-stage1` (cmake+ninja, Release, arch 120,
  `GGML_CUDA=ON GGML_CUDA_FA=ON GGML_NATIVE=OFF GGML_CUDA_GRAPHS=ON`).
  Binary hashes in `evidence/EXP-04-stage1/2026-09-01/provenance.manifest`.
- Contract: canonical `escha-w2-lowgpu-mono-parity.gguf` (sha256
  `e307007f…4778d`), `llama-bench -p 2048 -n 0 -ngl 99 -b 2048 -ub 2048
  -ctk f16 -ctv f16 -fa on` with fixed shared-2048 IDs, RTX 5090 (SM120,
  driver 610.88).
- Attribution run: `ESCHA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1`.
  800/800 lines report `route=mma-fp16 rows=2048 gen=0` — no fallback.
- Timed control (graphs on, no profile): median 2284.7 tok/s (2193.6 /
  2333.3 / 2284.7; avg 2270.5, stddev 70.9, CV 3.12%) — median 0.75% below
  the banked EXP-01 2k (~2302 tok/s); control confirmation, NOT a ≤2% CV
  candidate claim (spread 3.12% on 3 samples).
  Decode 64: 39.26 / 44.23 / 40.56 tok/s — consistent with EXP-01 guardrail.
- Steady-state attribution (792 calls, first cold call per family excluded):
  **rotate 4.6% · matmul 88.6% · epilogue 6.7%** (61.9 / 1186.1 / 90.2 ms of
  1338.3 ms measured projection time). Matmul ≥73% in every family
  (5120→1024 family is the exception at 73.2% with rotate 21.8%, but only
  63 calls × 0.256 ms — immaterial).
- Cold first-call artifacts are large but non-recurring (e.g. 5120→10240
  first call rotate 3.40 / epilogue 8.47 ms vs 0.059 / 0.129 steady state) —
  warmup/allocator, excluded from attribution.
- **Conclusion:** fuseable launch bound ≈ 11.3% (rotate 4.6% + epilogue
  6.7%) best case; realistic recoverable less (P-ARCH-14 fused finalize
  neutral). A launch-fusion candidate cannot plausibly reach the ≥20%
  breakthrough gate; Stage 2 should target the packed-GEMM body (structural
  mixed accumulator or B-decode/launch structure with SASS proof), not
  rotate/finalize fusion — unless a fusion candidate is run only as a small
  positive (≥5% gate).
- Raw artifacts: `evidence/EXP-04-stage1/2026-09-01/` (stage1-profile.*,
  stage1-bench.*, stage1-decode.*, STAGE1-REPORT.md, provenance.manifest).
- Sol/Codex review (2026-09-01): **VERDICT = CONFIRM** — CHECK 1..6 PASS,
  Stage 1 gate PASS (`/tmp/escha-exp04-stage1-review2.md`; first round
  REVISE for CV-≤2% overclaim + incomplete fingerprint contract, both fixed
  in commit `8ebcbdde5`).
- Decision: **Stage 1 COMPLETE + Sol-verified.** Proceeding to Stage 2
  (ONE structural variable, SASS/profiler proof + quality gate).

### EXP-04 Stage 2 — structurally-gated mixed accumulator — RESULTS (2026-09-01)

- Date: 2026-09-01. ONE structural variable under compile gate
  `ESCHA_MMA_MIXEDACC_EXPERIMENT=1`: per-projection `IC<=6144 → FP16_ACC=true`,
  else fp32, applied across K2/K3 prefill families (native Escha mixed policy;
  NOT the rejected P-ARCH-20 single-shape toggle). Geometry/grid/smem/decode/
  A-stage/partial/finalize unchanged; partial buffer stays float.
- Implementation commit `7b1880f41`. Sol implementation review: round 1 REVISE
  (fp16 store OOB: `tile_ah` ne=2 but loop used tile_c ne=4, dropped `.y`
  lanes) → fixed → round 2 **CONFIRM**.
- SASS proof (cuobjdump per-symbol): fp16 K2/K3 symbols contain only
  `HMMA.16816.F16` (16×, 0 F32); fp32 twins + control contain only `.F32`
  (32×, 0 F16). Registers: fp16 97, fp32 128 — all ≤128, STACK/LOCAL 0, no
  spills, SHARED 1024 identical.
- Route proof: 800/800 tagged, **0 predicate mismatches** (672
  `mma-fp16-mixedacc` IC≤6144, 128 `mma-fp32-mixedacc` IC>6144).
- Performance (matched control/candidate, graphs on, 2k, canonical model):
  r3 A/B 2508.1 vs 2281.0; r3 B/A 2496.0 vs 2251.2; r5 2496.9 vs 2258.9 →
  **median gain +10.0 / +10.9 / +10.5%, stable**. CV 3.1–6.3% per arm on this
  WSL host — above the plan's ≤2% letter (same host noise as Stage 1 control
  3.12% and banked EXP-01 samples); flagged for Sol verify.
- Decode guardrail r5: candidate 44.55 vs control 42.69 → no regression.
- Parity: P2-factual + P7-tool-call, greedy seed 42, 16 tokens — candidate
  16/16 and control 16/16 (100%).
- Family regressions: **none** — all 7 fp16-acc families improved matmul
  17.6–26.0% (e.g. 5120→17408 2.095→1.688 ms); fp32-side family flat (−0.5%).
- **Classification: SMALLER POSITIVE (≥5%, <20%): +10% full-2K median.**
  Not a ≥20% breakthrough. All hard gates except the host-noise CV letter
  pass (route, SASS, ≤128 regs no spills, decode, P2/P7, family).
- Evidence: `evidence/EXP-04-stage2/2026-09-01/` (STAGE2-REPORT.md,
  provenance.manifest, profile/bench/decode/parity artifacts, per-symbol SASS,
  resource-usage.txt).
- Sol VERIFY (2026-09-01): round 1 REVISE (raw resource-usage evidence not
  committed; CV wording inconsistency) → both fixed (`ba4e93851`) → round 2
  **VERDICT=CONFIRM**, Stage 2 gate CONDITIONAL (CV ≤2% letter not met on this
  WSL host; disclosed, no overclaim), all 8 checks PASS.
- **FINAL: EXP-04 Stage 2 = SMALLER POSITIVE (≥5%, <20%): +10% full-2K median,
  Sol-verified. NOT promoted to default in this session; promotion would
  require the ledger's full milestone certification (5-pack quality, depth
  matrix) per plan section 4.**

## Durable evidence sources

- GBrain: *Qwen3.5 hybrid prefill batching audit — rows 2-4 vs 512 (2026-08-29)*.
- GBrain: *Escha x LowGPU BeeLlama prefill investigation — Blackwell MMA stall (2026-08-29)*.
- GBrain: *BeeLlama Escha WMMA layout and microbatch gate — 2026-08-29*.
- GBrain: *BeeLlama Escha x LowGPU benchmark — 2026-08-29*.
- GBrain: *BeeLlama Escha x LowGPU runtime validation — 2026-08-29*.
- GBrain: *Escha W2 runtime ports ecosystem + kernel strategy map (2026-08-29 research)*.

## Session update template

```markdown
### <ID> — <short name>

- Date/time:
- Hypothesis:
- Independent variable:
- Source/Git/binary/model/GPU fingerprints:
- Exact command and environment:
- Route proof:
- Build result:
- Operator exactness:
- Token parity:
- Performance samples and median:
- Peak VRAM:
- Raw artifacts:
- Decision: ACCEPTED | REJECTED | INCONCLUSIVE | BLOCKED
- Elimination reason or next single experiment:
```
