# P-ARCH-18 — original LowGPU Qwen3.8-27B control in BeeLlama

**Status: CLOSED — CASE D. Bee's native quant execution runs the whole
64-layer qwen35 2k prefill faster than the Escha W2 path, and faster than the
retained Escha-runtime reference.**

## Scope and control

No CUDA kernel, loader, graph default, production default, or model was
modified. The original LowGPU GGUF was run directly in the existing
`build-cuda-parch10-async` binary with the P-ARCH-17 control methodology:
RTX 5090, immutable 2,048 raw IDs (`shared-2048.ids`, SHA-256
`695c3609bc35a32003a23be3ba1fbacc16cc94955548c2e855e91661c3f62350`),
`-p 2048 -n 0 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on`, CUDA graphs on
(`GGML_CUDA_DISABLE_GRAPHS` unset), 4 samples per capture, two captures per
artifact, median of wall times. All runs exit 0 with zero CUDA-error lines.

- Binary: SHA-256
  `32911ef90000dfc31d7149d5cf9897e7b087547f7ac87ff348d8ebbb97865edc`.
- Runtime CUDA library: SHA-256
  `695001fedab34997a548f196571ce4ceaf07fc56ae16e67c4caf4b56093d0cd5`.
- Source base: detached `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d`;
  `CMAKE_CUDA_FLAGS=-DESCHA_MMA_SM120_ASYNC_EXPERIMENT=1`.
- `docs/escha-architecture/PROTOCOL.md` was not present in this checkout (same
  as P-ARCH-17); the retained protocol boundary in P-ARCH-05/15/17 was used.

The P-ARCH-17 medians were retained as a reference, and the hybrid and
original Escha W2 artifacts were additionally re-run in the same session
(control-003/004/006/007) so the three-way comparison is same-session
balanced: two captures per artifact, alternating order, 8 samples each.

## Phase 1 — exact original LowGPU artifact

| property | value |
|---|---|
| path | `beellama-release/models/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf` |
| source | `TheWegemann/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS-GGUF` (HF), commit `a437d69f588a579bd548d26391c40158f6f22a6f`, quantized by TheWegemann from `Qwen/Qwen3.8-27B`, imatrix-calibrated, created 2026-08-21 |
| file name | `Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf` |
| file size | 9,570,663,040 bytes |
| SHA-256 | `ad85e40a28aafd907eeb6ff6b21786b897dd750b0918427f1243d6d84ebcc72` |
| GGUF | v3, `general.architecture=qwen35`, quantization_version 2, file_type 24 |
| hparams | 64 layers, 5,120 embed, 17,408 FFN, 24 heads / 4 KV, context 262,144, full-attention interval 4, SSM inner 6,144 / state 128 / conv 4 |
| tokenizer | gpt2/qwen35 pre, vocab 248,320, bos/pad 248044, eos 248046, add_bos false |
| tensors | 851 |
| quant mix | f32 353, q8_0 96, q2_k 112, q4_k 10, q5_k 2, iq1_m 14, iq1_s 33, iq2_s 50, iq2_xs 12, iq2_xxs 27, iq3_s 27, iq3_xxs 67, iq4_xs 48 |
| embedding | `token_embd.weight` IQ4_XS, 675,430,400 B `[5120, 248320]` |
| LM head | `output.weight` Q4_K, 715,161,600 B `[5120, 248320]` |
| escha/lowgpu sidecars | none (no `*.escha_*`, no `*.lowgpu_*`, no version keys) |

The file matches the HF `totalFileSize` and the local `.cache` metadata record.
This is the whole-model LowGPU quant the project's "LowGPU" vocabulary lineage
is named after; the hybrid re-uses the LowGPU 3-bit concept only for the vocab
boundary and keeps the Escha body.

## Phase 2 — model structure comparison (same BeeLlama runtime)

| Property | LowGPU | Hybrid | Escha W2 |
| --- | --- | --- | --- |
| total size (bytes) | 9,570,663,040 | 8,619,127,360 | 12,691,575,008 |
| tensor count | 851 | 2,058 | 2,054 |
| embedding | IQ4_XS raw tensor | 3 LowGPU sidecars (I8 codes + F16 scales + I8 zps) | F16 raw tensor (2,542,796,800 B) |
| LM head | Q4_K raw tensor | 3 LowGPU sidecars | F16 raw tensor |
| attention tensors | raw quant (attn_q 16, attn_k 16, attn_v 16, attn_qkv 48, attn_gate 48, attn_output 16) | 400 escha sidecars (code/rin/rout) | identical sidecar set |
| MLP tensors | raw quant (64 x ffn_gate/up/down) | 192 escha sidecars | identical sidecar set |
| K2 tensors | none | 272 `*.escha_code` (I16) + `escha_dep_k2` | identical |
| K3 tensors | none | 128 `*.escha_code` (I16) + `escha_dep_k3` | identical |
| codec tables | none | `escha_lut` F16 + dep K2/K3 | identical |
| other quant families | f32 353, q8_0 96, q2_k 112, q4_k 10, q5_k 2, iq* mix | f16 899, f32 753, i16 402, i8 4 | f16 899, f32 753, i16 402 |
| architecture metadata | qwen35, no version keys | qwen35 + `escha.version=1` + `lowgpu.version=1` | qwen35 + `escha.version=1` |
| tokenizer metadata | identical gpt2/qwen35, 248,320 | identical | identical |
| tensor layout/order | standard llama.cpp order | codec tables first, sidecars interleaved, lowgpu sidecars last | F16 vocab first, sidecars interleaved |
| Bee loader transformations | none (raw block data mmap/copy; no dequant/repack) | none on compatible sidecars | none |

The hybrid and Escha bodies share 2,052 name/shape/type-identical descriptors
(2,058 - 6 lowgpu sidecars = 2,052); the LowGPU model is a different,
vanilla-quant build with no escha machinery.

## Phase 3 — load LowGPU without changing Bee

**Direct load succeeded.** No metadata or format adaptation was required:
`qwen35.escha.version` and `qwen35.lowgpu.version` both default to 0 when
absent, the loader creates standard tensors, and every projection falls back
to `GGML_OP_MUL_MAT`. Verified by the load/run itself (exit 0) and by source
gating (`src/models/qwen35.cpp:21-24, 86, 131, 147, 164, 320-612`) plus the
loader audit. `llama-bench` reports `qwen35 27B IQ1_S - 1.5625 bpw` (its
dominant-type label for the mixed IQ3XXXS file).

## Phase 4 — verified execution path

| item | LowGPU | Hybrid / Escha W2 |
| --- | --- | --- |
| op family | standard `MUL_MAT`/`GET_ROWS` for every projection | `GGML_OP_ESCHA_MUL_MAT` for all 400 projections (272 K2 + 128 K3) |
| kernel family | stock CUDA quant kernels: dequant + tensor-core/MMQ for IQ1/IQ2/IQ3/IQ4/Q2_K/Q4_K/Q5_K/Q8_0 | `escha_matmul_dense_tiled_mma` (rotate -> MMA partial -> finalize), route `mma-fp16` 800/800 in P-ARCH-17 |
| W2 route | none | K2/K3 `mma-fp16`, no cublas/WMMA/tiled-FMA |
| codebook/decode | n/a | LUT + dep-table tile decode per projection |
| loader conversion/repack | none | none |
| graphs | on (same setting) | on (same setting) |
| ubatch / batch | 2048 / 2048 | 2048 / 2048 |

The LowGPU model does **not** exercise the escha W2 path at all; that absence
is the control variable. Per-kernel launch counts for the standard path were
not instrumented (no build change allowed); the operator/kernel-family mapping
is deterministic from the tensor inventory and dispatch source.

## Phase 5 — matched 2k prefill result (same-session, n=8 per artifact)

| Model in BeeLlama | Prefill tok/s | Prefill ms | Relative to best Bee | Relative to Escha runtime |
| --- | ---: | ---: | ---: | ---: |
| original LowGPU | **3447.60** | **594.037** | 1.000 (best) | **-75.517 ms (-11.28%) vs 669.554 ms** |
| current hybrid | 2327.07 | 880.076 | -32.50% time (1.481x slower) | reference-only; different harness |
| original Escha W2 | 2412.69 | 848.847 | -30.02% time (1.429x slower) | reference-only; different harness |

Medians are of wall times (reciprocal -> tok/s). All 8 samples per artifact
and per-capture values are in `aggregate.json` and the control-*/ stdout.json
files. Retained P-ARCH-17 medians (different session, same runner): hybrid
905.168 ms / 2262.56 tok/s, Escha W2 844.617 ms / 2424.77 tok/s. The
same-session hybrid re-run is ~2.8% faster than its P-ARCH-17 median
(session drift), which is why the same-session series is the primary
comparison. The Escha-runtime reference (3058.75 tok/s / 669.554 ms) is
retained separately and is **not** mixed into the Bee-internal A/B.

## Phase 6 — interpretation: CASE D

**LowGPU is much faster than both hybrid and Escha-W2-in-Bee.** It is also
faster than the retained Escha-runtime reference (3447.60 vs 3058.75 tok/s).
This is not a memory-footprint effect: LowGPU (9.57 GB) is larger than the
hybrid (8.62 GB) and only ~2.3 GB smaller than Escha W2.

First responsible difference (attribution): the operator path. LowGPU runs
the entire 64-layer body on Bee's stock CUDA quant kernels (dequant +
tensor-core/MMQ), while the hybrid and Escha W2 run 400 projections through
Bee's ported escha code-GEMM path (LUT/dep tile decode, rin/rout rotation,
separate MMA partial + finalize), which carries the residual quantified in
P-ARCH-16 (K2 W2 350.756 ms Bee vs 185.725 ms Escha; all-W2 651.132 ms of the
919.410 ms hybrid prefill). Replacing that path with standard quant GEMMs
removes ~286-305 ms of the hybrid's 2k prefill and lands below the Escha
reference. The vocab boundary (IQ4_XS/Q4_K vs LowGPU sidecars vs F16) is not
exculpated by this gate alone: the LM head is not executed by `-n 0`, and the
embedding is a single gather at 2k tokens; the dominant delta is the body
path.

Per the gate, no model or kernel was modified. The alternative opened by this
control, to be evaluated in a future bounded experiment: selectively replace
escha W2 projections with standard low-bit quant placements (e.g., FFN
gate/up/down or K2-only layers) and measure quality-vs-prefill tradeoff, or
route only the W2 layers that remain quality-critical through escha.

## Phase 7 — lightweight attribution

Stage-level per-kernel timing for the standard path would require a new
instrumented build (prohibited here). The existing P-ARCH-16 hybrid stage
attribution (K2 W2 350.756 ms, K3 W2 300.377 ms, non-W2 267.788 ms at
919.410 ms) plus the operator mapping above is sufficient for the gate:
the whole LowGPU prefill (594.037 ms) is faster than the hybrid's W2-only
stage (651.132 ms), so the standard quant path executes the same-shape GEMMs
(plus all norms/SSM/attention) in less time than the escha port's W2
projections alone. No further attribution was warranted before closing.

## Hermes DeepSeek V4 Flash workers

All four parallel Hermes lanes completed (rc=0, no OAuth blocker):
- lane 1 metadata/tensor comparison — `20260830_055601_1f632b`
- lane 2 Bee loader/dispatch path — `20260830_055601_020ea3`
- lane 3 W2/non-W2 operator mapping — `20260830_055601_2f446f`
- lane 4 benchmark/evidence audit — `20260830_055601_6e75e6`

Primary agent verified every decisive finding: artifact hashes and inventories
(identity-001), loader gating (`qwen35.cpp`), direct-load run (exit 0), and
the medians (aggregate.py). Lane outputs are in
`evidence/P-ARCH-18/2026-08-30/hermes-lanes/`.

## Answers to the gate questions

1. **How fast is original LowGPU in BeeLlama?** 594.037 ms / **3447.60 tok/s**
   median (same-session, n=8; 3409.82 tok/s on the first two captures).
2. **Faster or slower than the hybrid?** Faster: -286.039 ms (-32.50% time;
   1.481x throughput) vs the same-session hybrid 880.076 ms.
3. **Faster or slower than original Escha W2 in Bee?** Faster: -254.810 ms
   (-30.02% time; 1.429x) vs the same-session Escha W2 848.847 ms.
4. **Which path accounts for the difference?** Bee's native GGML quant
   execution replaces the entire escha W2 path: no LUT/dep codebook decode,
   no rin/rout rotation, no separate MMA-partial/finalize staging, and stock
   quant GEMM kernels (dequant + tensor-core/MMQ) handle all 64 layers.
5. **Plausible easier route to ~3059 tok/s parity?** Yes — and it already
   exceeds it: Bee + the original LowGPU quant reaches 3447.60 tok/s, above
   the retained 3058.75 tok/s Escha reference, without any kernel work.
   Selective standard-quant substitution for W2 projections is now the
   cheapest candidate, subject to a quality gate.

## Evidence

`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-18/2026-08-30/`:
- `identity-001/` — model-inspect.py, per-artifact summaries, SHA-256
- `control-001/..007/` — raw A/B/C JSON, stderr, manifests (2 captures per artifact)
- `aggregate.py`, `aggregate.json` — reproducible medians
- `hermes-lanes/` — four worker audits + prompts

Reproducible runner: `scripts/escha-parch18-control.sh`.

**CLOSE P-ARCH-18 (CASE D). NEXT_GATE: selective quant-family substitution
experiment (e.g., FFN or K2-layer standard-quant placement vs escha W2) with
a matched quality + prefill gate; do not change production defaults first.**
