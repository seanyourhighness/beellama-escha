# P-ARCH-17 — original Escha W2 inside BeeLlama

**Status: CLOSED — Case B, material but partial hybrid-model delta.**

## Scope and control

No CUDA kernel, K2 geometry, K3, ubatch, graph default, or production default
was changed. `docs/escha-architecture/PROTOCOL.md` was not present in this
checkout; the retained protocol boundary in P-ARCH-05/15 and the exact shared
ID file were used instead.

Both Bee runs used the same RTX 5090, candidate `llama-bench`, `-p 2048 -n 0
-b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on`, CUDA graphs enabled, and the
immutable 2,048 raw IDs:

- IDs: `shared-2048.ids`, SHA-256
  `695c3609bc35a32003a23be3ba1fbacc16cc94955548c2e855e91661c3f62350`.
- Binary: SHA-256
  `32911ef90000dfc31d7149d5cf9897e7b087547f7ac87ff348d8ebbb97865edc`.
- Runtime CUDA library: SHA-256
  `695001fedab34997a548f196571ce4ceaf07fc56ae16e67c4caf4b56093d0cd5`.
- Source base: detached `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d`;
  binary hashes are authoritative because the worktree is dirty.

## Artifact parity

| property | Original Escha W2 | Current hybrid | result |
|---|---:|---:|---|
| path | `models/escha-w2-original/Escha-Qwen3.8-27B-W2.gguf` | `escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf` | direct-load compatible |
| SHA-256 | `0d326e18a2cd7d9b279271f3f3ea1131d45868aac6bf2ba30156e0a75612ebbd` | `e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d` | pinned |
| bytes | 12,691,575,008 | 8,619,127,360 | different vocab representation |
| GGUF / architecture | v3 / `qwen35`, 64 layers, 5120 embed, 17408 FFN | same | compatible |
| tensors | 2,054 | 2,058 | 2,052 common descriptors exactly match name/shape/type |
| W2 projections | 400 (272 K2, 128 K3) | same | same topology |
| embedding / output | `token_embd.weight`, `output.weight`, F16 | six LowGPU code/scale/zp tensors | explicit delta |
| metadata | no `qwen35.lowgpu.version` | `qwen35.lowgpu.version=1` | only functional metadata delta |

The hybrid is **not only a vocabulary substitution**. Byte-level payload hashes
over the 2,052 common tensors find 1,271 equal and 781 different: 288 Escha
sidecars in the 48 GDN layers, plus 144 biases, 157 norms, and 192 SSM tensors.
All FFN K2/K3 W2 tensors and the LUT/dependency tensors are byte-identical.
Tensor ordering differs physically but the Bee loader consumes named tensors;
it performs no automatic dequant/repack on a compatible Escha sidecar.

## Bee execution-path proof

The original artifact loaded directly: no adapted artifact was needed. The
Qwen3.5 loader selects native `GGML_OP_ESCHA_MUL_MAT` from the same sidecar
names/shapes; missing or incompatible sidecars fail load rather than fall back.

Paired route-only captures (`ESCHA_PROFILE=1`, graphs deliberately disabled,
not used for throughput) each exited 0 with 800 records. Their ordered
`k|ic|oc|rows|gen|route` schema is byte-identical. The measured half has 272
K2 and 128 K3 records, all `mma-fp16`; there is no cublas, WMMA, or tiled-FMA
route. Thus the async/HMMA prefill route and K2/K3 launch topology are the
same under the matched shape/ubatch control.

## Matched 2k result

Two four-sample, uninstrumented captures were run in alternating artifact
order. Median below is across all eight raw samples per artifact; it is the
reciprocal of median wall time, not an average of reported rates.

| Bee artifact | median prefill ms | median tok/s | vs hybrid wall |
|---|---:|---:|---:|
| current hybrid | 905.168 | 2262.56 | — |
| original Escha W2 | **844.617** | **2424.77** | **-60.551 ms (-6.69%)** |
| retained Escha-runtime reference | 669.554 | 3058.75 | reference only; different runtime harness |

Against the contemporaneous Bee-hybrid residual to the retained Escha reference
(235.614 ms), the original artifact removes 60.551 ms, or **25.70%**. It does
not close the remaining 175.063 ms runtime/harness residual.

## Attribution and decision

| stage / difference | Hybrid Bee | Original W2 Bee | conclusion |
|---|---|---|---|
| K2 W2 route/count | 272, `mma-fp16` | 272, `mma-fp16` | topology identical; payloads differ only in GDN subset |
| K3 W2 route/count | 128, `mma-fp16` | 128, `mma-fp16` | topology identical; FFN K3 payloads equal |
| embedding | LowGPU sidecars | F16 tensor | model-level candidate |
| LM head | LowGPU sidecars | F16 tensor | not executed by `-n 0` prefill test |
| GDN / non-W2 body | 781 common payload differences total | original values | model-level candidate |
| graph / scheduler | same binary, IDs, batch, ubatch, graph setting | same | held constant |
| copies / conversion | no loader conversion path | no loader conversion path | no silent fallback |

An attempted `LOWGPU_PROFILE=1` diagnostic is excluded from attribution: it
printed its first one-token event then aborted with CUDA error/exit 134 in
`ggml_cuda_op_lowgpu_mul_mat`. The uninstrumented hybrid controls exit 0, so
this is an instrumentation-path limitation, not a model or kernel correctness
failure.

**Decision:** Case B. The original artifact materially improves Bee prefill,
but leaves most of the matched Escha-reference residual. Since the two model
deltas (F16 vocab boundary and GDN payload lineage) are confounded, the first
responsible tensor-level difference is not yet isolated. Preserve the 60.551-ms
model opportunity as P-ARCH-18, *hybrid model delta root cause*, before any
new-kernel work; P-ARCH-17 does not reopen P-ARCH-01..16.

## Evidence

`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-17/2026-08-30/`:

- `control-001/` and `control-002/`: raw A/B JSON, stderr, manifests, alternating order.
- `route-001/`: paired complete route captures and aggregates.
- `lowgpu-trace-001/`: diagnostic-only abort/backtrace; excluded from timing.

Reproducible runners: `scripts/escha-parch17-control.sh` and
`scripts/escha-parch17-route.sh`.
