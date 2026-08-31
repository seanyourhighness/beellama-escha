# Escha × LowGPU runtime audit

Primary artifact: `escha-w2-lowgpu-mono-parity.gguf` (8,619,127,360 bytes),
Qwen3.5 hybrid, 64 layers, hidden size 5120, vocabulary 248,320.

| component | working | broken / missing | best reference |
| --- | --- | --- | --- |
| GGUF metadata and tensor mapping | Yes: native Qwen3.5 mapping loads 2,058 tensors and recognizes `qwen35.escha.version=1` and `qwen35.lowgpu.version=1`. | None found in the validated artifact. | KaedeTai reference GGUF and Ajay9o9 `escha-w2-dense` |
| Packed Escha weights | Yes: 400 I16 code tensors, shared F16 LUT, I16 dependency tables, and F16 scales remain packed in the CUDA model buffer. | SM89 runtime still needs a physical Ada test. | Ajay9o9 `escha-w2-dense` |
| CPU/reference reconstruction | Yes: the existing reference reconstruction matches the Escha tile decoder; deterministic SGLang token parity is recorded. | No standalone all-tensor conformance runner. | YanissAmz `feature/escha-format`, KaedeTai codec |
| CUDA Escha decode | Yes: native CUDA dispatch runs all 65 layers in the validated model. | No RTX 4070 Ti runtime measurement. | Ajay9o9 `escha-w2-dense` |
| CUDA Escha prefill | Yes: route-tagged profiling proves true prefill reaches `rows=512` on the Blackwell FP32 tiled-FMA route; production graph-mode P000-R2 measured 658.303 prompt tok/s mean (655.468 median) at exactly 2,048 tokens with `llama-bench`. | Operator-level bitwise candidate harness and Ada-specific profiling are missing. | Current FP32 tiled-FMA control |
| LowGPU embedding / lm_head | Yes: packed i8 codes, F16 scales, i8 zero-points; CUDA head profile confirms dispatch. | The hybrid schedule calls the head with one token, so normal end-to-end inference does not exercise its multi-token GEMM path. | Existing LowGPU implementation and GGUF layout |
| LowGPU multi-token prefill | Direct CUDA regression test passes with 6 tokens and 17 packed vocabulary rows; all logits match an independently packed 3-bit reference. Full-vocabulary FP16 expansion was replaced with 4,096-row stripes (~40 MiB temporary weights). | Needs only Ada runtime profiling on the actual target GPU. | cuBLAS GEMM layout in the prior implementation |
| GGML graph / backend dispatch | Yes: ESCHA and LowGPU operators are registered in GGML, CPU, CUDA, and Qwen3.5 wiring. | CUDA-graph behavior on Ada is unverified. | Ajay9o9 `escha-w2-dense` |
| Deterministic generation | Yes: P1, P2, P6, P7 each matched 16/16 SGLang reference tokens at temperature 0/seed 42. | Full continuations were not re-run after this change. | SGLang e3 suite |
| Long context | Yes: P5, 1,544 prompt tokens, matched 16/16 reference continuation tokens. | 4k+ context and 12 GB KV-capacity measurements remain to run on RTX 4070 Ti. | SGLang e3 suite |

## Current validation evidence

- `build-cuda-verify`: fresh SM120 CUDA build; `llama-cli` and `llama-server` linked.
- `build-cuda-sm89/.../lowgpu.cu.o`: modified LowGPU unit compiled for Ada SM89.
- `/tmp/escha-parity-striped-20260829/compare-report.json`: four short deterministic prefix checks.
- `/tmp/escha-longctx-striped-20260829/compare-report.json`: long-context prefix check.
- `build-cuda-verify/bin/test-lowgpu-prefill`: six-token CUDA prefill conformance test; `LOWGPU_PROFILE` reported `tokens=6`.

The final hardware acceptance gate is still an actual RTX 4070 Ti 12 GB run.

The earlier 326.86 tok/s server result remains historical end-to-end evidence and
must not be compared directly with the current compute-only `llama-bench` control.
The former interpretation that recurrent execution forced all Escha prefill calls
through token-sized rows is superseded by the route-tagged 512-row profile.
