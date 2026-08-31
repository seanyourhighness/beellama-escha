# P-ARCH-05 — qualified Bee MMA vs Escha HMMA performance gap

**Status:** `COMPLETE — MATCHED SHARED-TOKEN PREFILL COMPARISON`  
**Prerequisite:** P-ARCH-04 `PASS`; official Bee SM120 baseline `1230.03 tok/s`.

## Primary question

Why is qualified Bee `escha_matmul_dense_tiled_mma<K,128,128>` materially slower than Escha `escham_code_gemm_kernel<1,K,128,64,2,true,true>` for comparable prefill?

Before attributing any gap, establish the best apples-to-apples Escha measurement possible. The historical SGLang result `2794.545 tok/s` is TTFT-derived for an approximately 2k-token HTTP workload, whereas Bee `1230.03 tok/s` is compute-only `llama-bench` at exactly 2,048 tokens. They are not yet a strict A/B benchmark.

## Starting evidence

- Bee qualified automatic SM120 MMA: controlled samples `1246.82 / 1230.03 / 1217.32`, median **1230.03 tok/s**.
- Historical Escha/SGLang e3: `results/benchmark-results.json`, `prefill2k.prefill_tokens_per_sec = 2794.545`, TTFT `0.572902 s`, usage absent; generated from 7,200 input characters by `source/run_experiment.py:200-266`.
- Historical E0 reference: `2793.620 tok/s` under the same SGLang harness.
- Escha selected kernels for representative M=512: `<1,K,128,64,2,true,true>` with `HMMA.16816.F16`.

## Controlled-comparison gate

Record on the same RTX 5090:

- exact Escha model/runtime/wheel hashes;
- input token count, batch/chunking, graph and attention settings;
- timing semantics and repetitions;
- selected Escha kernel(s);
- completion/error state.

Prefer a compute-only operator or prefill timing comparable to Bee. If only SGLang TTFT is feasible, label remaining semantic differences explicitly and do not state a strict speed ratio.

## Architecture-diff ledger

| Area | Bee MMA | Escha HMMA | Equivalent? | Evidence | Expected Impact | Status |
|---|---|---|---|---|---|---|
| benchmark semantics | Exact 2,048-token compute-only llama-bench | Historical HTTP TTFT-derived ~2k prefill | No | P-ARCH-04; benchmark-results.json | Critical to valid gap | `PROVEN DIVERGENT` |
| selected tensor-core path | `escha_matmul_dense_tiled_mma<K,128,128>` | `escham_code_gemm_kernel<1,K,128,64,2,true,true>` | Both tensor-core, geometry differs | Source/SASS | High | `PROVEN DIVERGENT` |
| tile geometry | CTA BM=128, BN=128 | Template includes 128,64; parameter meanings not fully decoded | Unknown detailed equivalence | Bee source; Escha binary symbols | High | `UNKNOWN` |
| block/warp organization | 256 threads, 8 warps, WN=2, WM=4 | Unknown | Unknown | Bee source only | High | `UNKNOWN` |
| operand staging | Bee synchronous SM120 activation copies, shared decoded weights, ldmatrix | Native details unavailable | Unknown | Bee source / Escha SASS | High | `UNKNOWN` |
| pipeline depth/barriers | Bee barriers each tile; SM120 synchronous staging | Unknown | Unknown | Bee source | High | `UNKNOWN` |
| HMMA issue pattern | m16n8k16 via ggml_cuda_mma | `HMMA.16816.F16` | Primitive family equivalent; issue structure unknown | SASS | High | `UNKNOWN` |
| accumulation | Bee FP32 tile_c | Escha mixed policy/template details partly unknown | Unknown | Source/wrapper/binary | Medium-high | `UNKNOWN` |
| epilogue/fusion | Separate Bee finalize kernel | Escha fused code-GEMM output | Likely divergent; quantify | Runtime/source evidence | High | `UNKNOWN` |
| launch count/scheduling | Bee rotate + matmul + finalize per projection | Escha fused op; internal launches unknown | Unknown | Runtime tracing needed | Medium-high | `UNKNOWN` |

## One-change rule

After a first high-impact architectural divergence is proven and measured, stop broad investigation. Apply one minimal experimental change, then verify correctness, route, relevant profile metric, and controlled 2k throughput. Revert or preserve as negative evidence if it fails.

## Controlled reference run — `escha-controlled-server-002`

`001` remains preserved as historical startup-only evidence. `002` is a fresh supervised run on the same RTX 5090 (SM120, driver 610.88), using the Escha / LowGPU safetensors model, Triton attention, `ESCHA_PREFILL=fused`, mixed accumulation, thinking off, and CUDA graphs captured for batches `1 2 4 8`. Its model, tokenizer, wheel, serve-script, Python, Torch, GPU, and launch fingerprints are recorded in `fingerprints-and-launch.txt`.

The server workload was calibrated against its own usage object, then every warmup/trial asserted `usage.prompt_tokens == 2048`. It retains content, client-rendered token IDs/hash, template settings, calibration, and all JSON responses. HTTP TTFT samples were 0.624309 / 0.621783 / 0.626189 seconds (median 0.624309; 3280.425 tok/s derived). This is explicitly `NOT COMPARABLE` to Bee's compute-only result.

The narrowest available reference boundary was then measured with synchronized `sglang.bench_one_batch` static prefill: 0.612048 / 0.612564 / 0.610817 seconds for batch 1, input length 2048, output length 1, or 3346.14 / 3343.33 / 3352.89 tok/s (median **3346.14**). Bee's official corresponding compute-only gate is **1230.03 tok/s**. These results have partially comparable timing scope, same GPU, and same prompt length, but remain `PROVEN DIVERGENT` in token generator, harness, and model artifact. No exact-same-input ratio or operator/kernel-time ratio is claimed. Raw artifacts are under `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/`.

## Exact shared-token compute-prefill comparison

P-ARCH-05's remaining token-stream gap is now closed at the prefill-region
boundary. `shared-2048.txt` tokenizes with the Escha tokenizer, without special
tokens, to exactly 2,048 IDs. The persisted `shared-2048.ids` file has SHA-256
`695c3609bc35a32003a23be3ba1fbacc16cc94955548c2e855e91661c3f62350`;
independent re-tokenization of the SGLang prompt file reproduced the same count
and hash.

Bee's `llama-bench` gained an opt-in `--prompt-tokens-file` test-harness input
only. It validates count and vocabulary range, leaves its default synthetic
benchmark behavior unchanged, and was rebuilt successfully. Both measured paths
therefore consumed the identical persisted IDs after their respective warmups.

| Layer | Bee | Escha | Comparable? | Evidence | Result |
|---|---|---|---|---|---|
| tokenization / workload | raw `shared-2048.ids`, 2,048 IDs | `shared-2048.txt` re-tokenizes to identical IDs | `PROVEN EQUIVALENT` | `shared-2048.json`, `shared-2048.ids`, independent hash check | Same 2,048-token stream. |
| prefill-region timing | `llama-bench`: 1243.98 / 1225.83 / 1249.86 tok/s; median **1243.98** | synchronized `bench_one_batch`: 3005.18 / 3058.75 / 3319.13 tok/s; median **3058.75** | `PROVEN EQUIVALENT` boundary, with stated harness differences | `bee-shared-input-stdout.json`; `escha-shared-input-prefill.jsonl` | Escha median is 2.458842× Bee at this matched full-prefill boundary. |
| duration at median | 1.646329 s for 2,048 tokens | 0.669554 s for 2,048 tokens | `PROVEN EQUIVALENT` conversion of above rates | same | Difference: 0.976774 s. |
| selected path | `escha_matmul_dense_tiled_mma<K,128,128>` | fused `escham_code_gemm`; Triton attention | `PROVEN DIVERGENT` implementation | P-ARCH-04 route proof; `server.log`; compute logs | This comparison measures the full prefill region, not a single GEMM duration. |
| operator/kernel duration | no new per-kernel measurement | no new per-kernel measurement | `UNKNOWN` | N/A | No operator-level speed ratio is claimed. |

**P-ARCH-05 conclusion:** For the exact same persisted 2,048-token input on the
same RTX 5090, Bee used its qualified
`escha_matmul_dense_tiled_mma<K,128,128>` path and completed full compute-only
prefill in a median 1.646329 s (1243.98 tok/s); Escha used fused
`escham_code_gemm` within the SGLang static prefill region and completed in a
median 0.669554 s (3058.75 tok/s). This is a matched full-prefill-region
comparison, not an HTTP comparison and not a matched individual-kernel result.
