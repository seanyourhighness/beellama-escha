# P-ARCH-03 — forced SM120 MMA completion revalidation

**Status:** `RESOLVED — FORCED MMA COMPLETE, CORRECT, AND FASTER`  
**Closed prerequisites:** `P-ARCH-01`, `P-ARCH-02`  
**Primary answer:** The current `ESCHA_FORCE_MMA=1` route completes on SM120 with hard process evidence. It selected `mma-fp16` for 800/800 512-row calls, passed the established Escha/SGLang parity suite, and achieved a controlled 2,048-token median of `1243.72 tok/s`.

P001/P002 do not prove a current execution stall. No kernel-internal stall instrumentation was performed.

External evidence directory:
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-03/2026-08-29/`.

## Closed facts carried forward

- **P-ARCH-01 CLOSED:** K2/K3 payloads, all 256 decoded FP16 weights per selected tile, and Bee `ldmatrix` B fragments match Escha's actual HMMA B-fragment ABI bit-for-bit. No inverse permutation is required.
- **P-ARCH-02 CLOSED:** Bee's first post-fragment divergence is its explicit Blackwell dispatch veto at `ggml/src/ggml-cuda/escha-moe.cu:1611-1612`:

```cpp
const bool mma_arch_ok = cc >= GGML_CUDA_CC_TURING
    && (cc < GGML_CUDA_CC_BLACKWELL || getenv("ESCHA_FORCE_MMA") != nullptr);
```

On RTX 5090 / `cc=1200` without `ESCHA_FORCE_MMA`, `mma_arch_ok=false`, `use_mma=false`, and Bee selects `tiled-fma-fp32`.
- Fresh normal-route evidence from Codex WSL JSONL is complete, not pending: `build-cuda/bin/llama-bench`, SHA-256 `0283fb9a56b5544e14ac5b9bc052f8d42b27584a42477ad4963119763efe6114`, graph disabled, profiling enabled, 512 prompt tokens, `tiled-fma-fp32` for 800/800 512-row calls, representative `IC=5120 OC=17408 rows=512 gen=0`, diagnostic `614.562 tok/s`.
- That diagnostic is not the performance baseline. Controlled graph-mode 2,048-token baseline remains `666.312 / 653.131 / 655.468 tok/s`, median **655.468 tok/s**.
- P-ARCH-02 route evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-02/2026-08-29/codex-route-proof-512.log`, SHA-256 `8cec11aea89256c899842b5e6fd2447a8d1de8fad5d8b671ccfeb142dc9a3307`.

## Evidence correction and scope

P001/P002 historically classified forced MMA as a WSL/SM120 stall. A stale Codex progress message was later disproved by the command's own completed tool result, so P-ARCH-03 required hard revalidation. The fresh run below proves the current forced route completes. P001/P002 remain historical context only; they are superseded as evidence of a present-day stall.

## Phase A — controlled revalidation result

**Classification: COMPLETE.** The current forced-MMA route did not reproduce a hang, crash, or timeout.

- Binary: `build-cuda/bin/llama-bench`, SHA-256 `0283fb9a56b5544e14ac5b9bc052f8d42b27584a42477ad4963119763efe6114`.
- Source: Git `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d`; `escha-moe.cu` SHA-256 `c8b609b6db63994cd60a3489de1dbfabb15858ba5a85da516da89c8e3fd05d3d` (untracked local source).
- Environment: `ESCHA_FORCE_MMA=1`, `ESCHA_PROFILE=1`, `GGML_CUDA_DISABLE_GRAPHS=1`.
- Workload: 512 prompt tokens, `-b 2048 -ub 512`, F16 KV, FA on; external timeout 180 seconds.
- Start/end: `14:50:56` to `14:53:21` PDT; elapsed 145 seconds.
- Runner PID `407208`; timeout PID `407341`.
- Shell exit `0`; timeout PID absent after return; complete JSON output; no remaining benchmark or GPU compute process.
- Route: `mma-fp16` for **800/800** 512-row calls; `tiled-fma-fp32` count 0.
- Representative `IC=5120 OC=17408 rows=512` calls completed through timed epilogue.
- Diagnostic throughput: `894.768 tok/s` (single profiled graph-disabled 512-token sample; not the controlled result).
- Evidence: `phase-a-forced-mma-001/`.

The following was the capture contract used:

- `ESCHA_FORCE_MMA=1`
- CUDA graphs disabled
- 512 prompt tokens, `-b 2048 -ub 512`
- profiling enabled where useful to prove route
- same hybrid model and CUDA device
- explicit external timeout only as a bounded safety mechanism

Capture all of the following in the external evidence directory:

1. exact binary path and SHA-256;
2. Git HEAD, dirty state, relevant source SHA-256;
3. complete `ESCHA_*` and graph environment relevant to dispatch;
4. exact command line and timeout value, if any;
5. start/end timestamps and elapsed time;
6. launched shell/benchmark PID and parent PID;
7. CUDA device, compute capability, driver, and pre/post GPU process state;
8. selected route and kernel, when observable;
9. complete stdout and stderr;
10. shell exit code and signal, if any;
11. proof whether the captured process still exists after command return;
12. GPU/process state after command return.

### Hard result classification

- **COMPLETE:** shell/tool returns; exit code captured; benchmark output complete; captured process no longer exists.
- **CRASH:** nonzero exit, signal, CUDA error, or launch failure captured. Treat the explicit error as the new first divergence.
- **TIMEOUT:** bounded external timeout kills a still-running process. Record exactly what remained alive and observable. Timeout alone is not deadlock.
- **PROVEN STALL:** requires stronger, reproducible evidence of no forward progress beyond merely hitting a timeout or appearing pending.

## Phase B — correctness and controlled performance

### Correctness: PASS

The established Bee-vs-Escha/SGLang e3 harness ran with `ESCHA_FORCE_MMA=1`, graphs disabled, temperature 0, seed 42, and the freshly linked `build-cuda/bin/llama-server` (SHA-256 `bdf49d1a1859dc24a42fd325eda6ee913a8bf6b798fac4992bbfa87ba13855f2`). It exited 0 with empty runner stderr.

- P1 conversation: 16/16 prefix tokens.
- P2 factual: 16/16.
- P5 long context: 1,544 prompt tokens, 16/16.
- P6 structured JSON: 16/16.
- P7 tool call: 16/16.
- Report SHA-256: `367f10d2b6cd0f2d862b0ee99dc60139c2ec0cfd60f188ffa2424380c25d8de4`.
- Evidence: `phase-b-parity-001/`.

This is the reference-faithful correctness gate. Exact byte equality to Bee's FP32 tiled-FMA control is not required because the forced path intentionally rounds activations to FP16.

### Controlled 2,048-token performance: COMPLETE

The unchanged production contract ran with `ESCHA_FORCE_MMA=1`, `ESCHA_ALLOW_CUDA_GRAPHS=1`, profiling unset, graph disabling unset, exactly 2,048 prompt tokens, `-b 2048 -ub 512`, F16 KV, FA on, and three repetitions.

- Shell exit `0`; timeout process absent after return; complete JSON output; no remaining benchmark/GPU compute process.
- Elapsed: 129 seconds, below the 300-second safety timeout.
- Samples: `1243.72 / 1229.43 / 1254.32 tok/s`.
- Median: **1243.72 tok/s**; mean `1242.490 tok/s`.
- Normal controlled median: `655.468 tok/s`.
- Improvement: **1.897x**, **+89.745%**, `+588.252 tok/s`.
- Evidence: `phase-b-controlled-2k-001/`.

No internal stall isolation is applicable because the failure did not reproduce.

## Three-lane execution ledger

Only these statuses are permitted: `PROVEN EQUIVALENT`, `PROVEN DIVERGENT`, `UNKNOWN`, `NOT APPLICABLE`.

| Area | Bee normal | Bee forced MMA | Escha HMMA | Evidence | Status |
|------|------------|----------------|------------|----------|--------|
| process completion | Completes; fresh normal probe exited 0 | **Completes**, exit 0, process gone, output complete | Completes in P-ARCH-01 reference invocation | P-ARCH-02 route log; P-ARCH-03 Phase A; P-ARCH-01 profiler | `PROVEN EQUIVALENT` |
| dispatch | `tiled-fma-fp32`, 800/800 | `mma-fp16`, **800/800** | `escham_code_gemm<1,K,128,64,2,true,true>` | Runtime route counts and reference SASS | `PROVEN DIVERGENT` |
| kernel launch | Normal tiled-FMA completes | `escha_matmul_dense_tiled_mma<K,128,128>` route completes | Selected native SM120 kernels complete | Bee source plus completed runtime calls | `PROVEN EQUIVALENT` |
| launch geometry | Known from source and completing | Source geometry completes for all 800 calls | Template known; full native launch source unavailable | Bee source/runtime; reference binary | `UNKNOWN` |
| shared memory | Normal allocation completes | MMA allocation completes in end-to-end calls | Native details unavailable | Bee runtime/source | `UNKNOWN` |
| initial staging | Normal FP32 staging completes | FP16 rotation/synchronous staging completes as part of timed calls | Native input is FP16; detailed staging unavailable | Completed route timings/reference wrapper | `UNKNOWN` |
| first barrier | Normal path completes | Overall forced kernel completes; barrier not separately instrumented | Native details unavailable | End-to-end completion only | `UNKNOWN` |
| first ldmatrix | `NOT APPLICABLE` | Overall MMA kernel completes; fragment ABI proven | HMMA symbol completes | P-ARCH-01 and Phase A | `PROVEN EQUIVALENT` |
| first HMMA | `NOT APPLICABLE` | Overall MMA kernel completes; device code uses MMA primitive | Selected symbols contain HMMA | Bee source/runtime and reference SASS | `PROVEN EQUIVALENT` |
| K loop | Normal completes | Forced MMA completes all 800 calls | Reference call completes | Runtime completion | `PROVEN EQUIVALENT` |
| epilogue/store | Normal completes | Timed epilogue and complete model output produced | Reference output produced | Profile lines, JSON, parity report | `PROVEN EQUIVALENT` |

## Stop conditions

Do not:

- reopen P-ARCH-01 or P-ARCH-02;
- reinterpret `614.562 tok/s` as the controlled baseline;
- remove the Blackwell veto globally;
- optimize decode or tune arbitrary tile sizes;
- broadly rewrite the MMA kernel;
- make throughput claims before correctness;
- trust progress narration as process-state evidence;
- call a timeout a deadlock.

## Final record

```text
Forced-MMA status: COMPLETE
Hard evidence: shell exit 0; complete JSON; timeout process gone; no remaining benchmark/GPU compute process; 800/800 profiled calls completed
Selected route: mma-fp16
Selected kernel: escha_matmul_dense_tiled_mma<K,128,128>
Correctness: PASS — P1/P2/P5/P6/P7 each 16/16 vs Escha/SGLang; P5 prompt 1,544 tokens
First divergence: NONE within forced-MMA execution; prior current-stall premise disproved
Minimal correction: retain opt-in ESCHA_FORCE_MMA=1 while deciding dispatch policy; no kernel correction required
Normal controlled baseline: 655.468 tok/s
Forced-MMA controlled result: 1243.72 tok/s median (1243.72 / 1229.43 / 1254.32)
```

P-ARCH-03 success criteria are complete. Any source-default change remains a separate dispatch-policy decision and was not made here.
