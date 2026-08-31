# P-ARCH-07 — W2 projection chunking root cause

**Status:** `COMPLETE — MIXED: UBATCH CHUNKING PROVEN; MMA SCALING REMAINS DOMINANT`

## Causal answer

The four-to-one Bee/Escha W2 projection-count difference is real and is caused
by Bee's **physical ubatch**, not by a W2 kernel row cap, scheduler
workaround, tensor split, or workspace limit.

The matched run set `n_batch=2048` and `n_ubatch=512`.  The normal single-stream
batch allocator repeatedly calls `split_simple(n_ubatch)`, which returns at most
`n_ubatch` sequential tokens.  Thus the 2,048 persisted IDs are executed as
four 512-token graph evaluations.  Every graph contains the same 400 logical
W2 dense projections, and each calls `ggml_cuda_op_escha_mul_mat()` with an
activation tensor having `n_rows = x->ne[1] * x->ne[2] = 512`.

The custom CUDA op does **not** impose 512: it sizes its rotate buffer,
partial buffer, MMA grid, and finalization grid from `n_rows`; the MMA row
grid is `(n_rows + 127)/128`.  Larger row invocations are already supported.

Therefore the exact mapping is proven:

```
one 2,048-token logical prefill
  Bee, -ub 512  -> 4 physical graphs × 400 W2 projections = 1,600
  Bee, -ub 1024 -> 2 physical graphs × 400 W2 projections =   800
  Bee, -ub 2048 -> 1 physical graph  × 400 W2 projections =   400
  Escha          -> 400 fused escham_code_gemm launches
```

Source chain: `src/llama-context.cpp:555` establishes the physical ubatch;
`src/llama-kv-cache.cpp:1917` selects `split_simple()` for one stream;
`src/llama-batch.cpp:476-507` bounds that split at `n_ubatch`;
`ggml/src/ggml-cuda/escha-moe.cu:1489` obtains its `n_rows` from the graph
activation and `:1810-1823` uses it directly to form the MMA launch grid.

## Controlled experiment

All runs used build `0b035b3a2`, RTX 5090 / SM120, the same model and F16 KV,
FlashAttention enabled, the persisted 2,048-ID file (SHA-256
`695c3609bc35a32003a23be3ba1fbacc16cc94955548c2e855e91661c3f62350`),
the same MMA path, and `-b 2048`.  Profiled W2 times are CUDA-event sums from
the measured pass; full-prefill numbers are independent unprofiled `-r 3`
runs.

| Effective rows | Bee projection launches | W2 linear ms | MMA body ms | Full prefill mean ms | tok/s mean | Runtime result |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 1,600 | 1318.314 | 1221.119 | 1656.993 | 1236.03 | completed |
| 1024 | 800 | 1221.557 | 1142.983 | 1472.931 | 1390.52 | completed |
| 2048 | 400 | 1189.650 | 1119.014 | 1430.285 | 1432.34 | completed, 3/3 |

The 512 W2 row is the P-ARCH-06 measured-pass control.  The 1024 and 2048
profile breakdowns were respectively:

| Rows | Rotation ms | MMA ms | Epilogue ms |
|---:|---:|---:|---:|
| 1024 | 39.254 | 1142.983 | 39.316 |
| 2048 | 29.052 | 1119.014 | 41.584 |

Moving from 512 to 2048 removes 1,200 projected invocations but improves W2
linear time by only 128.664 ms (9.76%); MMA body improves 102.105 ms (8.36%).
So launch/setup/rotation/finalize overhead is not the dominant remaining W2
cost.  Bee still has 673.085 ms of W2 gap against Escha's 516.565 ms control.
This classifies the issue as **D. MIXED**: A / ubatch chunking is a proven,
isolated contributor, while B / Bee's per-invocation MMA implementation is the
larger residual contributor.

## Smallest isolated correction

Use `-ub 2048` (or `--ubatch-size 2048`) with the already-existing
`-b 2048` matched-prefill configuration.  This makes the physical graph cover
the entire logical prompt; no GEMM-path rewrite or dispatch change is needed.
It improves the independent unprofiled mean from 1656.993 ms to 1430.285 ms
(+15.9% throughput).  The P-ARCH-05 median comparison was 1646.329 ms / 1243.98
tok/s, so the 2048-row run is directionally a 216.044 ms / 13.1% full-prefill
improvement, subject to normal run-to-run variance and the different repetition
sets.

Workspace impact is bounded and understood: the W2 op's `u_buf` and `p_buf`
both scale linearly with `n_rows` (`escha-moe.cu:1637-1638` and `:1695`).  For the
widest 5120→17408 projection, raising 512→2048 rows raises these two temporary
buffers by about 15.0 MiB and 102.0 MiB respectively (about **117 MiB**
combined worst-case operator workspace); allocations are pool-backed and
reused rather than retained per layer.

No source code was changed.  The command accepted the identical persisted
stream and all three 2048-row repetitions completed.  A direct logits/checksum
comparison between ubatch settings is not exposed by `llama-bench`; add that
numerical-output assertion in P-ARCH-08 before making 2048 the broad default.

## Gate

P-ARCH-07 is closed for causal explanation and isolated correction discovery.
P-ARCH-08 should validate output equality, peak VRAM telemetry, prompt-length
envelope, and repeated performance stability before promoting this
workload-specific `-ub 2048` setting.
