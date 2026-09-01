# ESCHA-W2 PREFILL — next-focused runtime plan

Scope: `/mnt/d/CODEX WORKSPACE/beellama-escha`, branch `escha-w2-prefill`,
HEAD `463cbcd88043f69e7237bb1437b6c539d96798a8`. Canonical full-Escha artifact
only; no artifact reconstruction/substitution. Every new path remains narrowly
Escha-metadata- and architecture-gated, opt-in until promoted.

## 1. Status summary

- EXP-01 is banked: the SM120 double-buffered `cp.async` A-stage is now the
  default (`215aa4ac3`). Matched 2K moved from 1412.5 ms / 1449.9 tok/s to
  889.5 ms / 2302.4 tok/s (**+58.8%**); default reproduction was 2306.5 tok/s.
  Decode, P2/P7, standard-GGUF isolation, route proof, and the milestone medium
  5-pack (65/75) passed.
- EXP-02 is rejected and reverted. Direct K2 decode into warp-owned MMA B
  fragments produced 916.0 ms / 2235.7 tok/s versus 880.1 ms / 2327.0 tok/s
  (**-3.9%**). The target projection was about 2.5 ms versus about 2.0 ms on
  shared-B async. Its 176 registers/thread reduced residency, and each M-warp
  repeated codebook/dependency work. Current `escha-moe.cu` is identical to
  `215aa4ac3` and contains no direct-fragment route.
- The packed matmul remains the target: P011 measured 95.2% of projection time
  in matmul. P-ARCH-13 (standalone 128x64), P-ARCH-14 (single-slice finalize
  fusion), and P-ARCH-20 (FP16 accumulator only) were respectively -0.78%,
  neutral, and -44.22% at their decisive gates.

## 2. Gap to 3000–3600 tok/s

Using the promoted 889.5 ms / 2302.4 tok/s result:

| target | throughput gain | 2K latency | latency to remove |
|---|---:|---:|---:|
| 3000 tok/s | +30.3% | 682.7 ms | 206.8 ms |
| 3339 tok/s reference | +45.0% | 613.4 ms | 276.1 ms |
| 3600 tok/s | +56.4% | 568.9 ms | 320.6 ms |

The often-quoted approximately 70 ms / 260 tok/s residual belongs to the
P-ARCH-23G artifact-substitution trajectory (about 3079 -> 3339 tok/s), not the
current full-Escha 2300 tok/s baseline. It must not be used to understate this
phase's runtime gap.

Historical attribution says K2 is the largest directly measured post-async W2
residual; K3 was already close to the matched reference. A runtime-side packed
GEMM/schedule correction could plausibly recover roughly 150–250 ms if it
approaches the better execution structure across all important K2 families.
Transform/launch/graph work is more plausibly another 10–30 ms combined. Thus
3000 is credible but requires a broad K2 win, not a one-shape curiosity; 3339 is
an aggressive runtime-parity goal; 3600 is presently a stretch with no measured
support. Artifact-side substitutions demonstrate a ceiling but contribute 0%
to this runtime-only plan.

## 3. Current kernel facts and concrete levers

The promoted `escha_matmul_dense_tiled_mma<K,128,128>` uses 256 threads (8
warps), a 128x128 CTA output tile, two column-warp groups and four row-warp
groups. Each warp holds `MT=2` by `NTT=8` FP32 accumulator fragments. A is
double-buffered through `cp.async`; B is still synchronously decoded by the CTA
into `[128][16]` shared half storage, published by a CTA barrier, reloaded with
`ldmatrix`, then consumed by HMMA. There are three CTA barriers per 16-wide K
tile. The host chooses split-K from a target of 512 CTAs; at matched M=2048 the
dominant 5120->17408 shape is naturally unsplit.

The compiled async kernel is **128 registers/thread**, 13,824 B dynamic shared
memory, `__launch_bounds__(256,1)`. Register capacity therefore limits it to at
most two such CTAs/SM (16 resident warps) before other limits; a measured SM120
occupancy counter is unavailable (`ERR_NVGPUCTRPERM`), so do not present an
estimated percentage as measured occupancy. EXP-02 at 176 registers/thread
falls to at most one CTA/SM. Important levers still open are CTA M/N aspect
ratio while preserving CTA count, cooperative B-decode amortization, B-stage
pipeline depth, accumulator/decode live-range separation, and split-K/grid
policy. EXP-01 already solved A-stage overlap; do not duplicate that work.

## 4. Ranked next experiments

### 1 — EXP-03: shared-B 256x64 balanced tile (highest information/value)

**Hypothesis.** Keep the proven shared-B/`ldmatrix`/FP32-HMMA path, but change
K2 prefill from BMxBN 128x128 to **256x64**. Tile area, 256 threads, accumulator
fragments/thread, CTA count, partial layout, arithmetic, and split-K remain
constant. Each CTA decodes half as many B values and reuses them across twice
as many rows; A traffic doubles, but that stage is now asynchronously overlapped.
This directly tests the EXP-02 lesson: amortize codebook work across more M rows
without warp-local duplicate decode or added accumulator pressure. P-ARCH-13's
128x64 changed tile area and doubled CTA count, so it does not close this test.

Expected value: 5–12% full-prefill if decode/issue serialization is material;
risk: medium-high (larger double-buffered A footprint, row-tail efficiency, and
the compiler may not preserve 128 registers). Apply to K2 only initially; K3 is
not a target.

**Smallest discriminating probe (after adding only
`ESCHA_MMA_SM120_K2_BM256_BN64_EXPERIMENT` and a distinct route tag):**

```bash
cd '/mnt/d/CODEX WORKSPACE/beellama-escha'
cmake -S . -B build-cuda-exp03-control -G Ninja -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda-exp03-control -j 12 --target llama-bench llama-server
cmake -S . -B build-cuda-exp03-k2-256x64 -G Ninja -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_FLAGS=-DESCHA_MMA_SM120_K2_BM256_BN64_EXPERIMENT=1
cmake --build build-cuda-exp03-k2-256x64 -j 12 --target llama-bench llama-server

MODEL='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf'
IDS='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids'
OUT='/tmp/escha-prefill/EXP-03-k2-256x64'; mkdir -p "$OUT"
for ARM in control k2-256x64; do
  B="build-cuda-exp03-$ARM"
  ESCHA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1 "$B/bin/llama-bench" \
    -m "$MODEL" --prompt-tokens-file "$IDS" -p 2048 -n 0 -ngl 99 \
    -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json \
    >"$OUT/$ARM-profile.json" 2>"$OUT/$ARM-profile.log"
  GGML_CUDA_DISABLE_GRAPHS=1 "$B/bin/llama-bench" \
    -m "$MODEL" --prompt-tokens-file "$IDS" -p 2048 -n 0 -ngl 99 \
    -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 5 -o json -oe json \
    >"$OUT/$ARM-bench.json" 2>"$OUT/$ARM-bench.stderr"
  GGML_CUDA_DISABLE_GRAPHS=1 "$B/bin/llama-bench" -m "$MODEL" \
    -p 0 -n 64 -ngl 99 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on \
    -r 5 -o json -oe json >"$OUT/$ARM-decode.json" 2>"$OUT/$ARM-decode.stderr"
done
cuobjdump --dump-resource-usage build-cuda-exp03-control/bin/libggml-cuda.so \
  >"$OUT/control-resources.txt"
cuobjdump --dump-resource-usage build-cuda-exp03-k2-256x64/bin/libggml-cuda.so \
  >"$OUT/k2-256x64-resources.txt"
ESCHA_SERVER_BIN="$PWD/build-cuda-exp03-k2-256x64/bin/llama-server" \
  python3 scripts/escha-compare/run_compare.py --model "$MODEL" --only P2,P7 \
  --max-new-tokens 16 --ctx-size 4096 --outdir "$OUT/smoke"
```

Route proof must show 128 target K2 calls on the new tag and the other 672
prefill calls on `mma-fp16`, with no fallback. Aggregate `matmul_ms` for both
the 5120->17408 family and all K2 families; profiling is attribution only, not
the wall-speed result.

**KEEP:** target-family matmul improves >=10%, aggregate K2 >=7%, full 2K >=5%,
at least 4/5 candidate samples beat the control median, CV <=2%, no K2 family
regresses >5%, resources show no spills and <=128 registers, P2/P7 retain the
control's 16-token agreement, and decode is no worse than 2%. A 3–5% wall gain
gets one reversed-order confirmation; <3%, >128 registers/spills, or a route,
CUDA, output, tail, or family gate failure is REJECT.

**Rollback:** remove the compile-guarded candidate block/route tag and delete
only the fresh candidate build directory; the default 128x128 async path must
remain byte-for-byte unchanged. Do not promote until 128/512/1544/2048/4096 and
row-tail coverage pass.

### 2 — cooperative shared-B pipeline / register-lifetime scheduling

If EXP-03 proves decode amortization, double-buffer B and separate next-tile
payload/decode lifetime from current-tile MMA without duplicating decode across
M-warps. Test one pipeline-depth/ownership change at a time. Expected 3–8% wall;
high implementation risk (barriers, extra shared memory, register overlap). A
prototype is viable only with no spills, <=128 regs, >=7% aggregate-K2 matmul
and >=3% wall. Prefer this over reviving EXP-02.

### 3 — shared input-transform/projection fusion (old #4)

First measure the upper bound for FFN up+gate and qkv+z pairs. Historical rotate
is only tens of milliseconds, so expected wall value is 1–4%. Prototype FFN
up+gate only if duplicate rotate plus launch/host gaps exceed 3% of current wall;
rotate once, launch two unchanged packed GEMMs, preserve outputs and aliasing.
Medium graph/workspace risk; keep for >=3% wall and <=128 MiB extra workspace.

### 4 — split-K/grid policy on short-output K2 families (part of old #3/#6)

The dominant 17408-output M=2048 call is already unsplit, but smaller-output
families can still be over/under-sliced by the fixed 512-CTA target. Sweep only
the target/slice policy on the unchanged shared-B kernel and report every K2
family. Expected 0–4% wall, low code risk, medium numerical/partial-buffer risk.
Keep only >=3% wall with fixed slice order, <=128 MiB workspace delta, and no
family >5% slower. This is not a microbatch-special-case license.

### 5 — CUDA graphs (old #5)

Graphs were near-neutral and cannot remove the measured GEMM body. Revisit only
after a kernel win, using the same binary and alternating cold-process A/B/B/A
with exact IDs. Expected 0–3%; low kernel risk, medium capture/runtime risk.
Require proved capture/replay and >=3% across two independent five-sample sets.

### 6 — LUT/dependency access and microbatch (old #6)

The hot kernel computes `escha_codebook_h()`/`escha_dep_pi()` and does not use
the passed LUT/dep tensors. A LUT rewrite risks random/cache traffic and has no
current stall proof. Reopen only after SASS or a matched microprobe identifies
codec latency; require >=5% matching-K2 matmul and >=3% wall. Keep `-ub 2048`
as the primary contract; any 512/1024/2048 sweep is a broad envelope check, not
benchmark-token dispatch. Expected 0–3%; low priority.

## 5. Direct-fragment retry decision

Do **not** retry EXP-02 next. Reducing 176 to merely 120–128 registers would
still leave only two resident 256-thread CTAs and would not cure duplicated
codebook work. One later retry is justified only by a materially different
design that (a) performs CTA-cooperative decode once, (b) has compile proof of
**<=80 registers/thread or an occupancy calculation showing >=3 CTAs/SM**, (c)
has zero local-memory spills, and (d) preserves async A overlap. Without that
design proof, the shared-B 256x64 amortization/pipeline route is more promising.

## 6. Avoid / closed paths

- No P-ARCH-23I or other body/weight/vocabulary substitution; it is an artifact
  ceiling, not the goal architecture.
- No standalone 128x64 repeat, finalize-only fusion, FP16-acc-only, WMMA,
  per-call full dequant+cuBLAS, TF32, or scheduler work based on the disproved
  2–4-row prefill premise.
- No EXP-02 rerun without the <=80-register/cooperative-decode redesign above.
- No filename/model-name/benchmark gates, combined multi-variable patches,
  stale builds, best-of-N reporting, or profiler-instrumented throughput claims.

Primary evidence: `docs/escha-prefill-experiment-ledger.md`, P-ARCH-10/11/13/14/20
manifests, and current `escha-moe.cu`. GBrain corroboration: **“Escha P-ARCH-10
takeover: BeeLlama↔Escha investigation state & worker plan (2026-08-29)”** and
**“BeeLlama Escha architecture-diff ledger — handoff.”** Repository evidence is
authoritative where historical GBrain snapshots differ.
