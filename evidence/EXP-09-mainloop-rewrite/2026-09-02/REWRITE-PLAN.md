# EXP-09 — promoted Stage 2 mainloop rewrite plan (Gate 1)

Date: 2026-09-02  
Scope: planning only; no source change, build, benchmark, or promotion is authorized by this document.  
Control: promoted Stage 2 mixed-accumulator `escha_matmul_dense_tiled_mma` at
`1c193ad4c`, with the existing partial-to-`escha_finalize_dense` contract.

## 1. Decision and fixed boundaries

Implement a compile-guarded SM120 candidate that retains Bee's 128-row by
128-column CTA and expresses it as two independent 128x64 column bands. Each
band is owned by four row warps. Within a warp, decoded B is constructed
directly in the two-register `tile_b` fragment and consumed immediately; only
one 8-column B fragment is live at a time. This mirrors the important official
`escham_code_gemm` property—two 64-column bands and warp-local decoded B—while
retaining Bee's input rotation, split-K partial tensor, and separate finalize.

This is deliberately not another EXP-07 layout. EXP-07 v1 made a warp own all
128 rows (`MT=8`) and v2 made a warp own an 8-column slice of all 128 rows,
then replayed the second 64-column pass. Both expanded A/address lifetimes and
either held or generated full-CTA work in one warp (124/154 and 125/141
registers). EXP-09 keeps the control's bounded per-warp output footprint:
32 rows x 64 columns. It never holds a 128-column B tile, a 128-column
accumulator tile, or both 64-column bands in any warp.

Frozen behavior:

- CTA output geometry remains `BM=128`, `BN=128`; block is `dim3(32,8)` and
  grid is `(ceil(n_rows/128), OC/128, n_slices)`.
- K reduction tile remains 16. The order `ti=lo..hi-1` is unchanged.
- `cp.async` double-buffered A staging remains exactly the promoted path,
  including the synchronous fallback.
- Mixed accumulation remains FP16 for `IC <= 6144`, FP32 otherwise, for both
  K2 and K3.
- The partial layout and `escha_finalize_dense` call remain unchanged.
- Generation, ragged, cuBLAS, WMMA, non-SM120, standard-Qwen, model metadata,
  GGUF, rotations, and artifact construction remain untouched.

## 2. CTA, band, warp, and thread geometry

Constants are:

```text
NT=256, NW=8, BANDS=2, WARPS_PER_BAND=4
BM=128, BN=128, BAND_N=64
MT=BM/(16*WARPS_PER_BAND)=2
NTT=BAND_N/8=8
```

Use the existing warp coordinates, renamed to state ownership explicitly:

```text
lane      = threadIdx.x                 // 0..31
warp      = threadIdx.y                 // 0..7
band      = warp & 1                    // 0 or 1
row_group = warp >> 1                   // 0..3
band_col0 = 64*band
warp_row0 = 32*row_group
```

Warp `(row_group,band)` owns rows
`warp_row0..warp_row0+31` and CTA-local columns
`band_col0..band_col0+63`. Its accumulator is `acc[MT=2][NTT=8]`: two
16x8 row tiles by eight adjacent 8-column tiles. The two bands never exchange
B fragments or accumulators. Four warps repeat the cheap decode for their
different 32-row strips of the same band; this duplication is intentional.
Registers cannot be shared between warps, and avoiding the duplication would
reintroduce EXP-07's all-row ownership, cross-warp publication, or full-band
buffering. The official structure demonstrates that warp-local decode can be
the correct trade when its fragment lifetime is short.

### Register budget

Per thread, long-lived accumulator storage is 32 32-bit registers in FP16
mode (`2*8*tile_ah::ne`, with `tile_ah::ne=2`) or 64 in FP32 mode
(`2*8*tile_c::ne`, with `tile_c::ne=4`). The mainloop keeps two A fragments
(8 registers total), one B fragment (2 registers), one prefetched payload
word, and scalar/address temporaries. It must not declare `B[8]`, a decoded
64/128-column array, an all-row A array, or a second-band accumulator.

Budget forecast, to be replaced by `cuobjdump` measurements before any run:

| mode | accumulator | A+B live | forecast total | hard ceiling |
|---|---:|---:|---:|---:|
| FP16 acc | 32 | 10 | about **89 regs/thread** | **97** |
| FP32 acc | 64 | 10 | about **120 regs/thread** | **128** |

The forecast comes from the control (97/128) minus the current eight-fragment
B array (16 registers), plus one streamed B fragment and bounded decode
temporaries. Compilation, not this estimate, decides the gate. Any result over
97/128, any `STACK`/`LOCAL`, spill load/store, or loss of two-CTA/SM residency
rejects the candidate before timing.

## 3. Payload ownership and exact decode math

Keep Bee's payload-ahead staging because it is small and already overlaps the
global code load. For `BN=128`, `NTJ=8`, `NWD=8*K`, and `n_wd=8*K`:

```text
has_pay = tid < NTJ*n_wd
pt      = tid/n_wd                   // output 16-column code tile, 0..7
pw      = tid%n_wd                   // uint32 word, 0..15 (K2) or 0..23 (K3)
ppre    = code[(ti*nct + oc0/16 + pt)*(16*K) as uint32][pw]
```

Publishing `ppre` remains byte-identical:

```text
s_pay[pt][pw].y                  = ppre
s_pay[pt][(pw+1)%NWD].x         = ppre
```

Thus for pair index `dw0`, `.y` is payload word `dw0` and `.x` is payload word
`dw1=(dw0-1+NWD)%NWD`. Unused `s_pay[][NWD..ESCHA_MAX_W-1]` padding is retained
so existing indexing and allocation stay simple.

### Direct `tile_b` construction

For each `j=0..7`, construct exactly one `tile_b B` and immediately use it for
both row tiles `i=0,1`. The Turing/SM120 `tile<8,8,half2>` mapping is:

```text
n8(lane) = lane/4                    // output column inside this 8-col tile
q(lane,l)= 4*l + (lane&3), l=0,1     // K half2 index, 0..7
r0       = 2*q
r1       = 2*q + 1
c_local  = 8*j + n8                  // 0..63 inside owned band
c_cta    = 64*band + c_local         // 0..127
pt       = c_cta >> 4                 // payload tile 0..7
ccl      = c_cta & 15                 // column inside code tile
```

For each of `r=r0,r1`, use the current decode formula without algebraic
substitution:

```text
NB  = 32*NWD
sp  = ((32-K) - K*(escha_dep_pi(r) + 32*ccl + 4*(ccl>>3))) mod NB
g0  = sp >> 5
dw0 = g0 ? NWD-g0 : 0
dw1 = dw0 ? dw0-1 : NWD-1
dsh = sp & 31
idx = __funnelshift_r(s_pay[pt][dw0].y,
                      s_pay[pt][dw0].x, dsh) & 0xffff
w   = escha_codebook_h(idx)
```

Normalize negative `sp` by adding `NB`, exactly as control. `dw1` is written
above to make the ring proof explicit: `s_pay[pt][dw0].x` was populated by
word `dw1`. Do not reverse the two funnel operands.

Then form the fragment with no shared-B store or `ldmatrix` reload:

```text
B.x[l] = __halves2half2(w(r0,c_cta), w(r1,c_cta)), l=0,1
```

This is exactly what `load_generic(tile_b, sw2 + (band_col0+j*8)*8, 8)` would
select: lane `lane` owns physical B row `lane/4` and half2 K index
`4*l+(lane&3)`. Implementation should either use a small force-inlined
`decode_b_fragment<K>()` that writes those two registers or use
`load_generic` only as a debug oracle; it must not materialize a local array.

### Invariance proofs

- Within construction of one `B.x[l]`, `c_cta`, `pt`, and `ccl` are invariant
  between `r0` and `r1`; only `escha_dep_pi(r)` changes.
- Within a lane and fixed `j`, the output column is invariant across both
  half2 registers. Across `l`, `q` selects K pairs 0..3 and 4..7.
- Across lanes, `lane/4` covers each of eight columns four times, while
  `lane&3` covers four K pairs; with `l=0,1` this covers every `(r,c)` in the
  16x8 B fragment once and only once.
- Across `j=0..7`, one warp covers all 64 columns in its band. Across
  `band=0,1`, the CTA covers all 128 columns with no overlap.

### A/B/MMA schedule and barriers

After the current pipeline wait and payload publication:

1. `__syncthreads()` (CTA barrier 1) publishes `s_pay` and `su_cur`.
2. Issue the next tile's `cp.async` to the other A buffer, as control does.
3. Load `A[0]` and `A[1]` from the owned 32-row strip of `su_cur` with the
   existing `load_ldmatrix` mapping.
4. For each `j=0..7`: build one `B`; issue MMA to `acc[0][j]`, then
   `acc[1][j]`; let `B` die before `j+1`.
5. `__syncthreads()` (CTA barrier 2) proves every warp has stopped reading
   `s_pay` and `su_cur` before either can be overwritten/reused. In the sync-A
   fallback, write `su_nxt` only after this barrier; the next iteration's
   barrier publishes it.

There is no decode-publish barrier because decoded B never enters shared
memory. No explicit `__syncwarp`, named barrier, WARPSYNC collective, or
cross-warp shuffle is required: construction is lane-local and the MMA is a
warp-synchronous instruction. Therefore the candidate has **two CTA barriers
per K tile**, versus control's three. A third barrier, pair barrier, or a
band-to-band exchange is a design failure, not a permitted implementation
detail.

The logical work remains 16 `m16n8k16` MMA operations per warp per K tile
(`MT*NTT=2*8`), or **128 warp-level HMMAs per CTA K tile**. Expected static
SASS occurrences are compiler-dependent but should remain control-class
(currently 16 for the FP16 symbol and 32 for the FP32 symbol because of
compiler path duplication). The source/dynamic count and output coverage are
the correctness criteria; static count must not fall below its matched control
symbol.

## 4. Shared-memory budget

Remove only `s_w[BN][16]`. Keep the padded payload and double-buffered A:

```text
s_pay: 8 * 24 * sizeof(uint2)       = 1,536 B
s_u:   2 * 128 * 16 * sizeof(half) = 8,192 B
candidate total                     = 9,728 B
```

Predicted dynamic shared memory is therefore **9,728 bytes/CTA**, down from
13,824. Alignment is naturally at least 8 bytes after `s_pay`; retain the
existing cast/layout and add compile-time size/alignment assertions. Decoded-B
`STS.U16` count must be **zero**; A uses `LDGSTS`/shared loads and payload
publication may emit wider shared stores, but no halfword B store is allowed.

## 5. Accumulator and partial-store contract

### Tile indices

For warp-local `(i,j)` and fragment element `l`:

```text
m_base = 32*row_group + 16*i
n_base = 64*band + 8*j
```

FP32 uses `tile_c acc[2][8]`. Store:

```text
m   = m_base + tile_c::get_i(l)
n   = n_base + tile_c::get_j(l)
row = row0 + m
if (row < n_rows)
    partial[((int64_t)sl*n_rows + row)*OC + oc0 + n] = acc[i][j].x[l]
```

FP16 uses `tile_ah acc16[2][8]`, exactly two half2 registers per thread. The
promoted seam is frozen and copied literally:

```text
v=x[0]: v.x -> tile_c l=0, v.y -> tile_c l=1
v=x[1]: v.x -> tile_c l=2, v.y -> tile_c l=3
```

For source half2 register `h=0,1`, use `tile_c::get_i(2*h+{0,1})` and
`tile_c::get_j(2*h+{0,1})`, adding `m_base/n_base`; guard `row_a<n_rows` and
`row_b<n_rows` independently before converting to float and writing partial.
Do not use `tile_c::ne` to index `tile_ah`, do not transpose `j`, and do not
coalesce the two stores unless exhaustive lane mapping proves the identical
coordinates.

### Tails and split K

- The row-tail load remains zero-filled (`row<n_rows` predicate on the async
  or sync A copy); partial stores retain per-element row guards.
- Column tails do not enter this route: existing `OC % 128 == 0` is required.
- Preserve exactly
  `lo=(nit*sl)/n_slices`, `hi=(nit*(sl+1))/n_slices`. Prefetch only when
  `lo<hi`; loop only `ti in [lo,hi)`. Each slice writes only its existing
  `[sl,row,col]` region.
- Keep the fixed per-slice reduction order inside each partial and the fixed
  `s=0..n_slices-1` summation in `escha_finalize_dense`. The candidate must
  support `n_slices>1`; it may not silently fall back merely because split-K
  is active.

### Bit-compatibility argument

The code payload word, ring predecessor, funnelshift order, `dsh`, 16-bit
mask, `escha_dep_pi`, and `escha_codebook_h` call are identical to control.
The new fragment mapping produces the same two half words that control stores
to `s_w[c][r]` and reloads into `tile_b`; it only removes that round trip.
The K-tile order and the two MMAs contributing to each accumulator are
unchanged. Therefore FP32-acc partials are expected bitwise identical and
FP16-acc partials/output are also expected exact on the same compiler/device.
Any nonzero difference first triggers fragment-capture diagnosis; it is not
accepted as an approximate arithmetic change. `lut`/`dep` argument semantics
and all Escha metadata gating remain unchanged.

Before model parity, add or reuse a debug-only fragment oracle that compares
all manually built `B.x[0..1]` words against control's `s_w`+`ldmatrix` result
for captured K2 and K3 payloads. Exhaustively enumerate `(band,j,lane,l)` and
prove 16x128 coverage exactly once per row warp.

## 6. Implementation sequence and rollback

1. Add the candidate kernel/helper in `ggml/src/ggml-cuda/escha-moe.cu` under
   `#ifdef ESCHA_MMA_SM120_MAINLOOP_REWRITE_EXPERIMENT`. Do not edit the
   promoted kernel body.
2. Restrict host selection to the ordinary Escha `use_mma` prefill predicate,
   K2/K3, `OC%128==0`, and qualified SM120. Macro-off or another architecture
   must enter the existing branch byte-for-byte.
3. Allocate candidate smem with the exact 9,728-byte expression and launch
   the same grid/block/n_slices. Preserve the `IC<=6144` selection.
4. Candidate profile tags are `mma-fp16-mlw` and `mma-fp32-mlw`. Tags appear
   only when the candidate kernel is actually launched. Expected canonical
   route accounting is 672 FP16 + 128 FP32 = 800 candidate records, zero
   `mixedacc`/fallback records in the candidate arm; control remains its
   672/128 mixedacc split.
5. Keep partial allocation and unconditional separate finalize exactly as the
   gate-off path. Do not share code with or revive EXP-08 fused finalize.
6. Add only focused experiment evidence/scripts after implementation. Do not
   update the experiment ledger/current-state until a decisive gate result.

Rollback is mechanical: remove the macro-guarded candidate kernel/helper and
the guarded host selection/tag/smem block, then remove only the fresh
candidate build directory. With the macro absent, preprocessed dispatch and
the promoted kernel must be byte-identical to the starting control. No revert
may alter the control kernel, finalize, rotation, model loader, or artifact.

## 7. Definition of Done and verification gates

All paths and commands below are frozen for implementation. Run from the repo
root. A failed resource, fragment, CUDA, route, parity, or numerical gate stops
the experiment before timing.

### 7.1 Fresh control and candidate builds

```bash
cd '/mnt/d/CODEX WORKSPACE/beellama-escha'
cmake -S . -B build-cuda-exp09-control -G Ninja \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
cmake --build build-cuda-exp09-control -j 12 --target ggml-cuda llama-bench llama-server

cmake -S . -B build-cuda-exp09-mlw -G Ninja \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_FLAGS=-DESCHA_MMA_SM120_MAINLOOP_REWRITE_EXPERIMENT=1
cmake --build build-cuda-exp09-mlw -j 12 --target ggml-cuda llama-bench llama-server
```

Record git SHA, compiler/driver/GPU, configure logs, binary and model SHA-256,
and exit codes. Both builds must be from the same source tree; only the macro
may differ.

### 7.2 Resources and SASS

```bash
OUT='evidence/EXP-09-mainloop-rewrite/2026-09-02'
cuobjdump --dump-resource-usage build-cuda-exp09-control/bin/libggml-cuda.so \
  > "$OUT/control-resources.txt"
cuobjdump --dump-resource-usage build-cuda-exp09-mlw/bin/libggml-cuda.so \
  > "$OUT/candidate-resources.txt"
cuobjdump --dump-sass build-cuda-exp09-control/bin/libggml-cuda.so \
  > "$OUT/control-sass.txt"
cuobjdump --dump-sass build-cuda-exp09-mlw/bin/libggml-cuda.so \
  > "$OUT/candidate-sass.txt"
```

Extract all K2/K3 FP16/FP32 candidate symbols and matched controls. PASS is:
FP16 <=97 regs, FP32 <=128 regs, `STACK=0`, `LOCAL=0`, no `LDL`/`STL` or spill
instructions, launch smem exactly 9,728 B, two source CTA barriers/tile,
decoded-B `STS.U16=0`, no B `LDSM`, and logical HMMA/output coverage unchanged.
Also record static BAR/HMMA/LDGSTS/LDS/STS counts. Any compiler-emitted static
barrier expansion must be explained against the control before proceeding.

### 7.3 Route proof and direct matched smoke

```bash
MODEL='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf'
IDS='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids'
OUT='evidence/EXP-09-mainloop-rewrite/2026-09-02'
for ARM in control mlw; do
  B="build-cuda-exp09-$ARM"
  ESCHA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1 "$B/bin/llama-bench" \
    -m "$MODEL" --prompt-tokens-file "$IDS" -p 2048 -n 0 -ngl 99 \
    -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json \
    >"$OUT/$ARM-profile.json" 2>"$OUT/$ARM-profile.stderr"
  GGML_CUDA_DISABLE_GRAPHS=1 "$B/bin/llama-bench" \
    -m "$MODEL" --prompt-tokens-file "$IDS" -p 2048 -n 0 -ngl 99 \
    -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 5 -o json -oe json \
    >"$OUT/$ARM-smoke.json" 2>"$OUT/$ARM-smoke.stderr"
done
```

Candidate route proof must contain exactly 800 intended records: 672
`mma-fp16-mlw`, 128 `mma-fp32-mlw`, zero fallback/predicate mismatch; control
must have the same shapes/counts on `mixedacc`. Compare every family using
steady-state total projection time and matmul time. Before the final campaign,
require no family >5% slower and a credible aggregate mainloop improvement.

### 7.4 Numerical fragment, partial, tails, and split-K gates

- K2 and K3 captured fragment oracle: every candidate `B.x` word equals the
  control ldmatrix fragment word.
- Identical-input packed-op output: FP16 and FP32 families exact, finite, and
  no unwritten sentinel values.
- Row cases `1,15,16,17,127,128,129,512,1544,2048,4096` (within VRAM): exact
  output and no OOB/CUDA error.
- Force/observe both `n_slices=1` and `n_slices>1`; compare every partial and
  final output exactly. Confirm `lo/hi` coverage partitions `[0,nit)` once.
- Run compute-sanitizer on the smallest K2/K3 tail cases before full-model
  parity.

### 7.5 P2/P7 and decode

```bash
MODEL='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf'
OUT='evidence/EXP-09-mainloop-rewrite/2026-09-02'
for ARM in control mlw; do
  ESCHA_SERVER_BIN="$PWD/build-cuda-exp09-$ARM/bin/llama-server" \
    python3 scripts/escha-compare/run_compare.py --model "$MODEL" \
      --only P2-factual,P7-tool-call --max-new-tokens 16 --ctx-size 4096 \
      --outdir "$OUT/parity-$ARM"
  GGML_CUDA_DISABLE_GRAPHS=1 "$PWD/build-cuda-exp09-$ARM/bin/llama-bench" \
    -m "$MODEL" -p 0 -n 64 -ngl 99 -b 2048 -ub 2048 \
    -ctk f16 -ctv f16 -fa on -r 5 -o json -oe json \
    >"$OUT/$ARM-decode-r5.json" 2>"$OUT/$ARM-decode-r5.stderr"
done
```

P2 factual and P7 tool-call must each retain 16/16 greedy seed-42 agreement,
with no CUDA error/hang. Candidate median decode tok/s may regress by at most
2% versus control. Generation routes must not use `mlw`.

### 7.6 Canonical matched 9-pair ABBA campaign (graphs ON)

Use fresh process per trial, one unrecorded warmup per arm, and the exact
EXP-08 order `AB BA BA AB AB BA BA AB AB`:

```bash
REPO='/mnt/d/CODEX WORKSPACE/beellama-escha'
CTL="$REPO/build-cuda-exp09-control/bin/llama-bench"
CAND="$REPO/build-cuda-exp09-mlw/bin/llama-bench"
MODEL='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf'
IDS='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids'
OUT="$REPO/evidence/EXP-09-mainloop-rewrite/2026-09-02/bench"
mkdir -p "$OUT/noise-run"
export PATH="/usr/lib/wsl/lib:$PATH"
ARGS=(-p 2048 -n 0 -ngl 99 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json)
"$CTL" -m "$MODEL" --prompt-tokens-file "$IDS" "${ARGS[@]}" \
  >/dev/null 2>"$OUT/noise-run/warmup-control.stderr"
"$CAND" -m "$MODEL" --prompt-tokens-file "$IDS" "${ARGS[@]}" \
  >/dev/null 2>"$OUT/noise-run/warmup-candidate.stderr"
i=0
for PAIR in AB BA BA AB AB BA BA AB AB; do
  i=$((i+1))
  for POS in 1 2; do
    ARM=$(printf '%s' "$PAIR" | cut -c"$POS")
    if [ "$ARM" = A ]; then NAME=control; BIN="$CTL"; else NAME=candidate; BIN="$CAND"; fi
    "$BIN" -m "$MODEL" --prompt-tokens-file "$IDS" "${ARGS[@]}" \
      >"$OUT/noise-run/p${i}t${POS}-${NAME}.json" \
      2>"$OUT/noise-run/p${i}t${POS}-${NAME}.stderr"
  done
done
python3 evidence/EXP-08-fusedfinalize/2026-09-01/scripts/analyze-campaign.py \
  "$OUT/noise-run" | tee "$OUT/analysis.txt"
```

Before execution, copy/adapt the analyzer only if its CLI does not accept the
directory argument; do not alter trial order or statistics. Report all raw
samples, median/mean/CV per arm, paired candidate/control tok/s ratios,
geometric mean and 95% t-CI, candidate-faster count, hashes, graphs-on proof,
VRAM, and exit codes. With both arm CVs <=2%, median candidate tok/s gain >=5%
is SMALLER POSITIVE and eligible for promotion review; >=20% is BREAKTHROUGH.
If either CV exceeds 2%, require paired tok/s `G>=1.05`, lower 95% CI >1.0,
and candidate faster in >=8/9 pairs. Gain <2% is REJECT+revert; 2–5% is
informative only and requires an explicit promotion decision after all other
gates. Decode regression >2% always rejects.

## 8. Risks and watch-outs

- **Register allocation:** direct decode arithmetic can be unrolled into many
  temporaries. Force the `j` body into a small inlined helper or scoped block,
  keep one B fragment live, and inspect SASS after each implementation round.
  Do not solve pressure with `maxrregcount` spilling.
- **Decode duplication:** each of four row warps decodes its band. This is the
  planned official-style trade. Do not add cross-warp exchange unless Terra
  approves a new design; it changes the experiment and risks EXP-07 again.
- **Barrier safety:** exactly two CTA barriers/tile. The first publishes
  payload/current A; the second protects payload/current-A reuse. Removing the
  second races the next publication; adding a decoded-B barrier defeats the
  mechanism.
- **Shared memory:** dynamic allocation is exactly 9,728 B. Removing payload
  padding or overlaying A/payload is unnecessary risk. `s_w` must not survive
  under another name.
- **Fragment order:** funnel operands, ring predecessor, half2 low/high order,
  and `tile_b` physical mapping are the highest correctness risks. Prove them
  with captured K2/K3 fragments before model tests.
- **FP16 store seam:** preserve both halves of both `tile_ah` registers and map
  via tile-c coordinates `l=0..3`. EXP-08 showed this seam is easy to
  transpose incorrectly.
- **Tail rows:** zero-fill loads and independently guard both FP16 output
  stores. Never infer full rows from the canonical M=2048 benchmark.
- **Split K:** do not restrict the candidate to one slice, change target
  occupancy/slice policy, reorder K tiles, or fuse finalize.
- **SM120 scope:** do not route older CUDA/HIP/Vulkan, WMMA, ragged, cuBLAS,
  decode, or standard-Qwen execution through the candidate.
- **Do not touch:** `escha_finalize_dense`, input/output Hadamards, `rin/rout`,
  model metadata gates, GGUF conversion/artifact names, mixed-acc threshold,
  BM/BN/grid, cp.async policy, benchmark tokens, or any EXP-07/08 code.

## 9. Terra review flags (decide before implementation)

1. **Approve the intended duplication trade:** four row warps independently
   reconstruct the same band's streamed B fragment; no cross-warp sharing.
   Recommendation: approve, because this is what bounds ownership to 32x64
   and avoids the failed all-row/full-tile lifetimes.
2. **Approve direct fragment construction as the primary implementation:**
   manual two-`half2` writes using the `tile_b` mapping above, with
   `load_generic`/control ldmatrix only as an oracle. Recommendation: approve;
   a shared scratch fragment would restore STS/LDS and a barrier.
3. **Confirm hard-gate treatment of the register forecast:** 89/120 are
   planning estimates; measured <=97/128 with zero spills is pass, even if the
   exact estimates differ. Recommendation: keep the ceilings decisive rather
   than requiring the estimates exactly.

