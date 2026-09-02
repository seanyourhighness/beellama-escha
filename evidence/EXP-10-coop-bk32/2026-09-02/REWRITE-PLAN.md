# EXP-10 — cooperative BK32 double-buffered-B rewrite plan (Gate 1)

Date: 2026-09-02  
Scope: planning only. This document authorizes no source edit, build, benchmark,
promotion, commit, or push.  
Control: promoted Stage 2 mixed-accumulator
`escha_matmul_dense_tiled_mma<K,128,128,FP16_ACC>` at branch HEAD
`969db62df`, retaining the existing rotate -> partial ->
`escha_finalize_dense` operator contract.  
Decision: Sean's Decision B; this is the single authorized cooperative packed
runtime experiment. A resource failure or a performance result below the frozen
gate ends incremental work on this kernel.

## 1. Gate-1 decision

Implement one compile-guarded SM120 candidate named
`escha_matmul_dense_tiled_mma_coop_bk32`. It keeps the exact control CTA,
decode ownership, shared publication, MMA geometry, mixed-accumulator policy,
split-K partition, partial layout, and finalize order. The only mainloop change
is temporal:

- decoded B is a two-slot shared ring, one 4,096-byte `[BN][16]` tile per slot;
- after tile `N`'s A fragments are loaded, even warps execute
  `decode(B[N+1]) -> MMA(N)` while odd warps execute
  `MMA(N) -> decode(B[N+1])`;
- every one of the 256 threads still produces its original eight unique B
  weights for `N+1`; no row warp reconstructs another warp's weights and no B
  value is warp-private;
- B fragments are loaded by the same `ldmatrix` mapping as control and consumed
  one `j` fragment at a time. This shortens, rather than expands, B-register
  lifetime while preserving the per-accumulator K-tile order.

The warp-parity branch is uniform within every warp. It is not a semantic
producer/consumer partition: all eight warps are producers and consumers for
every tile. It merely exposes two independent instruction streams to SM120's
warp schedulers. Correctness does not depend on a particular warp issue order.

Do **not** use dedicated producer warps in EXP-10. Removing any warp from MMA
would either leave its 32x64 control output ownership uncovered or require a
new ownership/replay design. Do **not** revive EXP-09 direct fragments,
EXP-07 all-row ownership, or warp-private B.

## 2. Fixed geometry and shared-memory choice

The control constants remain:

```text
NT=256, NW=8, WN=2, WM=4
BM=128, BN=128, K_TILE=16
MT=BM/16/WM=2
NTT=BN/8/WN=8
NTJ=BN/16=8
DPT=(16*BN)/NT=8 decoded half values/thread/tile
```

### Exact shared-memory arithmetic

The existing dynamic allocation is:

| region | expression | bytes |
|---|---:|---:|
| padded payload pairs `s_pay` | `8 * 24 * sizeof(uint2)` | 1,536 |
| banked A `s_u[2][128][16]` | `2 * 128 * 16 * sizeof(half)` | 8,192 |
| one B tile `s_w[128][16]` | `128 * 16 * sizeof(half)` | 4,096 |
| **control dynamic total** | | **13,824 B = 13.5 KiB** |

The alternatives are:

| logical stage | B slots | B bytes | total dynamic/CTA | two CTAs dynamic | with measured 1,024-B static allocation, two CTAs |
|---|---:|---:|---:|---:|---:|
| **BK32** | 2 x K16 | 8,192 | **17,920 B = 17.5 KiB** | 35,840 B | **37,888 B = 37.0 KiB** |
| BK64 | 4 x K16 | 16,384 | **26,112 B = 25.5 KiB** | 52,224 B | **54,272 B = 53.0 KiB** |

SM120's supplied budget is 228 KiB/SM = 233,472 bytes. Either allocation is
far below the 116,736-byte/CTA two-resident-CTA share, and both are below the
ordinary 48-KiB dynamic-launch range. Shared memory therefore does not displace
the target two-CTA residency for either design. Registers remain the binding
resource: 128 regs x 256 threads x 2 CTAs = 65,536 registers exactly.

**Choose BK32.** Two B slots are exactly the live dependency: one is consumed
for tile `N`, the other is filled for `N+1`. A four-slot BK64 ring cannot remove
either full-CTA publication/reuse barrier because all 256 threads must finish
each cooperative decode and the single payload stage must be republished for
each tile. If implemented with only current/next state it has identical
scheduling and registers but wastes 8,192 bytes/CTA. If made genuinely four
deep, it needs at least two more per-thread payload/phase lookaheads plus more
addresses and buffer-ready state (forecast +3 to +7 registers), taking the
FP16 upper forecast beyond 104 and FP32 beyond its exact 128-register limit.
BK64 therefore has no admissible mechanism advantage in this experiment.

Layout in the candidate is:

```cpp
uint2 * s_pay = (uint2 *) s_raw;                         // [NTJ][ESCHA_MAX_W]
half  * s_u   = (half *)(s_pay + NTJ*ESCHA_MAX_W);       // [2][BM][16]
half  * s_w   = s_u + 2*BM*ESCHA_TILE;                   // [2][BN][16]

const int phase = (ti - lo) & 1;
half * sw_cur = s_w + phase*(BN*ESCHA_TILE);
half * sw_nxt = s_w + (phase ^ 1)*(BN*ESCHA_TILE);
```

Use relative `(ti-lo)&1`, not absolute `ti&1`, so every split-K slice starts in
slot zero and the prologue/epilogue are identical for uneven `lo/hi` ranges.
Retain at least 16-byte alignment for every `ldmatrix` base and add compile-time
size/alignment assertions. The host launch expression must be exactly 17,920 B
for BM=BN=128.

## 3. Exact cooperative decode and fragment contract

The decode assignment is copied from control, not re-derived into a warp-local
form:

```text
NWD  = 8*K
NB   = 32*NWD
dr   = tid % 16
dccl = tid / 16

sp = ((32-K) - K*(escha_dep_pi(dr) + 32*dccl + 4*(dccl >> 3))) % NB
if sp < 0: sp += NB
dg0 = sp >> 5
dw0 = dg0 ? NWD-dg0 : 0
dw1 = dw0 ? dw0-1 : NWD-1
dsh = sp & 31

for k = 0..DPT-1:
    pay = s_pay + k*ESCHA_MAX_W
    c   = dccl + 16*k
    idx = __funnelshift_r(pay[dw0].y, pay[dw0].x, dsh) & 0xffff
    sw_nxt[c*16 + dr] = escha_codebook_h(idx)
```

Payload publication is also literal:

```text
s_pay[pt][pw].y              = ppre
s_pay[pt][(pw+1)%NWD].x      = ppre
```

Thus `pay[dw0].y` is word `dw0` and `pay[dw0].x` is word `dw1`; the funnel
operands must not be reversed. `escha_dep_pi`, the negative-modulo correction,
`dw0/dw1`, `dsh`, 16-bit mask, and `escha_codebook_h` stay bit-for-bit the
same. `lut` and `dep` remain unused compatibility arguments just as in control.

Coverage proof per B tile:

- each thread stores `DPT=8` values;
- 256 x 8 = 2,048 stores = 16 x 128, exactly the B tile;
- `dr=tid%16` selects one K row and `dccl=tid/16` one column inside each of the
  eight 16-column payload tiles;
- `k=0..7` selects each output tile once;
- no expression uses `wm`, `wn`, or row ownership, so no row warp repeats a B
  decode performed by another warp.

### B fragment consumption

Keep the control `tile_b=tile<8,8,half2>` and exact ldmatrix address:

```cpp
const half2 * sw2 = (const half2 *) sw_cur;
for (int j = 0; j < NTT; ++j) {
    tile_b B;
    ggml_cuda_mma::load_ldmatrix(
        B, sw2 + (size_t)(wn*(8*NTT) + j*8)*8, 8);
    for (int i = 0; i < MT; ++i) {
        mma(acc_or_acc16[i][j], A[i], B);
    }
}
```

This is the same eight B fragments, the same `j=0..7` order, and the same
two MMAs per B fragment as control. Changing loop nesting from control's
`i`-outer to `j`-outer does not reorder contributions to any accumulator:
each `[i][j]` receives exactly one MMA for each successive K tile
`ti=lo..hi-1`. It lets one two-register B fragment die before `j+1`, avoiding
an eight-fragment register array. A captured fragment oracle must prove every
`B.x[0..1]` word matches control before any model timing.

## 4. Producer/consumer phase and exact barriers

No named barrier is required or allowed in the first implementation. Every B
tile requires contributions from all eight warps, so a named barrier with a
subset count cannot publish a complete tile. Two ordinary converged
`__syncthreads()` phases are sufficient and make correctness independent of
warp scheduling.

Let `T=hi-lo`. The barriers are:

1. **P(t), publication/free barrier.** Before it, threads publish payload `t`,
   drain the committed async A copy whose visibility is needed, and (in steady
   state) load the current tile's A fragments from the A slot that will be
   recycled. P(t) orders payload publication and A availability to every warp,
   and proves every current-A read finished before that A slot is reused.
2. **D(t), decoded-B/phase-complete barrier.** After cooperative decode of B
   tile `t` (and, in steady state, MMA of tile `t-1`), D(t) publishes the full B
   tile, proves all payload readers are finished before `s_pay` is overwritten,
   and proves all current MMAs/B reads are finished before the old B slot is
   recycled.

The schedule is:

```text
if T == 0: return                         // defensive; host normally makes T>=1

issue+commit A[0]
publish payload[0]
wait A[0]
P(0)
issue+commit A[1] if T > 1
cooperative decode B[0] by all 256 threads
D(0)

for n = 0 .. T-2:
    publish payload[n+1] from one-word ppre
    load A[n] fragments from s_u[n&1] into A[MT]
    wait committed A[n+1]
    P(n+1)                              // pay[n+1]/A[n+1] visible; A[n] free
    issue+commit A[n+2] into s_u[n&1] if n+2 < T
    prefetch immutable code word for payload[n+2] if n+2 < T

    even warps: decode B[n+1] into s_w[(n+1)&1], then MMA(A[n],B[n])
    odd  warps: MMA(A[n],B[n]), then decode B[n+1] into s_w[(n+1)&1]
    D(n+1)                              // B[n+1] ready; phase complete

load A[T-1] fragments
stream-load B[T-1] fragments and perform MMA(T-1)
store accumulators                       // no final shared reuse barrier needed
```

`A[n]` is loaded before P(n+1), so issuing `A[n+2]` to the recycled A slot
after P(n+1) cannot race an `ldmatrix`. B uses a different slot for `n+1`, so
the even warps' early writes cannot race the odd warps' reads of B[n]. D(n+1)
prevents the old B slot from being selected for B[n+2] until every warp has
finished MMA(n). The last tile has no next decode and is consumed in the
epilogue.

There are exactly `2*T` source CTA barriers for a slice: P(0)+D(0), then one
P and one D for each of the remaining `T-1` tiles. That is **two barriers per
16-wide K tile**, including prologue/epilogue, versus control's `3*T`. There is
no barrier inside the even/odd phase and no dedicated-producer rendezvous.
Static SASS BAR occurrences may be duplicated by the FP16/FP32 compiler paths;
the measured dynamic/source ratio and matched-symbol comparison must be
reported.

### Payload/code-stream hazard proof

- D(n) completes after all threads have decoded B[n], so no thread can still
  read payload n when payload n+1 is written.
- P(n+1) completes after all payload n+1 words and both halves of every ring
  pair are published. Decode B[n+1] begins only after P(n+1), before either
  parity group begins its source-ordered operation sequence. Therefore every
  decode of N+1 uses data published before the mixed decode/MMA phase for N
  starts.
- `code` is read-only. `ppre` remains one 32-bit word per eligible thread; a
  word for n+2 may be globally prefetched during phase n but is not installed
  in `s_pay` until D(n+1) has released the previous payload.
- `has_pay`, `pt`, and `pw` remain control-identical. No second payload array
  is necessary for BK32.

## 5. A-stage interaction and slice edges

The banked SM120 cp.async A path from EXP-01 must remain enabled and must use
the existing zero-fill predicate:

```text
cp_m = tid / 2
cp_h = (tid % 2)*8
row  = row0 + cp_m
src_row = row < n_rows ? row : 0
copy 16 bytes to s_u[phase][cp_m][cp_h], src-size 0 for a tail row
```

Both rings use relative slice indices:

```text
A current = s_u[(ti-lo)&1]
A future  = s_u[((ti-lo)&1)^1]
B current = s_w[(ti-lo)&1]
B future  = s_w[((ti-lo)&1)^1]
```

Edge rules are exact:

- `lo=(nit*sl)/n_slices`, `hi=(nit*(sl+1))/n_slices`; never round or pad a
  slice and never move a K tile across its `[lo,hi)` boundary.
- For `T=1`, issue only A0, run P0/decode-B0/D0, then MMA0; no `+1` A/B/code
  access occurs.
- For `T>=2`, A1 is committed immediately after P0 so B0 decode hides part of
  its latency. In steady state, issue A[n+2] only when `n+2<T`.
- Publish/decode B[n+1] only when `n+1<T`; prefetch `ppre(n+2)` only when
  `n+2<T`.
- The synchronous A fallback retains the same two A slots and writes the next
  slot only after P proves the old fragment loads complete. It must be correct,
  but the SM120 candidate performance gate uses the promoted async path.

No change may serialize the existing A copy behind B decode. SASS must retain
`LDGSTS`/async-copy instructions and the implementation review must show the
A[n+2] issue located before the mixed decode/MMA phase that is intended to hide
it.

## 6. Accumulator, partial, and finalize seams

Copy the accumulator declarations, initialization, MMA function, and store
code literally from the promoted control, except for the local streamed-B loop
shown above.

- FP16: `tile_ah acc16[MT][NTT]`, with `tile_ah=tile<16,4,half2>`.
  Preserve the four logical `tile_c` coordinates exactly:
  `x[0].x -> l0`, `x[0].y -> l1`, `x[1].x -> l2`, `x[1].y -> l3`.
  Keep separate `row_a<n_rows` and `row_b<n_rows` guards.
- FP32: `tile_c acc[MT][NTT]`, loop `l=0..tile_c::ne-1`, and retain the
  per-element `row<n_rows` guard.
- Both modes write exactly
  `partial[((int64_t)sl*n_rows + row)*OC + oc0 + n]`.
- `n_slices>1` is mandatory. Every slice writes only its disjoint
  `[sl][row][col]` region, K tiles remain ascending within the slice, and
  `escha_finalize_dense` continues summing `s=0..n_slices-1` in fixed order
  before the Hadamard/rout operation.
- Do not fuse finalize, alter fp16/fp32 selection (`IC<=6144`), coalesce the
  fp16 stores, change row guards, or special-case canonical dimensions.

FP32 and FP16 results are expected bitwise identical to control on the same
compiler/device because each accumulator sees the same fragments in the same
K-tile order. Approximate agreement is not an acceptance criterion.

## 7. Register forecast and residency gate

The important live-range change is deliberate: current A fragments cross
P(n+1) and remain live while even warps decode the next B tile. Streaming one B
fragment avoids carrying control's full 16-register B array through that
phase. Counts below are 32-bit registers or equivalent logical values; address
and compiler-control rows are forecasts because ptxas coalesces them. The
measured total is decisive.

| live category/thread | control FP16 | BK32 FP16 forecast | control FP32 | BK32 FP32 forecast | constraint |
|---|---:|---:|---:|---:|---|
| accumulator | 32 | 32 | 64 | 64 | unchanged `[2][8]` |
| A fragments | 8 | 8 | 8 | 8 | two `tile_a`, live across P/decode on even warps |
| B fragments | 16 peak | **2 peak** | 16 peak | **2 peak** | one streamed `tile_b`; never `B[8]` |
| invariant decode state (`sp,dw0,dw1,dsh,dr,dccl`) | 6-8 | 6-8 | 6-8 | 6-8 | control formula, once/thread |
| transient funnel/codebook/load/store values | 4-6, not concurrent with B | 4-6, concurrent with A only | 4-6 | 4-6 | scope inside one decode iteration |
| payload lookahead | 1 | 1 | 1 | 1 | one `ppre`, not a BK64 queue |
| 64-bit pointers/addresses + indices/predicates | compiler allocated | control-class + 2-5 phase values | compiler allocated | control-class + 2-5 | aggressively scope aliases |
| **ptxas total** | **97 measured** | **forecast 97-104** | **128 measured** | **forecast <=128** | compile gate below |

The 14-register reduction from `B[8]` to one `B` is the headroom for the
longer A/decode overlap and phase variables. The implementation must not keep
both `sw_cur/sw_nxt` as long-lived 64-bit aliases when a phase offset can be
recomputed cheaply, unroll multiple decode iterations into simultaneously live
temporaries, or retain more than one code word.

Hard gate for all K2/K3 instantiations:

- FP16 **<=104 registers**, with <=97 preferred and any 98-104 result requiring
  Terra review against occupancy and SASS;
- FP32 **<=128 registers** (129 is an automatic failure);
- `STACK=0`, `LOCAL=0`, no spill loads/stores and no `LDL`/`STL` spill path;
- measured launch resources permit at least two 256-thread CTAs/SM.

Do not force the result with `--maxrregcount`, a changed accumulator, fewer
MMAs, or spills. Retain the control's `__launch_bounds__(256,1)` during the
first compile so the measurement reflects the schedule rather than a compiler
cap.

## 8. Guarded implementation and rollback

1. Add `escha_matmul_dense_tiled_mma_coop_bk32` and only its small helper(s)
   under `#ifdef ESCHA_MMA_COOP_BK32_EXPERIMENT` in
   `ggml/src/ggml-cuda/escha-moe.cu`. Do not edit the promoted kernel body.
2. Keep template support for K2/K3 and FP16/FP32 so the candidate covers the
   complete ordinary Escha MMA-prefill route. Gate host selection on the
   existing `use_mma`, qualified SM120, K in {2,3}, and `OC%128==0`. Generation,
   ragged, cuBLAS, WMMA, non-SM120 CUDA, HIP, Vulkan, and standard-Qwen paths
   remain unchanged.
3. Allocate exactly 17,920 bytes dynamic shared memory and launch the same
   `dim3(n_tb_mma,n_cb_mma,n_slices)` grid and `dim3(32,8)` block.
4. Retain `IC<=6144` mixed accumulation. Candidate route tags are
   `mma-fp16-bk32` and `mma-fp32-bk32`, assigned only when the candidate is
   actually launched. Control tags remain `mma-fp16-mixedacc` and
   `mma-fp32-mixedacc`.
5. Preserve the existing partial allocation and unconditional separate
   `escha_finalize_dense` launch. No `n_slices==1` restriction is permitted.
6. Under a separate experiment-diagnostics guard, add a standalone fragment
   capture/check that compares the candidate's two B slots after publication
   and their eight ldmatrix fragments against the existing control staging for
   K2 and K3. Diagnostic code must not be called or linked into the timed hot
   kernel path.

Rollback is mechanical: remove the `ESCHA_MMA_COOP_BK32_EXPERIMENT` kernel and
helpers, the guarded host launch/smem/tag block, and experiment-only diagnostic
files, then remove only `build-cuda-exp10-bk32`. With the macro absent, the
preprocessed kernel and dispatch must be byte-identical to the promoted
control. Do not touch rotate/finalize, metadata gating, model files, benchmark
inputs, or the promoted kernel while reverting.

## 9. Definition of Done and frozen verification gates

Run from the repository root. Record git SHA, dirty diff, compiler/CUDA/driver,
GPU, configure/build exit codes, binary/model SHA-256, full launch config, and
raw JSON/stderr. Any build/resource, fragment, CUDA, route, parity, or numeric
failure stops before timing.

### 9.1 Fresh matched builds

```bash
cd '/mnt/d/CODEX WORKSPACE/beellama-escha'
cmake -S . -B build-cuda-exp10-control -G Ninja \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
cmake --build build-cuda-exp10-control -j 12 \
  --target ggml-cuda llama-bench llama-server

cmake -S . -B build-cuda-exp10-bk32 -G Ninja \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_FLAGS=-DESCHA_MMA_COOP_BK32_EXPERIMENT=1
cmake --build build-cuda-exp10-bk32 -j 12 \
  --target ggml-cuda llama-bench llama-server
```

The two builds must come from the same source/diff and differ only by the
candidate macro. Never reuse an older experiment binary.

### 9.2 cuobjdump resource and SASS gate

```bash
OUT='evidence/EXP-10-coop-bk32/2026-09-02'
cuobjdump --dump-resource-usage build-cuda-exp10-control/bin/libggml-cuda.so \
  > "$OUT/control-resources.txt"
cuobjdump --dump-resource-usage build-cuda-exp10-bk32/bin/libggml-cuda.so \
  > "$OUT/candidate-resources.txt"
cuobjdump --dump-sass build-cuda-exp10-control/bin/libggml-cuda.so \
  > "$OUT/control-sass.txt"
cuobjdump --dump-sass build-cuda-exp10-bk32/bin/libggml-cuda.so \
  > "$OUT/candidate-sass.txt"
```

Extract matched K2/K3 FP16/FP32 symbols. PASS requires the register/spill limits
in section 7, dynamic smem exactly 17,920 B, two source CTA barriers/K tile
versus control's three, decoded-B `STS.U16` still present once per weight (the
broadcast is intentionally retained), B `LDSM` and `HMMA.16816` coverage
control-class, and cp.async/`LDGSTS` retained. Report BAR, HMMA, LDSM, STS.U16,
LDS, LDGSTS, and decode-ALU counts. A reduced HMMA count, missing store/reload,
or fourfold LDS/decode-ALU growth is an EXP-09 regression and rejects.

### 9.3 Fragment oracle

Implementation must provide the experiment-only
`scripts/check-exp10-fragments.py`; it launches/captures K2 and K3 tile 0 and a
phase-1 tile, enumerates all `(slot,j,lane,reg)` values, and exits nonzero on
the first mismatch or incomplete 16x128 coverage.

```bash
MODEL='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf'
OUT='evidence/EXP-10-coop-bk32/2026-09-02'
python3 "$OUT/scripts/check-exp10-fragments.py" \
  --binary "$PWD/build-cuda-exp10-bk32/bin/llama-bench" \
  --model "$MODEL" --out "$OUT/fragment-oracle"
```

PASS is K2 and K3, both B slots, all eight fragments: candidate staged half
bits equal control and candidate `tile_b.x[0..1]` words equal the control
ldmatrix oracle exactly. No approximate tolerance is allowed.

### 9.4 Route proof and direct family gate

```bash
MODEL='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf'
IDS='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids'
OUT='evidence/EXP-10-coop-bk32/2026-09-02'
for ARM in control bk32; do
  B="build-cuda-exp10-$ARM"
  ESCHA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1 "$B/bin/llama-bench" \
    -m "$MODEL" --prompt-tokens-file "$IDS" -p 2048 -n 0 -ngl 99 \
    -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json \
    >"$OUT/$ARM-profile.json" 2>"$OUT/$ARM-profile.stderr"
done
python3 "$OUT/scripts/check-route-and-family.py" \
  --control "$OUT/control-profile.stderr" \
  --candidate "$OUT/bk32-profile.stderr"
```

The helper must fail unless control has 800/800 intended records on mixedacc
and candidate has **672 `mma-fp16-bk32` + 128 `mma-fp32-bk32` = 800**, zero
fallback/predicate mismatch. For K2 5120->17408 at M=2048, aggregate candidate
`matmul_ms` must be at least 10% faster. No other family may regress more than
5%.

### 9.5 Tails, split-K, numeric equality, and sanitizer

Implementation must provide `scripts/run-exp10-op-gates.py`, using identical
packed inputs for matched control/candidate direct-op runs and capturing both
partials and final outputs. Run:

```bash
OUT='evidence/EXP-10-coop-bk32/2026-09-02'
python3 "$OUT/scripts/run-exp10-op-gates.py" \
  --control "$PWD/build-cuda-exp10-control/bin/llama-bench" \
  --candidate "$PWD/build-cuda-exp10-bk32/bin/llama-bench" \
  --rows 17,127,128,129,512,1544,2048,4096 \
  --formats 2,3 --force-slices 1,2,3,4 \
  --out "$OUT/op-gates"

compute-sanitizer --tool memcheck --error-exitcode=99 \
  "$PWD/build-cuda-exp10-bk32/bin/llama-bench" \
  -m "$MODEL" -p 17 -n 0 -ngl 99 -b 17 -ub 17 \
  -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json \
  >"$OUT/sanitizer-p17.json" 2>"$OUT/sanitizer-p17.stderr"
```

PASS requires exact finite partials/final outputs, no sentinel left unwritten,
independent fp16 half-row guards, exact `[0,nit)` coverage once across `lo/hi`,
both `n_slices=1` and `>1`, no OOB/race/CUDA error, and candidate routing for
the smallest prefill tail M=17. Also run the depth family smoke at
M=128,256,512,1024,2048,4096; no family/depth regression may exceed 5%.

### 9.6 P2/P7 and decode

```bash
MODEL='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf'
OUT='evidence/EXP-10-coop-bk32/2026-09-02'
for ARM in control bk32; do
  ESCHA_SERVER_BIN="$PWD/build-cuda-exp10-$ARM/bin/llama-server" \
    python3 scripts/escha-compare/run_compare.py --model "$MODEL" \
      --only P2-factual,P7-tool-call --max-new-tokens 16 --ctx-size 4096 \
      --outdir "$OUT/parity-$ARM"
  GGML_CUDA_DISABLE_GRAPHS=1 "$PWD/build-cuda-exp10-$ARM/bin/llama-bench" \
    -m "$MODEL" -p 0 -n 64 -ngl 99 -b 2048 -ub 2048 \
    -ctk f16 -ctv f16 -fa on -r 5 -o json -oe json \
    >"$OUT/$ARM-decode-r5.json" 2>"$OUT/$ARM-decode-r5.stderr"
done
```

P2 and P7 must each retain 16/16 greedy seed-42 agreement with no hang/CUDA
error. Median decode may regress at most 2%. Generation must never carry a
`bk32` route tag.

### 9.7 Canonical matched 9-pair ABBA campaign, graphs ON

One unrecorded warmup per arm, fresh process per trial, exact order
`AB BA BA AB AB BA BA AB AB`:

```bash
REPO='/mnt/d/CODEX WORKSPACE/beellama-escha'
CTL="$REPO/build-cuda-exp10-control/bin/llama-bench"
CAND="$REPO/build-cuda-exp10-bk32/bin/llama-bench"
MODEL='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf'
IDS='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids'
OUT="$REPO/evidence/EXP-10-coop-bk32/2026-09-02/bench"
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

Report every sample, median/mean/CV, paired ratios, geometric mean and 95% CI,
candidate-faster count, hashes, graphs-on proof, VRAM, and exit codes. The
Decision-B pass requires all earlier gates plus full-2K median throughput at
least 5% faster, at least 4 of the five frozen candidate samples above the
control median, target-family matmul at least 10% faster, no family/depth
regression over 5%, and decode within 2%. The 9-pair analysis is the matched
noise adjudicator; it does not relax those absolute gates. A full gain below
2%, any resource failure, or any correctness failure means REJECT + revert +
stop incremental kernel work. A 2-5% result is not promotable without Sean's
explicit new decision.

## 10. Risks and watch-outs

- **Register/liveness is the first fail-fast risk.** A fragments now cross a
  CTA barrier and, on even warps, the cooperative decode. If ptxas retains a
  full B array or unrolls eight decode iterations into parallel temporaries,
  the candidate can exceed 104/128 despite unchanged arithmetic. Scope one B
  fragment and one decode iteration; never solve this with spilling.
- **Overlap is opportunistic, not guaranteed.** The even/odd warp ordering
  exposes decode and HMMA concurrently, but the scheduler may still serialize
  them or producer/consumer durations may be imbalanced. Correctness must not
  assume concurrent progress. Failure to reach >=10% target-family gain closes
  the idea even if barrier counts improve.
- **Barrier correctness.** Omitting D races payload overwrite/B publication;
  omitting P races payload consumers and recycled A; issuing A[n+2] before P
  races current A ldmatrix. A named subset barrier cannot replace either full
  CTA barrier because all 256 threads contribute B.
- **Shared-memory occupancy.** 17,920 B is safe by arithmetic, but record both
  dynamic and static resource values and actual occupancy. Do not assume the
  supplied per-SM number is the only architectural allocation constraint.
- **cp.async plus decode on the same threads.** Commit future A before the
  mixed phase, wait only at its publication boundary, and preserve zero-fill.
  An early wait or late commit silently destroys EXP-01 overlap.
- **Decode identity.** Do not algebraically replace `sp/dw0/dw1/dsh`, reverse
  funnel operands, index B as `[k][n]`, or derive payload from `wm/wn`.
- **FP16 store seam.** Preserve all four logical lanes from the two
  `tile_ah` half2 registers and both independent row guards. This seam already
  invalidated P-ARCH-20 once.
- **Split-K.** Use relative ring phases, do not assume `lo==0`, do not prefetch
  across `hi`, preserve `[sl][row][col]`, and retain fixed finalize order.
- **Scope.** No filename/model-name/benchmark conditionals, metadata changes,
  standard-Qwen behavior changes, finalizer fusion, tile-aspect change, or
  direct-fragment construction belongs in EXP-10.

## 11. Terra review flags

Terra must confirm these before implementation:

1. **BK32 over BK64:** approve 17,920-B two-slot B staging. Confirm that BK64's
   26,112 B provides no barrier/overlap reduction under all-256-thread decode
   and that a useful four-deep state would violate the register envelope.
2. **Two full-CTA barriers/tile:** approve the exact P/D phase proof and
   `2*T` count versus control `3*T`; confirm no named barrier is needed.
3. **Symmetric warp phasing:** approve even
   `decode-next -> MMA-current`, odd `MMA-current -> decode-next`, with all
   eight warps still decoding and consuming every tile. Confirm dedicated
   producer warps are out of scope.
4. **Streamed control ldmatrix B:** approve one `tile_b` at a time and `j`-outer
   MMA traversal as fragment/order preserving, not a direct-fragment rewrite.
5. **A-ring seam:** confirm A[n] is loaded before P(n+1), A[n+2] is issued only
   after P(n+1), and relative `(ti-lo)&1` phases cover T=1 and uneven slices.
6. **Register forecast:** confirm 97-104 FP16 / <=128 FP32 is the only admissible
   range, with measured ptxas resources decisive and no forced cap/spill.
7. **Payload timing:** confirm the single `s_pay` plus D-before-overwrite and
   P-before-decode proof satisfies the requirement that B[n+1] reads only
   payload published before the mixed MMA(n) phase begins.

