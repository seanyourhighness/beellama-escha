# P-ARCH-01 — K2/K3 packing and fragment-order evidence

**Status:** `RESOLVED`  
**Ledger row:** `Escha weight packing`  
**Question:** Did this make the data presented to the kernel match the control?  
**Direct answer:** `YES` — both production tiles are byte-identical through the
checkpoint/GGUF/Bee payload boundary, decode to the same FP16 bits as Escha's
runtime, and reach Bee's HMMA B fragment ABI with no additional permutation.

Raw captures are outside the repository at
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-01/2026-08-29/`.

## Immutable run identity

| Field | Value |
| --- | --- |
| Date / operator | `2026-08-29 / Codex` |
| BeeLlama source revision | `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d` plus preserved local changes |
| Relevant source hashes | `escha-moe.cu c8b609b6...`, `mma.cuh 1b136f69...`, `analyze_parch01.py 7fa4d601...`, `convert_escha_to_gguf.py 00a7a8e4...`; full values in `source-files.sha256` |
| Hybrid GGUF | `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf`; SHA-256 `e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d` |
| Escha runtime reference | wheel `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/runtime/sglang/escha-1.2.0+qwen3dense-cp312-cp312-manylinux_2_28_x86_64.whl`; SHA-256 `735f4b7ae5c03ac11c754b2a6a4e44326a1d175139f4e87fd82b2e0f4e27c22e`; loaded `_C.so` SHA-256 `2c55cf7c087aa3937f79ba9d8ff64b8b7fe6fa1fc9240f2026ce78e7b88763ab` |
| Reference Python dispatch | `runtime/wheel-src/sglang_srt_layers_quantization_escha.py:1038` (`escham_code_gemm`); logical oracle `runtime/wheel-src/escha_transform.py:34` (`escham_reconstruct`) |
| GPU / CUDA / driver | NVIDIA GeForce RTX 5090, compute capability 12.0; CUDA 13.0 build `36424714_0`; driver `610.88` |
| Fixed activation | K2 and K3 captures are byte-identical, shape `512x5120` F32, SHA-256 `be1ef94803faa49728910569174ccf0451c71a005aafccafd08b3a28140a310b` |
| Machine-readable result | `summary.json`; SHA-256 `f6a5ddbc4fee99ef1a843cef76b192b04a090e8e40a9756d7d472ed1e24746f4` |

## Selected production tiles

| Format | Tensor name | Layer | Tile coordinate `(k_tile, out_tile)` | Production geometry | Payload dump / SHA-256 | `dep` dump / SHA-256 | `rin/rout` dumps / SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K2 | `blk.0.ffn_gate.escha_code` / checkpoint `model.language_model.layers.0.mlp.gate_proj.escha_code` | 0 | `(0,0)` | `IC=5120`, `OC=17408`, tile `16x16`, payload `16*K=32` I16 | `k2_tile_0_0_payload.i16.bin` / `34786170d16bcb52235a027f24597b5b100cde88e0ba710869d7d9ca0970108c` | `k2_dep.i16.bin` / `df422acf391810e8a97b5d3ec217d666c309c60bd448712de446207931ed9c0f` | `k2_rin.f16.bin` / `6c682039ee087342c5bb84ae8eefec78ab7342c683ec8a954be5e8783e0b9c5d`; `k2_rout.f16.bin` / `319c758c9a1fcb896dd7a08dc47b87ff7bed453f0c25d5b38400d990fe364f76` |
| K3 | `blk.0.ffn_up.escha_code` / checkpoint `model.language_model.layers.0.mlp.up_proj.escha_code` | 0 | `(0,0)` | `IC=5120`, `OC=17408`, tile `16x16`, payload `16*K=48` I16 | `k3_tile_0_0_payload.i16.bin` / `2bdc2ce5c71bd033bad754aca4cbb76eec9b80a216ca9d2e72caa98dfd510993` | `k3_dep.i16.bin` / `2afc0909281d9320382a4f45cb0ade7e1377904274ddb0139ab3a787a828dee1` | `k3_rin.f16.bin` / `45d6e5cf8b9a93151ecf9f3299d6c89a731bf7e15eafc7cf0da3c90eb0beba9c`; `k3_rout.f16.bin` / `45e0515454156c3a8cd10953305885da0542328dc7920e7238f344cb2f5cca5c` |

The captured `rin/rout` values also equal the converter's folded reference values
`fp16(rin*s_in)` and `fp16(rout*s_out)` bit-for-bit for both projections.

## Explicit mapping

The converter writes checkpoint `(IC/16, OC/16, 16*K)` bytes directly as GGUF
`(16*K, OC/16, IC/16)` (`convert_escha_to_gguf.py:116-137`). Tile `(0,0)`
therefore begins at payload word zero in every representation.

For logical weight `W[k,out]`, with `NW=8*K`, `NB=32*NW`:

```text
pi(k) = (k&1) | (((k>>3)&1)<<1) | (((k>>1)&3)<<3)
t     = pi(k) + 32*out + 4*(out>>3)
sp    = ((32-K) - K*t) mod NB
g0    = sp >> 5
w0    = g0 ? NW-g0 : 0
w1    = w0 ? w0-1 : NW-1
idx   = funnelshift_r(payload[w0], payload[w1], sp&31) & 0xffff
W     = escha_codebook_h(idx)
```

The captured `dep` rows are direct `k-major,out-minor`: row `16*k+out` lists
the same 16 physical payload bit indices, LSB to MSB. There is no hidden `dep`
row permutation in either selected production table.

Bee's safe FMA staging index is `k*128+out` for tile-local `out`. Its candidate
tensor-core B staging is column-major `[out][k]`, index `out*16+k`. The actual
`tile<8,8,half2>` / `load_ldmatrix` implementation (`mma.cuh:276,786`) maps to
the m16n8k16 B fragment as:

```text
fragment_tile = out / 8
lane          = 4*(out % 8) + ((k >> 1) & 3)
register      = k >> 3
half          = k & 1
```

Each per-format CSV contains all 256 mappings, including `pi`, `t`, source
words/bytes/bits, both Bee staging indices, and fragment lane/register/half.

## Mapping evidence and equality

| Format | Packed payload order | Logical equality | Bee staging / Escha fragment order | Evidence |
| --- | --- | --- | --- | --- |
| K2 | 64 captured bytes equal checkpoint tile and GGUF tile byte-for-byte | Independent closed-form FP16 bits equal `escham_reconstruct` for all 256 weights; common SHA-256 `91ed3f01f412da5c3e6b6aa7e0435ed5113d67622c3a8180abc7b7753833decc` | Captured Bee B-fragment bytes equal reference logical weights under the HMMA B ABI; both SHA-256 `681ae999d7ffda4af24169676c0ac9edb10f6ecf3db4077a0e427730873bda03` | `k2_packed_logical_fragment_mapping.csv`, SHA-256 `feedd61625108fddbc739076c4fa9718c0ea48407b3bc233525badea55ea06bf` |
| K3 | 96 captured bytes equal checkpoint tile and GGUF tile byte-for-byte | Independent closed-form FP16 bits equal `escham_reconstruct` for all 256 weights; common SHA-256 `08f8411e130b806a68dc901250ef36b21730cf4aea1da1ca1ecf83b76a16c031` | Captured Bee B-fragment bytes equal reference logical weights under the HMMA B ABI; both SHA-256 `7c19c02fa17412b0bc8a42cf3056b5b2568601d07d832b6e167b476707e92910` | `k3_packed_logical_fragment_mapping.csv`, SHA-256 `39ee1cf8f42b11dd20a0e782dd944902eaa1f7b4066c1882a0385d4c5f788f03` |

The reference fused prefill op was invoked once per format with the fixed
512-row activation. It selected:

- K2: `escham_code_gemm_kernel<1,2,128,64,2,true,true>`; output FP16-bit
  SHA-256 `13431950d21470435a230333785f7a8e4ec4417bfdf699391d378e630ce0cacb`.
- K3: `escham_code_gemm_kernel<1,3,128,64,2,true,true>`; output FP16-bit
  SHA-256 `e9c7ac975e4f2bd7ee0f70c6f29d2d0b1c0c8d320f368706e51377b2dafd9a4e`.

The exact SM120 symbols contain `HMMA.16816.F16`; decisive PCs and symbols are
in `reference-sm120-hmma-evidence.txt`, SHA-256
`13f5a11dbf4c3507e964a034bdd153625f6f905a827314ecac698456bb968ed8`.
This runtime invocation is selection evidence, not a benchmark or an output
parity claim across activation/rotation semantics.

## Decision

- [x] Identical logical and fragment mapping: mark `Escha weight packing` as
  `RESOLVED`; proceed only to the activation/rotation row.
- [ ] Logical weights equal but fragment order differs: `MISMATCH`.
- [ ] Logical weights differ: `MISMATCH`.

No inverse-permutation adapter is required. No dispatch, MMA/WMMA execution,
epilogue, or tile-size change was made. The production Bee prefill route remains
`tiled-fma-fp32`; the fragment kernel was capture-only and did not execute MMA.

## Ledger update

| Ledger file / GBrain page | Row changed | Status | Impact | Artifact location |
| --- | --- | --- | --- | --- |
| `docs/escha-architecture-diff-ledger.md` / `projects/beellama-escha-architecture-diff-ledger` | `Escha weight packing` | `RESOLVED` | `critical` | this manifest plus the external evidence directory above |
