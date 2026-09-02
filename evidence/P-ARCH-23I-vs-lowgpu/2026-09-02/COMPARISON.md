# P-ARCH-23I vs LowGPU IQ3_XXS

Date: 2026-09-02

## Bottom line

**P-ARCH-23I** is a derived BeeLlama deployment artifact built from the EschaLabs Qwen3.8-27B-Escha-W2 checkpoint (project manifest revision `f0eadefa2f9679f7c04a115214c1cd883979a529`). It is not the original packed Escha body. It is a deliberately hybrid reconstruction: all 192 FFN weights and all 64 full-attention projection weights are copied raw from the compatible LowGPU GGUF; the 48 GDN gates and 96 linear-attention QKV/SSM projections are reconstructed from the Escha checkpoint and quantized to standard Q2_K; the embedding is standard Q4_K; and only the LM head remains in Bee's packed LowGPU representation. It retains Escha metadata because its recurrent-state serialization/semantics still follow the Escha checkpoint.

**LowGPU IQ3_XXS** is the standard, NoMTP qwen35 quant used as the reference. Project records identify it as TheWegemann's quant of the compatible Qwen3.8-27B model. Its transformer and vocabulary weights are ordinary GGML tensors, with a layer-wise mixture of IQ/K quants selected by an importance matrix; it has no Escha projection sidecars and no Bee LowGPU packed-vocabulary sidecars. Its GGUF records `general.file_type=24` and importance-matrix provenance (200 chunks, 496 entries, `daphne-calibration-v1.txt`, `imatrix-generic-200.gguf`).

## Shared architecture and lineage

The two GGUF metadata blocks agree on the model geometry:

| Field | Both artifacts |
| --- | ---: |
| `general.architecture` | `qwen35` |
| Base/model family | Qwen3.8-27B |
| Blocks | 64 |
| Full-attention cadence | every fourth block (16 full-attention, 48 recurrent/GDN blocks) |
| Attention heads / KV heads | 24 / 4 |
| Key / value length | 256 / 256 |
| Embedding width | 5120 |
| FFN width | 17408 |
| Context length | 262144 |
| SSM inner / state / groups / time-step rank / conv kernel | 6144 / 128 / 16 / 48 / 4 |

P-ARCH-23I explicitly embeds `general.base_model.0.name=Qwen3.8 27B` and the Qwen repository URL. LowGPU lacks those base-model fields in the inspected metadata, but its qwen35 geometry is identical. Neither inspected metadata block has a NextN/MTP-layer key.

## Major differences

1. **Provenance and meaning of the weights.** P-ARCH-23I begins with the Escha-W2 checkpoint at revision `f0eadefa...`, then substitutes or reconstructs selected families into GGML-native matrices. It therefore combines weights from two lineages: donor LowGPU bytes for FFN/full attention, and Escha-derived reconstructions for GDN gate/linear attention. LowGPU is a straight standard quant of its own compatible Qwen3.8-27B weights, with recorded importance-matrix calibration. P-ARCH-23I's `general.name` is `Escha W2 x LowGPU hybrid (Beellama port)`; LowGPU's embedded `general.name` is `SafetensorKindergarten`.

2. **Marker keys change semantics, not just labeling.** Only P-ARCH-23I has `qwen35.escha.version=1` and `qwen35.lowgpu.version=1`. LowGPU has neither. In BeeLlama, `escha.version=1` enables the Escha serialization contract: it converts stored raw `A_log` to `-exp(A_log)` at graph construction and uses the checkpoint's grouped/repeat-interleaved GDN Q/K head order. A standard GGUF without the marker uses the already-converted SSM decay and normal pre-tiled/cyclic head layout. The marker also permits per-projection Escha-code loading, but only when a corresponding `*.escha_code` tensor exists. `lowgpu.version=1` enables per-side packed vocabulary loading; P-ARCH-23I uses that facility for its output head only.

3. **Tensor inventory and sidecars.** P-ARCH-23I has 856 tensors; LowGPU has 851. P-ARCH-23I's six codec/LowGPU sidecars are `escha_lut`, `escha_dep_k2`, `escha_dep_k3`, `output.lowgpu_codes`, `output.lowgpu_scales`, and `output.lowgpu_zps`. LowGPU has zero such tensors. The net count difference is five: the three Escha tables add three, while replacing one `output.weight` with three packed-output tensors adds two. Crucially, P-ARCH-23I has no layer-local `*.escha_code`, `*.escha_rin`, or `*.escha_rout` projection tensors; the shared Escha tables are present but are not a hot body representation.

4. **FFN family is identical by construction, not merely similar in type mix.** Both files contain 192 standard `blk.*.ffn_{gate,up,down}.weight` tensors, with matching names, shapes, per-tensor types, and a total 5,587,271,680 payload bytes. P-ARCH-21A/P-ARCH-23I did not re-quantize the Escha FFNs: `write_standard_ffn()` copies each donor tensor's raw data, raw dtype, and endianness from this exact LowGPU GGUF. Thus P-ARCH-23I's FFN payload bytes are the LowGPU donor's importance-matrix-quantized bytes. They are not reconstructed Escha-W2 FFN weights. The historical correlation check was about 0.835, sufficient for the validated hybrid, but not identity to the Escha checkpoint.

5. **The 48 GDN gate tensors have the same role and shape but different values and quant policy.** P-ARCH-23I stores every `blk.*.attn_gate.weight` as Q2_K with shape `(5120, 6144)` and 10,321,920 bytes per tensor. Each is reconstructed from that Escha layer's checkpoint `linear_attn.in_proj_z` code/rin/rout sidecars, then quantized. LowGPU stores its own 48 gate matrices as a mixed standard quant set: 23 Q2_K, 9 IQ2_S, 4 IQ2_XXS, 3 IQ2_XS, 2 IQ3_XXS, 5 IQ3_S, 1 IQ1_S, and 1 IQ1_M. These weights are not interchangeable. P-ARCH-21B copied LowGPU's apparent `attn_gate` donor and generated garbage: its correlation to the required Escha `in_proj_z` was only about 0.04, versus about 0.835 for the valid FFN donor relationship. Reconstructing the true gate restored coherent output in P-ARCH-21C/P-ARCH-23.

6. **Attention projections split into copied full attention and reconstructed linear attention.** On blocks 3, 7, ..., 63, P-ARCH-23I raw-copies all 64 Q/K/V/O tensors from LowGPU, preserving the donor's exact mixed IQ/K quant types. On the other 48 blocks, P-ARCH-23I reconstructs 48 `attn_qkv.weight` matrices `(5120, 10240)` and 48 `ssm_out.weight` matrices `(6144, 5120)` from Escha code/rin/rout, storing all 96 as Q2_K. LowGPU instead has its own 96 matrices in a mixed set of IQ1/IQ2/IQ3/IQ4/Q2_K types. Raw donor substitution was rejected for this family: measured correlations were about 0.40 for linear QKV and 0.04 for SSM output. In both final files these attention tensors are ordinary GGML weight tensors and execute through the normal matmul path; none remains Escha-packed.

7. **Embedding and LM-head representations are almost opposite.** P-ARCH-23I has a standard Q4_K `token_embd.weight` (715,161,600 bytes), used by normal `GET_ROWS`, but no `output.weight`; its output is three packed LowGPU sidecars: I8 codes (476,774,400 bytes), F16 scales (19,865,600 bytes), and I8 zero points (9,932,800 bytes), used by `LOWGPU_MUL_MAT`. LowGPU has a standard IQ4_XS embedding (675,430,400 bytes) and a standard Q4_K `output.weight` (715,161,600 bytes), with no packed-vocabulary operator. P-ARCH-23I spends about 39.7 MB more on embedding but about 208.6 MB less on its head, a net vocabulary saving of 168,857,600 bytes.

8. **P-ARCH-23I is smaller despite being hybrid.** The exact files are 9,345,100,992 B versus 9,570,663,040 B: P-ARCH-23I is 225,562,048 B (225.6 decimal MB, 2.36%) smaller. Extra marker metadata and the three small Escha tables are negligible. The packed three-bit output head produces most of the saving, and the uniform Q2_K reconstructed GDN/linear-attention projections contribute to the remaining net saving; these more than offset P-ARCH-23I's larger Q4_K embedding and other representation differences. “Hybrid” describes provenance and dispatch, not a larger or duplicated body.

9. **Runtime dispatch is materially different even though both bodies use standard GGML.** LowGPU has no codec markers or sidecars, so all body projections and both vocabulary sides use normal GGML `MUL_MAT`/`GET_ROWS`. P-ARCH-23I's qwen35 loader sees `escha.version=1`, loads the three shared tables, then falls back per tensor to standard weights because no body `*.escha_code` exists. Its FFN, full-attention, reconstructed GDN gate, reconstructed linear QKV, and reconstructed SSM output therefore use standard GGML matmuls too. Only its LM head uses Bee's LowGPU custom packed matmul. The CUDA Escha decoder in `escha-moe.cu` is not what makes P-ARCH-23I fast; the artifact-side speedup came from removing the packed Escha body projections from that path.

10. **Measured performance is the same class; evidence depth differs.** In documented matched 2K prefill runs with graphs disabled, P-ARCH-23I measured 620.978/621.992/621.469 ms (3295.36 tok/s average), then about 619 ms median / 3300.4 tok/s on confirmation. LowGPU measured 593.039/595.284/655.886 ms, reported as 614.736 ms / 3338.76 tok/s. The project therefore records P-ARCH-23I within about 1.2% of the LowGPU target. P-ARCH-23I passed coherent-completion quality and improved the lightweight parity probes (P1 5.6%, P2 38.8%, P6 8.8%, P7 100%, P5 0%). LowGPU separately has the milestone 75-case result, 66/75. P-ARCH-23I was not included in that full 75-case certification and should not yet be described as equivalently certified.

## What this means for Sean

These are different routes to the same prefill-performance class. LowGPU is the straight standard IQ3_XXS-class quant. P-ARCH-23I is an Escha-derived deployment hybrid: it reuses LowGPU's exact FFN and full-attention bytes, reconstructs the Escha-specific gate and linear-attention projections into standard Q2_K, retains Escha semantic markers, and keeps a packed LowGPU LM head. Its demonstrated speedup is evidence for standard GGML body representation and mature `MUL_MAT` kernels—not evidence that Bee's Escha codec kernel reached LowGPU speed.

## Verification gaps

- The GGUF itself does not encode `f0eadefa...`; that revision is established by the external project manifest. Likewise, the inspected LowGPU GGUF does not identify TheWegemann as author or provide a source repository—the file embeds `general.name=SafetensorKindergarten`. TheWegemann attribution is project-level provenance, while the importance-matrix fields are artifact-native.
- The original quantization commands and tool versions for LowGPU are not embedded beyond `general.file_type=24` and the importance-matrix provenance fields.
- P-ARCH-23I has only the documented completion/lightweight-parity pass. No 75-case milestone result for it was found; only LowGPU's 66/75 result is certified in the inspected records.
- Performance numbers are recorded benchmark evidence, not properties derivable from GGUF bytes. No models were executed for this comparison.

## Evidence basis

- Direct metadata and bounded family inventories: `/tmp/gguf_meta2_out.txt`, `/tmp/gguf_compare_out.txt`, and targeted `gguf-py` reads of the two named files.
- Construction semantics: `convert_escha_to_gguf.py` (`--standard-ffn-gguf`, `--standard-gdn-gate-quant`, `--embed-vocab`, `--head-vocab`, `--standard-attn-ffn`, `--standard-linear-ffn`).
- Runtime semantics and per-tensor dispatch: `src/models/qwen35.cpp`; Escha CUDA operator host dispatch: `ggml/src/ggml-cuda/escha-moe.cu`.
- Results and causal record: `docs/current-state.md`, `docs/escha-prefill-experiment-ledger.md`, and `evidence/EXP-retrospective/2026-09-02/SOL-RETROSPECTIVE.md`.
