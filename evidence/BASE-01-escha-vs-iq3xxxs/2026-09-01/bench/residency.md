# BASE-01 residency proof (block 1)

## Arm A
- layer lines: 65
- devices: {'CUDA0,': 65}
- CPU_Mapped/CPU buffer lines:
  - `done_getting_tensors: tensor 'token_embd.lowgpu_codes' (i8) (and 5 others) cannot be used with preferred buffer type CUDA_Host, using CPU instead`
  - `load_tensors:   CPU_Mapped model buffer size =  7726.18 MiB`
- VRAM bytes: `{'cuda_used_model_bytes': 0, 'cuda_used_context_bytes': 0, 'cuda_used_peak_bytes': 0, 'cuda_total_bytes': 0, 'cuda_context_buffer_bytes': 0, 'cuda_compute_buffer_bytes': 0}`

## Arm B
- layer lines: 65
- devices: {'CUDA0,': 65}
- CPU_Mapped/CPU buffer lines:
  - `done_getting_tensors: tensor 'token_embd.weight' (iq4_xs) (and 0 others) cannot be used with preferred buffer type CUDA_Host, using CPU instead`
  - `load_tensors:   CPU_Mapped model buffer size =   644.14 MiB`
- VRAM bytes: `{'cuda_used_model_bytes': 0, 'cuda_used_context_bytes': 0, 'cuda_used_peak_bytes': 0, 'cuda_total_bytes': 0, 'cuda_context_buffer_bytes': 0, 'cuda_compute_buffer_bytes': 0}`

