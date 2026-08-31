#!/usr/bin/env python3
"""Spot-check the dequantized GGUF vocab against the reference LowGPU dequant."""

import sys
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gguf-py"))

import gguf  # noqa: E402
from convert_escha_to_gguf import dequant_lowgpu_rows  # noqa: E402

GGUF = os.environ.get("ESCHA_GGUF", "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-dequant.gguf")
SF = os.environ.get("ESCHA_SF", "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono/model-00001-of-00002.safetensors")

r = gguf.GGUFReader(GGUF)
for name in ("token_embd.weight", "output.weight"):
    t = next(x for x in r.tensors if x.name == name)
    prefix = "model.language_model.embed_tokens" if name.startswith("token") else "lm_head"
    f = safe_open(SF, framework="np")
    codes = f.get_tensor(prefix + ".weight_lowgpu_codes")
    scales = f.get_tensor(prefix + ".weight_lowgpu_scales")
    zps = f.get_tensor(prefix + ".weight_lowgpu_zps")
    v = codes.shape[0]
    ref = dequant_lowgpu_rows(codes, scales, zps, 0, v)
    got = np.asarray(t.data, dtype=np.float16).reshape(v, -1)
    d = np.abs(got.astype(np.float32) - ref.astype(np.float32)).max()
    print(f"{name}: shape {t.shape}, max_abs_diff {d}")
    assert d == 0.0, f"{name} mismatch"
print("vocab dequant bit-exact OK")
