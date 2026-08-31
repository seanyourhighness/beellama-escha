#!/usr/bin/env python3
"""Verify that the P-ARCH-18 F16-vocabulary control differs only at vocab."""

from __future__ import annotations

import hashlib
import sys
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gguf-py"))
sys.path.insert(0, str(REPO))

import gguf  # noqa: E402
from convert_escha_to_gguf import dequant_lowgpu_rows  # noqa: E402

HYBRID = Path(os.environ.get("ESCHA_HYBRID", "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf"))
CANDIDATE = Path(os.environ.get("ESCHA_CANDIDATE", "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity-p18-f16vocab.gguf"))
SAFETENSORS = Path(os.environ.get("ESCHA_SF", "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono/model-00001-of-00002.safetensors"))


def digest(t: gguf.ReaderTensor) -> str:
    a = np.ascontiguousarray(t.data)
    return hashlib.sha256(memoryview(a).cast("B")).hexdigest()


def main() -> None:
    h = {t.name: t for t in gguf.GGUFReader(HYBRID).tensors}
    c = {t.name: t for t in gguf.GGUFReader(CANDIDATE).tensors}
    common = sorted(h.keys() & c.keys())
    assert len(common) == 2052, len(common)
    assert set(h) - set(c) == {
        "token_embd.lowgpu_codes", "token_embd.lowgpu_scales", "token_embd.lowgpu_zps",
        "output.lowgpu_codes", "output.lowgpu_scales", "output.lowgpu_zps",
    }
    assert set(c) - set(h) == {"token_embd.weight", "output.weight"}

    mismatches = []
    for name in common:
        ht, ct = h[name], c[name]
        if tuple(ht.shape) != tuple(ct.shape) or ht.tensor_type != ct.tensor_type or digest(ht) != digest(ct):
            mismatches.append(name)
    assert not mismatches, mismatches[:10]
    print(f"common tensors byte-identical: {len(common)}")

    sf = safe_open(str(SAFETENSORS), framework="np")
    for name, prefix in (("token_embd.weight", "model.language_model.embed_tokens"),
                         ("output.weight", "lm_head")):
        codes = sf.get_tensor(prefix + ".weight_lowgpu_codes")
        ref = dequant_lowgpu_rows(
            codes,
            sf.get_tensor(prefix + ".weight_lowgpu_scales"),
            sf.get_tensor(prefix + ".weight_lowgpu_zps"),
            0, codes.shape[0],
        )
        got = np.asarray(c[name].data, dtype=np.float16).reshape(ref.shape)
        d = np.abs(got.astype(np.float32) - ref.astype(np.float32)).max()
        assert d == 0.0, (name, d)
        print(f"{name}: exact LowGPU dequant")


if __name__ == "__main__":
    main()
