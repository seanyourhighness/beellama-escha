#!/usr/bin/env python3
"""BASE-01 Phase 1f: fix ffn_gate + attn_output reconstruction (try both layouts)."""
import sys, os, json
import numpy as np

REPO = "/mnt/d/CODEX WORKSPACE/beellama-escha"
ARM_A = "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf"
ARM_B = "/mnt/d/CODEX WORKSPACE/beellama-release/models/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf"

sys.path.insert(0, os.path.join(REPO, "gguf-py"))
sys.path.insert(0, "/home/sean/research/escha-refs/yaniss/tools/escha")
from gguf import GGUFReader
from gguf.quants import dequantize
from escham_cpu import reconstruct_deploy_weight, reconstruct_code

def get_tensor(reader, name):
    for t in reader.tensors:
        if t.name == name:
            return t
    return None

def raw_flat(t):
    data = t.data
    if t.tensor_type == 1:
        return np.array(data, copy=False).astype(np.float32).reshape(-1)
    elif t.tensor_type == 0:
        return np.array(data, copy=False).astype(np.float32).reshape(-1)
    return np.asarray(dequantize(data, t.tensor_type), dtype=np.float32).reshape(-1)

def corr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    ma, mb = a.mean(), b.mean()
    num = ((a - ma) * (b - mb)).sum()
    den = np.sqrt(((a - ma) ** 2).sum() * ((b - mb) ** 2).sum())
    return float(num / den) if den > 0 else float("nan")

ra = GGUFReader(ARM_A)
rb = GGUFReader(ARM_B)
out = {}

for prefix, ic, oc, K, label in [
    ("blk.0.ffn_gate", 5120, 17408, 3, "ffn_gate"),
    ("blk.3.attn_output", 5120, 5120, 2, "attn_output"),
]:
    row = {}
    code = np.array(get_tensor(ra, prefix + ".escha_code").data, copy=False)
    rin = np.array(get_tensor(ra, prefix + ".escha_rin").data, copy=False).astype(np.float32)
    rout = np.array(get_tensor(ra, prefix + ".escha_rout").data, copy=False).astype(np.float32)
    row["code_shape"] = [int(x) for x in code.shape]
    for layout_name, code_in in [
        ("gguf_layout", code),
        ("transposed", np.transpose(code, (2, 1, 0))),
    ]:
        try:
            w = reconstruct_deploy_weight(code_in, rin, rout, ic, oc, K, True, False)
            wa = np.ascontiguousarray(w.T, dtype=np.float32).reshape(-1)
            wb = raw_flat(get_tensor(rb, prefix + ".weight"))
            row[layout_name] = {"corr": round(corr(wa, wb), 4), "ok": True}
            print(f"{label} {layout_name}: corr={corr(wa, wb):.4f}")
        except Exception as e:
            row[layout_name] = {"corr": None, "ok": False, "error": str(e)[:160]}
            print(f"{label} {layout_name}: ERROR {str(e)[:120]}")
    out[label] = row

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
print("WROTE", sys.argv[1])
