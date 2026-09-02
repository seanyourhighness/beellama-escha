#!/usr/bin/env python3
"""BASE-01 Phase 1g: retry ffn_gate + attn_output with corrected dims/orientation.

attn_output: full-attn output proj is (n_embd_head_k*n_head)=6144 -> n_embd=5120.
ffn_gate: 5120 -> 17408. Try both code orientations.
"""
import sys, os, json
import numpy as np

REPO = "/mnt/d/CODEX WORKSPACE/beellama-escha"
ARM_A = "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf"
ARM_B = "/mnt/d/CODEX WORKSPACE/beellama-release/models/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf"

sys.path.insert(0, os.path.join(REPO, "gguf-py"))
sys.path.insert(0, "/home/sean/research/escha-refs/yaniss/tools/escha")
from gguf import GGUFReader
from gguf.quants import dequantize
from escham_cpu import reconstruct_deploy_weight

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

cases = [
    ("blk.0.ffn_gate", 5120, 17408, 3, "ffn_gate"),
    ("blk.3.attn_output", 6144, 5120, 2, "attn_output"),
]
for prefix, ic, oc, K, label in cases:
    row = {}
    ct = get_tensor(ra, prefix + ".escha_code")
    code = np.array(ct.data, copy=False)
    rin = np.array(get_tensor(ra, prefix + ".escha_rin").data, copy=False).astype(np.float32)
    rout = np.array(get_tensor(ra, prefix + ".escha_rout").data, copy=False).astype(np.float32)
    row["code_data_shape"] = [int(x) for x in code.shape]
    row["code_attr_shape"] = [int(x) for x in ct.shape]
    wb = raw_flat(get_tensor(rb, prefix + ".weight"))
    for layout_name, code_in in [
        ("as_is", code),
        ("transposed", np.transpose(code, (2, 1, 0))),
    ]:
        try:
            w = reconstruct_deploy_weight(code_in, rin, rout, ic, oc, K, True, False)
            wa = np.ascontiguousarray(w.T, dtype=np.float32).reshape(-1)
            n = min(wa.size, wb.size)
            c = corr(wa[:n], wb[:n])
            row[layout_name] = {"corr": round(c, 4), "w_shape": [int(x) for x in w.shape]}
            print(f"{label} [{layout_name}] ic={ic} oc={oc}: corr={c:.4f}")
        except Exception as e:
            row[layout_name] = {"error": str(e)[:140]}
            print(f"{label} [{layout_name}]: ERR {str(e)[:110]}")
    out[label] = row

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
print("WROTE", sys.argv[1])
