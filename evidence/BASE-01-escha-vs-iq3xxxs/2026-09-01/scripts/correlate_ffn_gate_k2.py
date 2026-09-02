#!/usr/bin/env python3
"""ffn_gate K=2 fix (code shape [32,1088,320] = 16*2)."""
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

def gt(r, n):
    for t in r.tensors:
        if t.name == n:
            return t

def flat(t):
    if t.tensor_type == 1:
        return np.array(t.data, copy=False).astype(np.float32).reshape(-1)
    if t.tensor_type == 0:
        return np.array(t.data, copy=False).astype(np.float32).reshape(-1)
    return np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(-1)

def corr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    ma, mb = a.mean(), b.mean()
    den = np.sqrt(((a - ma) ** 2).sum() * ((b - mb) ** 2).sum())
    return float(((a - ma) * (b - mb)).sum() / den) if den > 0 else float("nan")

ra = GGUFReader(ARM_A)
rb = GGUFReader(ARM_B)
out = {}
for prefix, ic, oc, K, label in [("blk.0.ffn_gate", 5120, 17408, 2, "ffn_gate"),
                                 ("blk.1.ffn_gate", 5120, 17408, 2, "ffn_gate_l1")]:
    code = np.array(gt(ra, prefix + ".escha_code").data, copy=False)
    rin = np.array(gt(ra, prefix + ".escha_rin").data, copy=False).astype(np.float32)
    rout = np.array(gt(ra, prefix + ".escha_rout").data, copy=False).astype(np.float32)
    wb = flat(gt(rb, prefix + ".weight"))
    for cn, code_in in [("as_is", code), ("transposed", np.transpose(code, (2, 1, 0)))]:
        try:
            w = reconstruct_deploy_weight(code_in, rin, rout, ic, oc, K, True, False)
            wa = np.ascontiguousarray(w.T, dtype=np.float32).reshape(-1)
            n = min(wa.size, wb.size)
            c = corr(wa[:n], wb[:n])
            print(label, cn, "K=", K, "corr=", round(c, 4))
            out[label + "_" + cn] = round(c, 4)
        except Exception as e:
            print(label, cn, "ERR", str(e)[:120])
            out[label + "_" + cn] = {"error": str(e)[:120]}
with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
print("WROTE", sys.argv[1])
