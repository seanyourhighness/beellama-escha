#!/usr/bin/env python3
"""BASE-01 Phase 1d: attn_qkv block dissection using raw-flat alignment.

Empirical orientation (validated by CORRELATION.json: ffn_up 0.87, attn_k 0.96):
raw GGUF tensor flat data == escha_reconstruct(...).T flat data (both (oc,ic)
order with ic fastest). Row block for oc in [a,b) is flat[a*IC : b*IC].
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

prefix = "blk.0.attn_qkv"
IC, OC = 5120, 10240
code = np.array(get_tensor(ra, prefix + ".escha_code").data, copy=False)
rin = np.array(get_tensor(ra, prefix + ".escha_rin").data, copy=False).astype(np.float32)
rout = np.array(get_tensor(ra, prefix + ".escha_rout").data, copy=False).astype(np.float32)
w = reconstruct_deploy_weight(code, rin, rout, IC, OC, 2, True, False)
wa = np.ascontiguousarray(w.T, dtype=np.float32).reshape(-1)  # (oc,ic) order
wb = raw_flat(get_tensor(rb, prefix + ".weight"))

print("full raw corr:", round(corr(wa, wb), 4))
out["full"] = round(corr(wa, wb), 4)

# Qwen3.5 linear-attn fused qkv: try plausible splits. q usually 5120 (hidden),
# kv heads small. Try (q, k, v) covering OC=10240.
splits = [
    ("q5120_k2560_v2560", 5120, 2560, 2560),
    ("q6144_k2048_v2048", 6144, 2048, 2048),
    ("q4096_k3072_v3072", 4096, 3072, 3072),
    ("q5120_k5120_v0", 5120, 5120, 0),
    ("q7168_k1536_v1536", 7168, 1536, 1536),
    ("q7680_k1280_v1280", 7680, 1280, 1280),
    ("q8192_k1024_v1024", 8192, 1024, 1024),
    ("q2560_k3840_v3840", 2560, 3840, 3840),
]
for label, q, k, v in splits:
    if q + k + v != OC:
        continue
    cq = corr(wa[0:q*IC], wb[0:q*IC])
    ck = corr(wa[q*IC:(q+k)*IC], wb[q*IC:(q+k)*IC]) if k else float("nan")
    cv = corr(wa[(q+k)*IC:], wb[(q+k)*IC:]) if v else float("nan")
    print(f"{label}: q={cq:.4f} k={ck:.4f} v={cv:.4f}")
    out[label] = {"q": round(cq,4), "k": round(ck,4), "v": round(cv,4)}

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
print("WROTE", sys.argv[1])
