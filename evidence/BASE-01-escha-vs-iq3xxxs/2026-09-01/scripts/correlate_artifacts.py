#!/usr/bin/env python3
"""BASE-01 Phase 1: provenance correlation between Arm A (ESCHA) and Arm B (IQ3 LowGPU).

Samples corresponding projection families from both artifacts, dequantizes them
(ESCHA via escham_cpu.reconstruct_deploy_weight; IQ3 via gguf quants dequantize),
and reports correlation + cosine similarity to establish common ancestry.

Read-only: no writes to artifacts; writes only the report JSON.
"""
import sys, json, os
import numpy as np

REPO = "/mnt/d/CODEX WORKSPACE/beellama-escha"
ARM_A = "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf"
ARM_B = "/mnt/d/CODEX WORKSPACE/beellama-release/models/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/correlation.json"

sys.path.insert(0, os.path.join(REPO, "gguf-py"))
sys.path.insert(0, "/home/sean/research/escha-refs/yaniss/tools/escha")
from gguf import GGUFReader
from gguf.quants import dequantize
from escham_cpu import reconstruct_deploy_weight

ra = GGUFReader(ARM_A)
rb = GGUFReader(ARM_B)

def get_tensor(reader, name):
    for t in reader.tensors:
        if t.name == name:
            return t
    return None

def tensor_data(t, n):
    """Return float32 flattened data for a GGUF tensor (dequantized)."""
    data = t.data
    if t.tensor_type in (0, 1):  # F32 / F16
        arr = np.array(data, copy=False)
        if t.tensor_type == 1:
            arr = arr.astype(np.float32)
        return arr.reshape(-1).astype(np.float32)
    # quantized: dequantize via gguf
    try:
        dq = dequantize(data, t.tensor_type)
        return np.asarray(dq, dtype=np.float32).reshape(-1)
    except Exception as e:
        return None

def corr_cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    ma, mb = a.mean(), b.mean()
    num = ((a - ma) * (b - mb)).sum()
    den = np.sqrt(((a - ma) ** 2).sum() * ((b - mb) ** 2).sum())
    corr = num / den if den > 0 else float("nan")
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return corr, cos

def escha_reconstruct(prefix, ic, oc, K):
    code_t = get_tensor(ra, prefix + ".escha_code")
    rin_t = get_tensor(ra, prefix + ".escha_rin")
    rout_t = get_tensor(ra, prefix + ".escha_rout")
    code = np.array(code_t.data, copy=False)
    rin = np.array(rin_t.data, copy=False).astype(np.float32)
    rout = np.array(rout_t.data, copy=False).astype(np.float32)
    w = reconstruct_deploy_weight(code, rin, rout, ic, oc, K, True, False)
    # converter writes w.T into GGUF [OC, IC]; replicate
    return np.ascontiguousarray(w.T, dtype=np.float32)

# sample projections: layer 0 (linear attn) and layer 3 (full attn)
samples = [
    # (label, escha_prefix, ic, oc, K)
    ("blk.0.attn_qkv",    "blk.0.attn_qkv",  5120, 10240, 2),
    ("blk.0.attn_gate",   "blk.0.attn_gate", 5120, 6144,  2),
    ("blk.0.ssm_out",     "blk.0.ssm_out",   6144, 5120,  2),
    ("blk.0.ffn_gate",    "blk.0.ffn_gate",  5120, 17408, 3),
    ("blk.0.ffn_up",      "blk.0.ffn_up",    5120, 17408, 3),
    ("blk.0.ffn_down",    "blk.0.ffn_down",  17408, 5120, 3),
    ("blk.3.attn_q",      "blk.3.attn_q",    5120, 12288, 2),
    ("blk.3.attn_k",      "blk.3.attn_k",    5120, 1024,  2),
    ("blk.3.attn_v",      "blk.3.attn_v",    5120, 1024,  2),
    ("blk.3.attn_output", "blk.3.attn_output", 5120, 5120, 2),
]

# Arm B corresponding tensor name is the same label + ".weight"
results = []
for label, prefix, ic, oc, K in samples:
    row = {"projection": label, "ic": ic, "oc": oc, "K": K}
    try:
        wa = escha_reconstruct(prefix, ic, oc, K)
        row["escha_shape"] = [int(x) for x in wa.shape]
    except Exception as e:
        row["escha_error"] = str(e)
        results.append(row)
        continue
    bt = get_tensor(rb, label + ".weight")
    if bt is None:
        row["iq3_error"] = "missing tensor " + label + ".weight"
        results.append(row)
        continue
    wb = tensor_data(bt, bt.n_elements)
    if wb is None:
        row["iq3_error"] = f"dequant failed type={bt.tensor_type}"
        results.append(row)
        continue
    row["iq3_type"] = int(bt.tensor_type)
    row["iq3_shape"] = [int(x) for x in bt.shape]
    # align: wa is [OC, IC] row-major (flattened); wb is GGUF row-major [OC, IC]
    fa = wa.reshape(-1)
    fb = wb.reshape(-1)
    n = min(fa.size, fb.size)
    fa, fb = fa[:n], fb[:n]
    corr, cos = corr_cos(fa, fb)
    row["n_aligned"] = int(n)
    row["correlation"] = round(float(corr), 6)
    row["cosine"] = round(float(cos), 6)
    # scale check
    row["escha_std"] = round(float(fa.std()), 6)
    row["iq3_std"] = round(float(fb.std()), 6)
    results.append(row)
    print(f"{label}: corr={corr:.4f} cos={cos:.4f} n={n} (escha_std {fa.std():.4f} vs iq3_std {fb.std():.4f})", flush=True)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print("WROTE", OUT)
