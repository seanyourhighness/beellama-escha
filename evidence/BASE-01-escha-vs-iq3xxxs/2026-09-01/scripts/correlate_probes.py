#!/usr/bin/env python3
"""BASE-01 Phase 1b: block-alignment and cross-artifact probes for low-correlation projections.

1) attn_qkv: try [q;k;v] split alignments between Escha reconstruction and IQ3 GGUF.
2) ssm_out: cross-check Escha vs UD-IQ3 (standard quant of base) to see which side moved.
3) attn_gate: cross-check against UD-IQ3 as well (documented P-ARCH-21C mismatch).
"""
import sys, os, json
import numpy as np

REPO = "/mnt/d/CODEX WORKSPACE/beellama-escha"
ARM_A = "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf"
ARM_B = "/mnt/d/CODEX WORKSPACE/beellama-release/models/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf"
ARM_C = "/home/sean/Models/cpu-gguf/Qwen3.8-27B-UD-IQ3_XXS.gguf"

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

def tensor_float2d(t):
    data = t.data
    if t.tensor_type == 1:
        arr = np.array(data, copy=False).astype(np.float32)
    elif t.tensor_type == 0:
        arr = np.array(data, copy=False).astype(np.float32)
    else:
        arr = np.asarray(dequantize(data, t.tensor_type), dtype=np.float32)
    # GGUF tensor shape is [ne1, ne0]; ne0 fastest. Reshape to (ne1, ne0).
    ne1, ne0 = int(t.shape[0]), int(t.shape[1])
    return arr.reshape(ne1, ne0)

def escha_reconstruct(prefix, ic, oc, K):
    code = np.array(get_tensor(ra, prefix + ".escha_code").data, copy=False)
    rin = np.array(get_tensor(ra, prefix + ".escha_rin").data, copy=False).astype(np.float32)
    rout = np.array(get_tensor(ra, prefix + ".escha_rout").data, copy=False).astype(np.float32)
    w = reconstruct_deploy_weight(code, rin, rout, ic, oc, K, True, False)
    return np.ascontiguousarray(w.T, dtype=np.float32)  # [OC, IC] row-major

def corr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    ma, mb = a.mean(), b.mean()
    num = ((a - ma) * (b - mb)).sum()
    den = np.sqrt(((a - ma) ** 2).sum() * ((b - mb) ** 2).sum())
    return float(num / den) if den > 0 else float("nan")

ra = GGUFReader(ARM_A)
rb = GGUFReader(ARM_B)
rc = GGUFReader(ARM_C)

out = {}

# ---- 1) attn_qkv split probes ----
print("== attn_qkv (5120->10240) split probes ==")
# Escha recon [OC=10240, IC=5120]; IQ3 [5120, 10240] -> transpose for [OC, IC]
wa = escha_reconstruct("blk.0.attn_qkv", 5120, 10240, 2)   # [10240, 5120]
wbi = tensor_float2d(get_tensor(rb, "blk.0.attn_qkv.weight")).T  # [10240, 5120]
wci = tensor_float2d(get_tensor(rc, "blk.0.attn_qkv.weight")).T if get_tensor(rc, "blk.0.attn_qkv.weight") else None
print("full corr (IQ3):", round(corr(wa, wbi), 4))
if wci is not None:
    print("full corr (UD):", round(corr(wa, wci), 4))
# q=24 heads * 128 = 3072; k=4*128=512; v=4*128=512  (fused qkv: 12288? no - 10240 = q 24*128=3072? hmm)
# 5120->10240: q(24h*128=3072) + k(4h*128=512) + v(4h*128=512) = 4096... not 10240.
# For linear-attn qkv, likely q 5120, k 5120?? Actually in_proj_qkv -> 10240 = q(5120) + k(2560) + v(2560)?
for split in [(5120, 2560, 2560), (3072, 1024, 1024), (4096, 3072, 3072), (6144, 2048, 2048)]:
    q, k, v = split
    if q + k + v != 10240:
        continue
    cq = corr(wa[:q], wbi[:q]); ck = corr(wa[q:q+k], wbi[q:q+k]); cv = corr(wa[q+k:], wbi[q+k:])
    print(f"split q{k}v {split}: q={cq:.4f} k={ck:.4f} v={cv:.4f}")
out["attn_qkv"] = {"full_iq3": round(corr(wa, wbi), 4), "ud": round(corr(wa, wci), 4) if wci is not None else None}

# ---- 2) ssm_out cross-check ----
print("== ssm_out (6144->5120) cross-check ==")
wa = escha_reconstruct("blk.0.ssm_out", 6144, 5120, 2)   # [5120, 6144]
wbi = tensor_float2d(get_tensor(rb, "blk.0.ssm_out.weight")).T  # [5120, 6144]
wci = tensor_float2d(get_tensor(rc, "blk.0.ssm_out.weight")).T if get_tensor(rc, "blk.0.ssm_out.weight") else None
print("escha vs IQ3:", round(corr(wa, wbi), 4))
if wci is not None:
    print("escha vs UD:", round(corr(wa, wci), 4))
    print("IQ3  vs UD:", round(corr(wbi, wci), 4))
out["ssm_out"] = {"escha_vs_iq3": round(corr(wa, wbi), 4),
                  "escha_vs_ud": round(corr(wa, wci), 4) if wci is not None else None,
                  "iq3_vs_ud": round(corr(wbi, wci), 4) if wci is not None else None}

# ---- 3) attn_gate cross-check ----
print("== attn_gate (5120->6144) cross-check ==")
wa = escha_reconstruct("blk.0.attn_gate", 5120, 6144, 2)  # [6144, 5120]
wbi = tensor_float2d(get_tensor(rb, "blk.0.attn_gate.weight")).T
wci = tensor_float2d(get_tensor(rc, "blk.0.attn_gate.weight")).T if get_tensor(rc, "blk.0.attn_gate.weight") else None
print("escha vs IQ3:", round(corr(wa, wbi), 4))
if wci is not None:
    print("escha vs UD:", round(corr(wa, wci), 4))
    print("IQ3  vs UD:", round(corr(wbi, wci), 4))
out["attn_gate"] = {"escha_vs_iq3": round(corr(wa, wbi), 4),
                    "escha_vs_ud": round(corr(wa, wci), 4) if wci is not None else None,
                    "iq3_vs_ud": round(corr(wbi, wci), 4) if wci is not None else None}

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
print("WROTE", sys.argv[1])
