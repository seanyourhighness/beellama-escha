#!/usr/bin/env python3
"""BASE-01 Phase 1e: permutation/order probes for low-correlation blocks.

Question: are attn_qkv v-block, attn_gate, ssm_out genuinely different weights
between ESCHA and IQ3, or the same weights in a different order/layout?
Test block-swap / transpose candidates; report best achievable correlation.
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

def escha_recon(prefix, ic, oc, K):
    code = np.array(get_tensor(ra, prefix + ".escha_code").data, copy=False)
    rin = np.array(get_tensor(ra, prefix + ".escha_rin").data, copy=False).astype(np.float32)
    rout = np.array(get_tensor(ra, prefix + ".escha_rout").data, copy=False).astype(np.float32)
    w = reconstruct_deploy_weight(code, rin, rout, ic, oc, K, True, False)
    return np.ascontiguousarray(w.T, dtype=np.float32)  # [OC, IC] row-major (matches GGUF flat)

ra = GGUFReader(ARM_A)
rb = GGUFReader(ARM_B)
out = {}

# --- ssm_out: try transposed (swap OC/IC) and reversed dims ---
print("== ssm_out (6144->5120): order probes ==")
wa = escha_recon("blk.0.ssm_out", 6144, 5120, 2)   # [5120, 6144]
wb = raw_flat(get_tensor(rb, "blk.0.ssm_out.weight"))  # GGUF flat [IC, OC]
wbf = wb.reshape(6144, 5120)  # GGUF as [IC=6144, OC=5120] row-major? Actually flat = ic-major? verify
# GGUF flat for shape [IC, OC] with ne0=OC fastest: index = ic*OC + oc
# wa is [OC, IC] row-major numpy: index = oc*IC + ic  => wa.T is [IC, OC] with index ic*OC+oc
waT = wa.T  # [IC, OC]
print("  direct (raw):", round(corr(wa.reshape(-1), wb), 4))
print("  transposed :", round(corr(waT.reshape(-1), wb), 4))
out["ssm_out"] = {"direct": round(corr(wa.reshape(-1), wb), 4), "transposed": round(corr(waT.reshape(-1), wb), 4)}

# --- attn_gate ---
print("== attn_gate (5120->6144): order probes ==")
wa = escha_recon("blk.0.attn_gate", 5120, 6144, 2)  # [6144, 5120]
wb = raw_flat(get_tensor(rb, "blk.0.attn_gate.weight"))
waT = wa.T
print("  direct:", round(corr(wa.reshape(-1), wb), 4))
print("  transposed:", round(corr(waT.reshape(-1), wb), 4))
out["attn_gate"] = {"direct": round(corr(wa.reshape(-1), wb), 4), "transposed": round(corr(waT.reshape(-1), wb), 4)}

# --- attn_qkv v-block (last 6144 rows of OC=10240): try matching v against
# various 6144-row windows of the IQ3 tensor and both orders ---
print("== attn_qkv v-block (6144): window/order probes ==")
wa = escha_recon("blk.0.attn_qkv", 5120, 10240, 2)  # [10240, 5120]
wb = raw_flat(get_tensor(rb, "blk.0.attn_qkv.weight"))  # GGUF flat [IC=5120, OC=10240]
wa_v = wa[4096:, :]  # last 6144 OC rows, [6144, 5120]
wb_m = wb.reshape(5120, 10240)  # [IC, OC]? -> transpose to [OC, IC]
wb_m = wb_m.T  # [OC=10240, IC=5120]
best = []
for start in range(0, 10240 - 6144 + 1, 512):
    wb_win = wb_m[start:start+6144, :]
    c = corr(wa_v.reshape(-1), wb_win.reshape(-1))
    best.append((c, start))
best.sort(reverse=True)
print("  best v-window corr:", [(round(c, 4), s) for c, s in best[:5]])
out["attn_qkv_vblock"] = {"best_window": [(round(c,4), s) for c, s in best[:5]]}

# Also check if IQ3 v-block corresponds to ESCHA q+k (i.e., block order differs)
wa_qk = wa[:4096, :]
for start in range(0, 10240 - 4096 + 1, 512):
    wb_win = wb_m[start:start+4096, :]
    c = corr(wa_qk.reshape(-1), wb_win.reshape(-1))
    if c > 0.5:
        print("  qk matches window at", start, round(c, 4))

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
print("WROTE", sys.argv[1])
