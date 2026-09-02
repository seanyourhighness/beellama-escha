#!/usr/bin/env python3
"""BASE-01 Phase 1c: corrected block-alignment probes.

GGUF weight tensors are stored [IC, OC] row-major; escha_reconstruct returns
w.T which is also [OC, IC] numpy => row-major flatten of [IC, OC]? Fix:
- escha_reconstruct returns np.ascontiguousarray(w.T) with numpy shape (OC, IC).
- GGUF tensor data flattened is row-major over shape (ne1=IC, ne0=OC) => element
  (i,j) = W[i*OC + j] where i in [0,IC), j in [0,OC). That equals W numpy [IC, OC].
- So correct comparison: wa_np.T (numpy [IC, OC]) flattened == wb_flat.
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

def gguf_flat_2d(t):
    data = t.data
    if t.tensor_type == 1:
        arr = np.array(data, copy=False).astype(np.float32)
    elif t.tensor_type == 0:
        arr = np.array(data, copy=False).astype(np.float32)
    else:
        arr = np.asarray(dequantize(data, t.tensor_type), dtype=np.float32)
    ne1, ne0 = int(t.shape[0]), int(t.shape[1])
    return arr.reshape(ne1, ne0)  # numpy [IC, OC] row-major

def escha_np(prefix, ic, oc, K):
    code = np.array(get_tensor(ra, prefix + ".escha_code").data, copy=False)
    rin = np.array(get_tensor(ra, prefix + ".escha_rin").data, copy=False).astype(np.float32)
    rout = np.array(get_tensor(ra, prefix + ".escha_rout").data, copy=False).astype(np.float32)
    w = reconstruct_deploy_weight(code, rin, rout, ic, oc, K, True, False)
    # w numpy shape? converter does w.T before add_tensor; we need numpy [IC, OC].
    # If w is [IC, OC], w.T is [OC, IC]; GGUF flat is [IC, OC] row-major == w.
    return np.ascontiguousarray(w, dtype=np.float32)

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

def cmp_proj(name, prefix, ic, oc, K):
    try:
        wa = escha_np(prefix, ic, oc, K)  # expect [IC, OC]
    except Exception as e:
        print(name, "escha error", e)
        return
    fa = wa.reshape(-1)
    print("==", name, "wa shape", wa.shape)
    for tag, rdr in (("IQ3", rb), ("UD", rc)):
        bt = get_tensor(rdr, prefix + ".weight")
        if bt is None:
            print("  missing", tag)
            continue
        wb = gguf_flat_2d(bt)  # [IC, OC]
        fb = wb.reshape(-1)
        n = min(fa.size, fb.size)
        print(f"  {tag}: corr={corr(fa[:n], fb[:n]):.4f} type={int(bt.tensor_type)} shape={[int(x) for x in bt.shape]}")
    return wa

# attn_qkv split probes
wa = cmp_proj("attn_qkv", "blk.0.attn_qkv", 5120, 10240, 2)
bt = get_tensor(rb, "blk.0.attn_qkv.weight")
wb = gguf_flat_2d(bt)
for split in [(5120, 2560, 2560), (3072, 1024, 1024), (4096, 3072, 3072), (6144, 2048, 2048), (2560, 5120, 2560), (2048, 6144, 2048), (3072, 2048, 5120), (5120, 5120, 0)]:
    q, k, v = split
    if q + k + v != 10240:
        continue
    try:
        cq = corr(wa[:q].reshape(-1), wb[:q].reshape(-1))
        ck = corr(wa[q:q+k].reshape(-1), wb[q:q+k].reshape(-1)) if k else float("nan")
        cv = corr(wa[q+k:].reshape(-1), wb[q+k:].reshape(-1)) if v else float("nan")
        print(f"split {split}: q={cq:.4f} k={ck:.4f} v={cv:.4f}")
    except Exception as e:
        print("split err", split, e)

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=2)
