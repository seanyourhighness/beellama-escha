#!/usr/bin/env python3
"""BASE-01 Phase 3/6: projection family-gap accounting (Sol Gate 1 method).

Per-family gap = A_share_f * A_off - B_share_f * B_off, using share-scaled
graphs-off wall totals, then the canonical gap is reconciled with the measured
per-arm graphs-on/off deltas. Closure: >=95% or explicit remainder.
"""
import json, sys

def load_agg(path):
    d = json.load(open(path))
    total = d.get("_total_projection_ms") or d.get("_total_mulmat_ms") or sum(
        v["total_ms"] for k, v in d.items() if not k.startswith("_"))
    fam = {k: v for k, v in d.items() if not k.startswith("_")}
    for k, v in fam.items():
        v["share"] = v["total_ms"] / total
    return fam, total

A, A_tot = load_agg(sys.argv[1])   # ESCHA aggregate
B, B_tot = load_agg(sys.argv[2])   # IQ3 aggregate
A_off = float(sys.argv[3]) if len(sys.argv) > 3 else 816.2
B_off = float(sys.argv[4]) if len(sys.argv) > 4 else 590.3
A_on = float(sys.argv[5]) if len(sys.argv) > 5 else 880.192
B_on = float(sys.argv[6]) if len(sys.argv) > 6 else 637.484

# family mapping between arms
MAP = {
    "ffn_up": "ffn_up", "ffn_down": "ffn_down", "ffn_gate": "ffn_gate",
    "attn_qkv": "attn_qkv", "attn_gate": "attn_gate",
    "ssm_out_OR_attn_output": ["ssm_out", "attn_output"],
    "attn_q": "attn_q", "attn_kv": ["attn_k", "attn_v"],
}
# families that only appear on one side
B_ONLY = ["ssm_beta_alpha", "lm_head", "other"]

rows = []
usedB = set()
for af, bf in MAP.items():
    a = A.get(af)
    if isinstance(bf, list):
        b_tot = sum(B.get(x, {}).get("total_ms", 0.0) for x in bf)
        b_share = sum(B.get(x, {}).get("share", 0.0) for x in bf)
        b = {"total_ms": b_tot, "share": b_share}
        for x in bf:
            usedB.add(x)
    else:
        b = B.get(bf, {"total_ms": 0.0, "share": 0.0})
        usedB.add(bf)
    a_ms = a["share"] * A_off if a else 0.0
    b_ms = b["share"] * B_off if b else 0.0
    gap = a_ms - b_ms
    rows.append((af, a_ms, b_ms, gap))

for bf in B:
    if bf not in usedB:
        b_ms = B[bf]["share"] * B_off
        rows.append((bf + " (B only)", 0.0, b_ms, -b_ms))

rows.sort(key=lambda r: -abs(r[3]))
proj_gap = sum(r[3] for r in rows)
print(f"graphs-off totals: A={A_off:.1f} B={B_off:.1f} gap={A_off-B_off:.1f} ms")
print(f"{'family':36s} {'A ms':>9s} {'B ms':>9s} {'gap ms':>9s} {'% of gap':>9s}")
for name, a_ms, b_ms, gap in rows:
    print(f"{name:36s} {a_ms:9.1f} {b_ms:9.1f} {gap:+9.1f} {gap/(A_off-B_off)*100:8.1f}%")
print(f"{'PROJECTION GAP (share-scaled)':36s} {'':>9s} {'':>9s} {proj_gap:+9.1f} {proj_gap/(A_off-B_off)*100:8.1f}%")
closure = proj_gap / (A_off - B_off) * 100 if abs(A_off - B_off) > 1e-9 else float("nan")
print(f"closure of graphs-off gap by projection families: {closure:.1f}%")

# canonical reconciliation
print("\n== canonical reconciliation ==")
print(f"graphs-on gap: {A_on-B_on:.1f} ms")
print(f"graphs-off gap: {A_off-B_off:.1f} ms")
print(f"per-arm graph delta: A {A_on-A_off:+.1f} ms, B {B_on-B_off:+.1f} ms -> delta contribution {(A_on-A_off)-(B_on-B_off):+.1f} ms")
recon = (A_off - B_off) + ((A_on - A_off) - (B_on - B_off))
print(f"reconciled canonical gap: {recon:.1f} ms (measured {A_on-B_on:.1f} ms, diff {recon-(A_on-B_on):+.1f})")
