#!/usr/bin/env python3
"""BASE-01 Phase 3: aggregate IQ3 GGML_OP_PROFILE lines per family/shape.

Input lines:
  GGML_OP_PROFILE op=mul_mat name=<n> ne=<n0>x<n1>x<n2>x<n3> type_src0=<t> rows=<r> cols=<c> stream=<p> total_ms=<ms> host_ms=<hms> err_el=0 err_q0=0 err_q1=0
Filter: only lines with host_ms > 0 (skips the reserve/no-op pass where events
measure 0 and the op did not execute real work).
Output: per-family aggregate + median per call.
"""
import sys, re, json, statistics
from collections import defaultdict

PAT = re.compile(
    r"GGML_OP_PROFILE op=mul_mat name=(\S+) ne=(\d+)x(\d+)x(\d+)x(\d+) "
    r"type_src0=(\d+) rows=(\d+) cols=(\d+) stream=(\S+) total_ms=([\d.]+) host_ms=([\d.]+)")

def parse(path):
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            host_ms = float(m.group(11))
            if host_ms <= 0.0:
                continue  # reserve/no-op pass
            rows.append({"name": m.group(1), "ne0": int(m.group(2)), "ne1": int(m.group(3)),
                         "ne2": int(m.group(4)), "ne3": int(m.group(5)),
                         "type_src0": int(m.group(6)), "rows": int(m.group(7)),
                         "cols": int(m.group(8)), "total_ms": float(m.group(10)),
                         "host_ms": host_ms})
    return rows

def iq3_fam(name, ne0=None, cols=None):
    n = name
    if "ffn_up" in n: return "ffn_up"
    if "ffn_down" in n or "ffn_out" in n: return "ffn_down"
    if "ffn_gate" in n: return "ffn_gate"
    if "attn_gate" in n or "z-" in n or n.startswith("z_"): return "attn_gate"
    if "attn_qkv" in n: return "attn_qkv"
    if "Vcur" in n or "attn_v" in n: return "attn_v"
    if "Kcur" in n or "attn_k" in n: return "attn_k"
    if "Qcur" in n or "attn_q" in n: return "attn_q"
    if "ssm_out" in n or "linear_attn_out" in n: return "ssm_out"
    if "attn_output" in n: return "attn_output"
    if "result_output" in n or n.startswith("output"): return "lm_head"
    if "token_embd" in n or "get_rows" in n: return "embedding"
    # unnamed nodes: classify by shape (cols x ne0)
    if cols is not None and ne0 is not None:
        if ne0 == 10240 and cols == 5120: return "attn_qkv"
        if ne0 == 6144 and cols == 5120: return "attn_gate"
        if ne0 == 5120 and cols == 6144: return "ssm_out_OR_attn_output"
        if ne0 == 17408 and cols == 5120: return "ffn_gate_OR_ffn_up"
        if ne0 == 5120 and cols == 17408: return "ffn_down"
        if ne0 == 12288 and cols == 5120: return "attn_q"
        if ne0 == 1024 and cols == 5120: return "attn_kv"
        if ne0 == 48 and cols == 5120: return "ssm_beta_alpha"
        if ne0 == 248320 and cols == 5120: return "lm_head"
    return "other_" + n

def main():
    files = sys.argv[1:]
    out_json = None
    if len(files) > 1 and files[-1].endswith(".json"):
        out_json = files[-1]
        files = files[:-1]
    allrows = []
    for p in files:
        allrows.extend(parse(p))
    print(f"parsed {len(allrows)} executed GGML_OP_PROFILE lines from {len(files)} file(s)")
    fam = defaultdict(list)
    for r in allrows:
        fam[iq3_fam(r["name"], ne0=r["ne0"], cols=r["cols"])].append(r)
    out = {}
    total = sum(r["total_ms"] for r in allrows)
    for f, rows in sorted(fam.items(), key=lambda kv: -sum(r["total_ms"] for r in kv[1])):
        n = len(rows)
        agg = sum(r["total_ms"] for r in rows)
        med = statistics.median(r["total_ms"] for r in rows)
        shape = f"{rows[0]['cols']}x{rows[0]['ne0']}"
        print(f"{f:24s} calls={n:4d} total_ms={agg:9.1f} med/call={med:.4f} shape={shape} share={agg/total*100:.1f}%")
        out[f] = {"calls": n, "total_ms": round(agg, 2), "median_call_ms": round(med, 4),
                  "shape": shape, "share_pct": round(agg / total * 100, 2)}
    print(f"TOTAL executed mul_mat ms: {total:.1f}")
    out["_total_mulmat_ms"] = round(total, 2)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(out, f, indent=2)
            print("WROTE", out_json)

if __name__ == "__main__":
    main()
