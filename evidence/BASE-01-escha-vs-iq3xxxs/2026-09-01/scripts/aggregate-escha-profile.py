#!/usr/bin/env python3
"""BASE-01 Phase 3: aggregate ESCHA_PROFILE output per family/shape.

Input: stderr text with lines:
  ESCHA_PROFILE k=%d ic=%d oc=%d rows=%d gen=%d route=%s total_ms=... rotate_ms=... matmul_ms=... epilogue_ms=...
Output: per-family and per-shape aggregate + per-call medians.
"""
import sys, re, json, statistics
from collections import defaultdict

def parse(path):
    rows = []
    pat = re.compile(
        r"ESCHA_PROFILE k=(\d+) ic=(\d+) oc=(\d+) rows=(\d+) gen=(\d+) route=(\S+) "
        r"total_ms=([\d.]+) rotate_ms=([\d.]+) matmul_ms=([\d.]+) epilogue_ms=([\d.]+)")
    with open(path, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append({
                    "k": int(m.group(1)), "ic": int(m.group(2)), "oc": int(m.group(3)),
                    "rows": int(m.group(4)), "gen": int(m.group(5)), "route": m.group(6),
                    "total_ms": float(m.group(7)), "rotate_ms": float(m.group(8)),
                    "matmul_ms": float(m.group(9)), "epilogue_ms": float(m.group(10)),
                })
    return rows

def family(ic, oc, k):
    # map (ic, oc) to projection family by known shapes
    if ic == 5120 and oc == 17408 and k == 3:
        return "ffn_up"
    if ic == 17408 and oc == 5120 and k == 3:
        return "ffn_down"
    if ic == 5120 and oc == 17408 and k == 2:
        return "ffn_gate"
    if ic == 5120 and oc == 6144 and k == 2:
        return "attn_gate"
    if ic == 5120 and oc == 10240 and k == 2:
        return "attn_qkv"
    if ic == 5120 and oc == 12288 and k == 2:
        return "attn_q"
    if ic == 5120 and oc == 1024 and k == 2:
        return "attn_kv"
    if ic == 6144 and oc == 5120 and k == 2:
        return "ssm_out_OR_attn_output"  # same shape; disambiguated by call order/count
    if ic == 17408 and oc == 5120 and k == 2:
        return "ffn_down_k2"
    return f"other_{ic}x{oc}_k{k}"

def main():
    files = sys.argv[1:]
    out_json = None
    if len(files) > 1 and files[-1].endswith(".json"):
        out_json = files[-1]
        files = files[:-1]
    allrows = []
    for p in files:
        allrows.extend(parse(p))
    print(f"parsed {len(allrows)} ESCHA_PROFILE lines from {len(files)} file(s)")
    fam = defaultdict(list)
    for r in allrows:
        f = family(r["ic"], r["oc"], r["k"])
        r["family"] = f
        fam[f].append(r)
    out = {}
    for f, rows in sorted(fam.items()):
        n = len(rows)
        tot = sum(r["total_ms"] for r in rows)
        mm = sum(r["matmul_ms"] for r in rows)
        rt = sum(r["rotate_ms"] for r in rows)
        ep = sum(r["epilogue_ms"] for r in rows)
        med = statistics.median(r["total_ms"] for r in rows)
        med_mm = statistics.median(r["matmul_ms"] for r in rows)
        print(f"{f}: calls={n} total_ms={tot:.1f} matmul_ms={mm:.1f} rotate_ms={rt:.1f} epilogue_ms={ep:.1f} "
              f"median/call={med:.4f} median_matmul={med_mm:.4f} share_of_model={tot/sum(sum(x['total_ms'] for x in v) for v in fam.values())*100:.1f}%")
        routes = defaultdict(int)
        for r in rows:
            routes[r["route"]] += 1
        out[f] = {"calls": n, "total_ms": round(tot, 2), "matmul_ms": round(mm, 2),
                  "rotate_ms": round(rt, 2), "epilogue_ms": round(ep, 2),
                  "median_call_ms": round(med, 4), "median_matmul_ms": round(med_mm, 4),
                  "routes": dict(routes), "shape": f"{rows[0]['ic']}x{rows[0]['oc']} k={rows[0]['k']}"}
    grand = sum(sum(r["total_ms"] for r in v) for v in fam.values())
    print(f"TOTAL projection ms: {grand:.1f}")
    out["_total_projection_ms"] = round(grand, 2)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(out, f, indent=2)
            print("WROTE", out_json)

if __name__ == "__main__":
    main()
