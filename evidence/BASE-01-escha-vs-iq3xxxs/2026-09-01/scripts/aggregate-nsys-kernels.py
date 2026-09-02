#!/usr/bin/env python3
"""BASE-01 Phase 3: aggregate nsys kernel stats (IQ3 arm) by kernel name.

Input: nsys stats cuda_gpu_kern_sum CSV with columns
  Time (ns), Time (%), Instances, Avg (ns), Med (ns), Min (ns), Max (ns), Name
Output: per-kernel aggregate, grouped into operator families.
"""
import sys, csv, json
from collections import defaultdict

def load(fn):
    rows = []
    with open(fn, newline="", errors="replace") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if len(row) < 8:
                continue
            try:
                rows.append({
                    "time_ns": float(row[0]),
                    "pct": float(row[1]),
                    "instances": int(row[2]),
                    "avg_ns": float(row[3]),
                    "med_ns": float(row[4]),
                    "name": row[7],
                })
            except ValueError:
                continue
    return rows

def group(name):
    n = name.lower()
    if "escha" in n:
        return "escha_projection"
    if "flash_attn" in n or "flash-attn" in n or "fattn" in n:
        return "attention_flash"
    if "get_rows" in n:
        return "embedding_get_rows"
    if "norm" in n:
        return "normalization"
    if "rms" in n:
        return "normalization"
    if "rope" in n:
        return "rope"
    if "mmvq" in n or "mul_mat" in n and "q_" in n:
        return "quant_mulmat_mmq"
    if "mul_mat" in n:
        return "mul_mat"
    if "quant" in n and ("q8" in n or "q4" in n or "iq" in n or "q2" in n or "dequant" in n):
        return "dequant"
    if "cpy" in n or "copy" in n:
        return "memory_copy"
    if "memset" in n or "fill" in n:
        return "memory_fill"
    if "add" in n or "mul" in n or "silu" in n or "gelu" in n or "relu" in n or "sq" in n:
        return "elementwise"
    if "concat" in n or "permute" in n or "reshape" in n or "view" in n:
        return "layout"
    if "reduce" in n or "sum" in n:
        return "reduce"
    return "other"

def main():
    files = sys.argv[1:]
    agg = defaultdict(lambda: {"time_ns": 0.0, "instances": 0, "avg_ns": 0.0})
    detail = []
    for fn in files:
        rows = load(fn)
        print(f"== {fn}: {len(rows)} kernel rows, total {sum(r['time_ns'] for r in rows)/1e6:.1f} ms")
        for r in rows:
            g = group(r["name"])
            agg[g]["time_ns"] += r["time_ns"]
            agg[g]["instances"] += r["instances"]
            detail.append({"group": g, **r})
    total = sum(v["time_ns"] for v in agg.values())
    print("== grouped ==")
    out = {}
    for g, v in sorted(agg.items(), key=lambda x: -x[1]["time_ns"]):
        ms = v["time_ns"] / 1e6
        pct = v["time_ns"] / total * 100 if total else 0
        print(f"{g}: {ms:.1f} ms ({pct:.1f}%) instances={v['instances']} avg={v['time_ns']/v['instances']/1e3:.1f} us")
        out[g] = {"ms": round(ms, 2), "pct": round(pct, 2), "instances": v["instances"],
                  "avg_us": round(v["time_ns"] / v["instances"] / 1e3, 2)}
    out["_total_ms"] = round(total / 1e6, 2)
    if len(sys.argv) > len(files):
        with open(sys.argv[len(files)], "w") as f:
            json.dump({"groups": out, "detail": detail}, f, indent=1)
            print("WROTE", sys.argv[len(files)])

if __name__ == "__main__":
    main()
