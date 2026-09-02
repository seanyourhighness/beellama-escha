#!/usr/bin/env python3
"""BASE-01 Phase 4: per-shape family table (ESCHA vs IQ3) at each M.

Reads:
- A-escha-M<M>.profile.stderr : ESCHA_PROFILE lines (rotate/matmul/epilogue per call)
- B-iq3-M<M>.op_profile.stderr : GGML_OP_PROFILE lines (per mul_mat call)

Group by projection shape (ic, oc, k or by tensor name for IQ3). Reports
aggregate + median per call + gap share. Closure: sum of families vs the
graphs-off whole-run total at that M (from *.total.json).
"""
import sys, re, json, os, statistics
from collections import defaultdict

def escha_rows(path):
    pat = re.compile(
        r"ESCHA_PROFILE k=(\d+) ic=(\d+) oc=(\d+) rows=(\d+) gen=(\d+) route=(\S+) "
        r"total_ms=([\d.]+) rotate_ms=([\d.]+) matmul_ms=([\d.]+) epilogue_ms=([\d.]+)")
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append({"ic": int(m.group(2)), "oc": int(m.group(3)), "k": int(m.group(1)),
                             "rows": int(m.group(4)), "total_ms": float(m.group(7)),
                             "rotate_ms": float(m.group(8)), "matmul_ms": float(m.group(9)),
                             "epilogue_ms": float(m.group(10))})
    return rows

def iq3_rows(path):
    pat = re.compile(
        r"GGML_OP_PROFILE op=mul_mat name=(\S+) ne=(\d+)x(\d+)x(\d+)x(\d+) "
        r"type_src0=(\d+) rows=(\d+) cols=(\d+) stream=(\S+) total_ms=([\d.]+) host_ms=([\d.]+)")
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            if float(m.group(11)) <= 0.0:
                continue  # reserve/no-op pass
            rows.append({"name": m.group(1), "ne0": int(m.group(2)), "ne1": int(m.group(3)),
                         "ne2": int(m.group(4)), "ne3": int(m.group(5)),
                         "type_src0": int(m.group(6)), "rows": int(m.group(7)),
                         "cols": int(m.group(8)), "total_ms": float(m.group(10))})
    return rows

def fam(ic, oc, k):
    if ic == 5120 and oc == 17408 and k == 3: return "ffn_up"
    if ic == 17408 and oc == 5120 and k == 3: return "ffn_down"
    if ic == 5120 and oc == 17408 and k == 2: return "ffn_gate"
    if ic == 5120 and oc == 6144 and k == 2: return "attn_gate"
    if ic == 5120 and oc == 10240 and k == 2: return "attn_qkv"
    if ic == 5120 and oc == 12288 and k == 2: return "attn_q"
    if ic == 5120 and oc == 1024 and k == 2: return "attn_kv"
    if ic == 6144 and oc == 5120 and k == 2: return "ssm_out_OR_attn_output"
    return f"other_{ic}x{oc}_k{k}"

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

def total_ms(path):
    d = json.load(open(path))[0]
    return d["avg_ns"] / 1e6

def main():
    base = sys.argv[1]
    Ms = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [128, 512, 1024, 2048, 4096]
    allout = {}
    for M in Ms:
        er = escha_rows(f"{base}/A-escha-M{M}.profile.stderr")
        ir = iq3_rows(f"{base}/B-iq3-M{M}.op_profile.stderr")
        tA = total_ms(f"{base}/A-escha-M{M}.total.json")
        tB = total_ms(f"{base}/B-iq3-M{M}.total.json")
        print(f"===== M={M}: whole-run graphs-off A={tA:.1f} ms B={tB:.1f} ms gap={tA-tB:.1f} ms =====")
        ea = defaultdict(list); ib = defaultdict(list)
        for r in er: ea[fam(r["ic"], r["oc"], r["k"])].append(r)
        for r in ir: ib[iq3_fam(r["name"], ne0=r["ne0"], cols=r["cols"])].append(r)
        # shape-level: match families that exist in both
        fams = sorted(set(ea) | set(ib))
        rows_out = []
        for f in fams:
            a = ea.get(f, []); b = ib.get(f, [])
            ta = sum(r["total_ms"] for r in a); tb = sum(r["total_ms"] for r in b)
            meda = statistics.median([r["total_ms"] for r in a]) if a else float("nan")
            medb = statistics.median([r["total_ms"] for r in b]) if b else float("nan")
            gap = ta - tb
            na = len(a); nb = len(b)
            shape = f"{a[0]['ic']}x{a[0]['oc']} k{a[0]['k']}" if a else (f"{b[0]['cols']}x{b[0]['ne0']}" if b else "?")
            print(f"  {f:28s} shape={shape:16s} calls A={na:4d} B={nb:4d} "
                  f"A={ta:9.1f} B={tb:9.1f} gap={gap:+9.1f} medA={meda:.4f} medB={medb:.4f}")
            rows_out.append({"family": f, "shape": shape, "calls_A": na, "calls_B": nb,
                             "A_ms": round(ta, 2), "B_ms": round(tb, 2), "gap_ms": round(gap, 2),
                             "medA_ms": round(meda, 4), "medB_ms": round(medb, 4)})
        sumA = sum(x["A_ms"] for x in rows_out); sumB = sum(x["B_ms"] for x in rows_out)
        proj_gap = sumA - sumB
        print(f"  SUM projection: A={sumA:.1f} B={sumB:.1f} projection_gap={proj_gap:+.1f} "
              f"(whole-run gap {tA-tB:+.1f}; closure {proj_gap/(tA-tB)*100:.0f}%)")
        allout[M] = {"wholeA_ms": round(tA, 2), "wholeB_ms": round(tB, 2), "gap_ms": round(tA - tB, 2),
                     "sumA_ms": round(sumA, 2), "sumB_ms": round(sumB, 2), "projection_gap_ms": round(proj_gap, 2),
                     "closure_pct": round(proj_gap / (tA - tB) * 100, 1) if abs(tA - tB) > 1e-9 else None,
                     "families": rows_out}
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as f:
            json.dump(allout, f, indent=1)
            print("WROTE", sys.argv[3])

if __name__ == "__main__":
    main()
