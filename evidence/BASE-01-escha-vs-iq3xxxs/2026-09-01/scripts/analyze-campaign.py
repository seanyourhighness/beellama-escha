#!/usr/bin/env python3
"""BASE-01 Phase 2: analyze matched campaign results (block-aware).

Loads trial JSONs from noise-run/ across one or two 9-pair blocks
(filenames p<#>t<1|2>-<A|B>[-<block>].json). Computes per-arm stats (sample SD),
paired-log geometric ratio with correct df, and the preregistered CV>2%
contingency: if CV > 2%, a second block is required; if the combined 95% CI on
G spans 1.0, the comparison is declared INCONCLUSIVE.
Throughput convention: 2048 / measured_prompt_seconds (llama-bench avg_ns).
"""
import json, glob, os, sys, math, statistics
import numpy as np

# Two-sided 95% t critical values (no scipy available); interpolated for df>30.
_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
           7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
           13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
           19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
           25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}

def t_crit(df):
    if df <= 0:
        return float("nan")
    if df <= 30:
        return _T_CRIT[int(df)]
    return 1.96 + (2.042 - 1.96) * 30 / df  # smooth approach to z=1.96

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/CODEX WORKSPACE/beellama-escha/evidence/BASE-01-escha-vs-iq3xxxs/2026-09-01/bench"
NOISE = os.path.join(OUT, "noise-run")

def load_trial(path):
    d = json.load(open(path))
    r = d[0]
    return {
        "tokps": r["avg_ts"],
        "ns": r["avg_ns"],
        "median_ns": r.get("median_ns", r["avg_ns"]),
        "samples_ns": r.get("samples_ns", [r["avg_ns"]]),
        "n_prompt": r["n_prompt"],
        "n_gen": r["n_gen"],
        "build_commit": r.get("build_commit"),
    }

# filename: p<#>t<1|2>-<A|B>[-<block>].json  (block suffix only when block != 1)
trials = []  # (pair, pos, arm, block, data)
for f in sorted(glob.glob(os.path.join(NOISE, "p*t*.json"))):
    base = os.path.basename(f)[:-5]
    parts = base.split("-")
    tag, arm = parts[0], parts[1]
    if arm not in ("A", "B"):
        continue
    block = 1
    if len(parts) > 2 and parts[2].isdigit():
        block = int(parts[2])
    p = int(tag[1:].split("t")[0]); t = int(tag.split("t")[1])
    trials.append((p, t, arm, block, load_trial(f)))

def items_for(block=None, arm=None):
    res = []
    for p, t, a, b, d in trials:
        if block is not None and b != block:
            continue
        if arm is not None and a != arm:
            continue
        res.append((p, t, a, b, d))
    return res

def stats(name, items):
    tps = [d["tokps"] for *_, d in items]
    ns = [d["ns"] for *_, d in items]
    arr = np.array(tps)
    sdt = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    sdn = np.std(np.array(ns), ddof=1) if len(ns) > 1 else 0.0
    print(f"== {name} (n={len(items)}) ==")
    print(f"  raw tokps: {[round(x,2) for x in tps]}")
    print(f"  raw ns:    {[int(x) for x in ns]}")
    print(f"  median tokps: {statistics.median(tps):.2f}  mean: {np.mean(arr):.2f}  sample-SD: {sdt:.2f}  CV%: {sdt/np.mean(arr)*100:.2f}")
    print(f"  median latency ms: {statistics.median(ns)/1e6:.3f}  mean: {np.mean(ns)/1e6:.3f}  sample-SD: {sdn/1e6:.3f}  CV%: {sdn/np.mean(ns)*100:.2f}")
    return tps, ns

def paired_analysis(label, itemsA, itemsB, df_note=""):
    """itemsA/itemsB: sorted by (block, pair) so they align pairwise."""
    nsA = [d["ns"] for *_, d in itemsA]
    nsB = [d["ns"] for *_, d in itemsB]
    n = min(len(nsA), len(nsB))
    nsA, nsB = nsA[:n], nsB[:n]
    ratios = [a / b for a, b in zip(nsA, nsB)]
    lnr = [math.log(r) for r in ratios]
    mean_ln = sum(lnr) / len(lnr)
    G = math.exp(mean_ln)
    sd = statistics.stdev(lnr) if len(lnr) > 1 else 0.0
    se = sd / math.sqrt(len(lnr))
    df = len(lnr) - 1
    tcrit = t_crit(df)
    lo, hi = math.exp(mean_ln - tcrit * se), math.exp(mean_ln + tcrit * se)
    winsB = sum(1 for r in ratios if r > 1)
    inconclusive = lo <= 1.0 <= hi
    print(f"== {label} ({df_note}) ==")
    print(f"  ratios: {[round(r,4) for r in ratios]}")
    print(f"  paired-log G: {G:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (df={df}, t={tcrit:.3f})")
    print(f"  pairwise B-faster: {winsB}/{len(ratios)}")
    print(f"  INCONCLUSIVE (CI spans 1.0): {inconclusive}")
    return {"n": n, "G": G, "ci": [lo, hi], "winsB": winsB, "inconclusive": inconclusive, "df": df}

blocks = sorted({b for *_, b, _ in [(p, t, a, b, d) for p, t, a, b, d in trials]})
print("blocks found:", blocks)

# ---- block 1 analysis ----
A1 = sorted([x for x in items_for(block=1, arm="A")], key=lambda x: x[0])
B1 = sorted([x for x in items_for(block=1, arm="B")], key=lambda x: x[0])
if A1 and B1:
    stats("ARM A (ESCHA) block1", A1)
    stats("ARM B (IQ3) block1", B1)
    cvA = np.std([d["ns"] for *_, d in A1], ddof=1) / np.mean([d["ns"] for *_, d in A1]) * 100
    cvB = np.std([d["ns"] for *_, d in B1], ddof=1) / np.mean([d["ns"] for *_, d in B1]) * 100
    print(f"  latency sample-CV% block1: A={cvA:.2f} B={cvB:.2f}  -> CV contingency: {'TRIGGERED (run second 9-pair block with BLOCK=2)' if max(cvA, cvB) > 2.0 else 'not triggered'}")
    cv_triggered = max(cvA, cvB) > 2.0
    r1 = paired_analysis("paired (A latency / B latency) block1", A1, B1, "block 1, df=8")
else:
    cv_triggered = False

# ---- combined analysis (all blocks, pairwise aligned by (block, pair)) ----
Aall = sorted(items_for(arm="A"), key=lambda x: (x[3], x[0]))
Ball = sorted(items_for(arm="B"), key=lambda x: (x[3], x[0]))

def key_set(items):
    return {(b, p) for p, t, a, b, d in items}

# Sol REVISE (round 3): enforce completeness before a definitive combined decision.
complete = True
for blk in blocks:
    kA = key_set([x for x in Aall if x[3] == blk])
    kB = key_set([x for x in Ball if x[3] == blk])
    if len(kA) != 9 or len(kB) != 9 or kA != kB:
        complete = False
        print(f"  INCOMPLETE block {blk}: A={len(kA)}/9 B={len(kB)}/9 matched={kA==kB}")
# Sol REVISE (round 3): if the CV contingency was triggered on block 1, block 2
# MUST be present and complete; a definitive decision requires it.
if cv_triggered and 2 not in blocks:
    complete = False
    print("  WARNING: block-1 CV > 2% triggered the second-block contingency but block 2 is absent -> combined DECISION forced to INCONCLUSIVE")
if not complete:
    print("  WARNING: triggered/declared blocks incomplete -> combined DECISION forced to INCONCLUSIVE (no definitive claim)")

if Aall and Ball:
    stats("ARM A (ESCHA) combined", Aall)
    stats("ARM B (IQ3) combined", Ball)
    rall = paired_analysis("paired (A latency / B latency) combined", Aall, Ball,
                           f"all blocks (n={min(len(Aall), len(Ball))})")
    if not complete:
        rall["inconclusive"] = True
    nsA = [d["ns"] for *_, d in Aall][:rall["n"]]
    nsB = [d["ns"] for *_, d in Ball][:rall["n"]]
    medA, medB = statistics.median(nsA), statistics.median(nsB)
    print(f"  median latency gap (A-B): {(medA-medB)/1e6:.3f} ms")
    print(f"  median tokps: A={2048/(medA/1e9):.1f} B={2048/(medB/1e9):.1f}  gap (B-A)={2048/(medB/1e9)-2048/(medA/1e9):.1f} tok/s")
    print(f"  total latency gap (A-B): {(medA-medB)/1e6:.3f} ms (of ~300 ms target)")

    # historical comparison
    tpsA_med = 2048 / (medA / 1e9); tpsB_med = 2048 / (medB / 1e9)
    print("== historical comparison ==")
    print(f"  ESCHA historical ~2356 tok/s: {'REPRODUCED' if abs(tpsA_med-2356) < 100 else 'NOT REPRODUCED'} (median {tpsA_med:.1f})")
    print(f"  IQ3 historical 3339 tok/s: {'REPRODUCED' if abs(tpsB_med-3339) < 100 else 'NOT REPRODUCED'} (median {tpsB_med:.1f})")
    print(f"  IQ3 3600 claim: {'CONFIRMED' if tpsB_med >= 3500 else 'NOT SUPPORTED'} (median {tpsB_med:.1f})")
    print(f"  DECISION: {'INCONCLUSIVE (CI spans 1.0 or incomplete blocks)' if rall['inconclusive'] else ('B faster than A' if rall['G'] > 1.0 else 'A not slower than B')}")
else:
    print("  DECISION: NO COMBINED DECISION (blocks absent)")
