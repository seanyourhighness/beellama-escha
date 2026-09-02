#!/usr/bin/env python3
"""EXP-08 campaign analysis: control vs candidate medians + paired stats."""
import json, glob, os, sys, math, statistics
import numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/CODEX WORKSPACE/beellama-escha/evidence/EXP-08-fusedfinalize/2026-09-01/bench"
NOISE = os.path.join(OUT, "noise-run")
def load(fn):
    d = json.load(open(fn))[0]
    return d["avg_ts"], d["avg_ns"]
trials = {}
for f in sorted(glob.glob(os.path.join(NOISE, "p*t*.json"))):
    base = os.path.basename(f)[:-5]
    tag, arm = base.split("-")
    trials[(tag, arm)] = load(f)
def get(arm):
    items = [(t, v) for (t, a), v in trials.items() if a == arm]
    items.sort(key=lambda x: int(x[0][1:].split("t")[0]))
    return items
ctl = get("control"); cand = get("candidate")
def stats(name, items):
    tps = [v[0] for _, v in items]; ns = [v[1] for _, v in items]
    a = np.array(tps)
    sd = np.std(a, ddof=1) if len(a) > 1 else 0
    print(f"== {name} (n={len(items)}) ==")
    print(f"  raw tokps: {[round(x,2) for x in tps]}")
    print(f"  median tokps: {statistics.median(tps):.2f} mean {np.mean(a):.2f} sample-SD {sd:.2f} CV% {sd/np.mean(a)*100:.2f}")
    print(f"  median ns: {statistics.median(ns)/1e6:.3f} ms")
    return tps, ns
tC, nC = stats("CONTROL", ctl)
tK, nK = stats("CANDIDATE", cand)
ratios = [k/c for k, c in zip(nK, nC)]  # candidate faster if <1 (latency)
lnr = [math.log(k/c) for k, c in zip(nK, nC)]
mean_ln = sum(lnr)/len(lnr)
G = math.exp(mean_ln)  # latency ratio candidate/control, <1 = faster
sd = statistics.stdev(lnr); se = sd/math.sqrt(len(lnr)); df = len(lnr)-1
tcrit = 2.306 if df == 8 else 2.110 if df == 17 else 2.0
lo, hi = math.exp(mean_ln - tcrit*se), math.exp(mean_ln + tcrit*se)
wins = sum(1 for r in ratios if r < 1)
print(f"== paired ==")
print(f"  latency ratios (cand/ctl): {[round(r,4) for r in ratios]}")
print(f"  geometric ratio G={G:.4f} 95% CI [{lo:.4f},{hi:.4f}] (latency; <1 = candidate faster)")
print(f"  candidate-faster count: {wins}/{len(ratios)}")
medC, medK = statistics.median(nC), statistics.median(nK)
print(f"  median: control {2048/(medC/1e9):.1f} tok/s, candidate {2048/(medK/1e9):.1f} tok/s")
print(f"  gain: {(medK-medC)/medC*100:+.2f}% latency; {(2048/(medK/1e9)-2048/(medC/1e9))/(2048/(medC/1e9))*100:+.2f}% tok/s")
