#!/usr/bin/env python3
"""Classify an Escha decode torch-profiler trace into operator buckets.

Usage: escha_decode_trace_audit.py <trace.json.gz>

Prints per-bucket event count, total ms, and mean per-event ms, plus the
first few distinct kernel names per bucket so the classification is auditable.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict


def bucket(name: str) -> str:
    n = name
    if "escham_code_gemm" in n or "escha" in n.lower():
        return "escha-w2"
    if re.search(r"mamba|selective_state|ssm|conv1d|gdn|linear_attn|chunk", n, re.I):
        return "gdn-linear-attn"
    if re.search(r"flash|fwd_kernel|attention|attn|mha", n, re.I):
        return "attention"
    if re.search(r"rms_norm|rmsnorm|layer_norm|layernorm|norm", n, re.I):
        return "norm"
    if re.search(r"gemv|gemm|matmul|mm\b|lowgpu|logits|lm_head|head", n, re.I):
        return "head-gemm"
    if re.search(r"copy|cast|fill|cat\b|index|reshape|view|permute|transpose|to\b|elementwise|zero_", n, re.I):
        return "copy-convert"
    if re.search(r"rope|rotary", n, re.I):
        return "rope"
    return "other"


def main() -> int:
    path = sys.argv[1]
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    events = doc["traceEvents"]
    kerns = [e for e in events if e.get("cat") == "kernel"]
    buckets: dict[str, dict] = defaultdict(lambda: {"n": 0, "us": 0.0, "names": set()})
    for e in kerns:
        name = e.get("name", "")
        b = bucket(name)
        buckets[b]["n"] += 1
        buckets[b]["us"] += float(e.get("dur", 0.0))
        buckets[b]["names"].add(name)

    print(f"total kernels: {len(kerns)}")
    print(f"{'bucket':18s} {'events':>7s} {'ms':>10s} {'share':>7s}")
    print("-" * 50)
    total_us = sum(b["us"] for b in buckets.values())
    for b in sorted(buckets, key=lambda k: -buckets[k]["us"]):
        d = buckets[b]
        print(f"{b:18s} {d['n']:7d} {d['us']/1e3:10.2f} {d['us']/total_us*100:6.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':18s} {len(kerns):7d} {total_us/1e3:10.2f} 100.0%")
    print()
    for b in sorted(buckets, key=lambda k: -buckets[k]["us"]):
        d = buckets[b]
        names = sorted(d["names"])[:4]
        print(f"== {b} sample names:")
        for n in names:
            print("   ", n[:150])
    return 0


if __name__ == "__main__":
    sys.exit(main())
