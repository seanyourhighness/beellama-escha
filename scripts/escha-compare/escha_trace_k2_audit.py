#!/usr/bin/env python3
"""Audit the retained Escha K2 code-GEMM family totals from the P-ARCH-06
prefill trace.

Groups every escham_code_gemm kernel event by kernel-name template and grid y,
then prints per-group counts and durations (ms) so the retained family table
(P-ARCH-12) can be checked independently.
"""

from __future__ import annotations

import gzip
import json
import re
import os
import sys
from collections import defaultdict

TRACE = (os.environ.get("ESCHA_TRACE_DIR", "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-06/2026-08-29/") +
         "trace-001/escha-prefill_batch1_input2048_output1_prefill.trace.json.gz")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else TRACE
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    events = doc["traceEvents"]

    groups: dict[tuple[str, int, int], dict] = defaultdict(lambda: {
        "count": 0, "dur_us": 0.0, "grids": set(), "blocks": set(),
    })
    for e in events:
        name = e.get("name", "")
        if "escham_code_gemm" not in name:
            continue
        grid = e.get("args", {}).get("grid", [])
        block = e.get("args", {}).get("block", [])
        gx = int(grid[0]) if len(grid) > 0 else -1
        gy = int(grid[1]) if len(grid) > 1 else -1
        key = (name, gx, gy)
        g = groups[key]
        g["count"] += 1
        g["dur_us"] += float(e.get("dur", 0.0))
        g["grids"].add(tuple(grid))
        g["blocks"].add(tuple(block))

    print(f"{'kernel template':46s} {'gx':>4s} {'gy':>4s} {'n':>5s} {'ms':>10s}")
    print("-" * 78)
    totals = defaultdict(lambda: {"count": 0, "ms": 0.0})
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        name, gx, gy = key
        g = groups[key]
        ms = g["dur_us"] / 1e3
        short = re.sub(r"^void ", "", name)[:46]
        print(f"{short:46s} {gx:4d} {gy:4d} {g['count']:5d} {ms:10.3f}")
        tkey = ("k2" if re.search(r"<1,\s*2,", name) else "k3", gy)
        totals[tkey]["count"] += g["count"]
        totals[tkey]["ms"] += ms

    print("-" * 78)
    for (kind, gy), t in sorted(totals.items()):
        print(f"{kind} gy={gy}: {t['count']} events, {t['ms']:.3f} ms")

    k2 = sum(t["ms"] for (kind, _), t in totals.items() if kind == "k2")
    k2n = sum(t["count"] for (kind, _), t in totals.items() if kind == "k2")
    print(f"K2 grand total: {k2n} events, {k2:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
