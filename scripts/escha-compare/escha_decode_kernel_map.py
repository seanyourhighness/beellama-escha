#!/usr/bin/env python3
"""Map the Escha decode W2 kernels in a torch-profiler trace: per-kernel-name
event counts, total/mean durations, and a sample launch geometry."""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict


def main() -> int:
    path = sys.argv[1]
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    kerns = [e for e in doc["traceEvents"] if e.get("cat") == "kernel"]
    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "us": 0.0})
    for e in kerns:
        name = e.get("name", "")
        if "escham" in name:
            short = re.sub(r"^void ", "", name)
            short = re.sub(r"\(.*$", "()", short)
            agg[short]["n"] += 1
            agg[short]["us"] += float(e.get("dur", 0.0))

    print("escham decode kernels (trace spans 12 decode steps):")
    for name in sorted(agg, key=lambda k: -agg[k]["us"]):
        d = agg[name]
        print(f"{d['n']:6d} {d['us']/1e3:9.2f} ms  {name[:110]}")

    for e in kerns:
        if "gemv_bw" in e.get("name", ""):
            print("sample gemv_bw args:", json.dumps(e.get("args", {}))[:400])
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
