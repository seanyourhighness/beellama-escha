#!/usr/bin/env python3
"""Aggregate ESCHA_PROFILE records from a llama-bench stderr capture.

Usage:
    escha-profile-aggregate.py <stderr-file> [--json-out <path>]

Asserts a full 3,200-record capture (1,600 warm-up + 1,600 measured) and
aggregates only the last 1,600 records by (k, ic, oc), keeping rotate, MMA
body (matmul), and epilogue separate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

LINE = re.compile(
    r"^ESCHA_PROFILE k=(\d+) ic=(\d+) oc=(\d+) rows=(\d+) gen=(\d+) "
    r"route=(\S+) total_ms=([\d.]+) rotate_ms=([\d.]+) matmul_ms=([\d.]+) "
    r"epilogue_ms=([\d.]+)$"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stderr")
    ap.add_argument("--total", type=int, default=3200)
    ap.add_argument("--measured", type=int, default=1600)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    recs: list[dict] = []
    with open(args.stderr, "r", errors="replace") as fh:
        for line in fh:
            m = LINE.match(line.rstrip("\n"))
            if m:
                recs.append({
                    "k": int(m[1]), "ic": int(m[2]), "oc": int(m[3]),
                    "rows": int(m[4]), "gen": int(m[5]), "route": m[6],
                    "total": float(m[7]), "rotate": float(m[8]),
                    "matmul": float(m[9]), "epilogue": float(m[10]),
                })

    if len(recs) != args.total:
        print(f"ERROR: expected {args.total} ESCHA_PROFILE records, got {len(recs)}",
              file=sys.stderr)
        return 2

    measured = recs[-args.measured:]
    fam: dict[tuple[int, int, int], dict] = defaultdict(lambda: {
        "count": 0, "total": 0.0, "rotate": 0.0, "matmul": 0.0,
        "epilogue": 0.0, "routes": set(),
    })
    for r in measured:
        key = (r["k"], r["ic"], r["oc"])
        f = fam[key]
        f["count"] += 1
        f["total"] += r["total"]
        f["rotate"] += r["rotate"]
        f["matmul"] += r["matmul"]
        f["epilogue"] += r["epilogue"]
        f["routes"].add(r["route"])

    out: dict = {"records_total": len(recs), "records_measured": len(measured),
                 "families": {}, "aggregate": {}}
    for key in sorted(fam):
        f = fam[key]
        out["families"][f"{key[0]}|{key[1]}|{key[2]}"] = {
            "count": f["count"],
            "total_ms": round(f["total"], 3),
            "rotate_ms": round(f["rotate"], 3),
            "matmul_ms": round(f["matmul"], 3),
            "epilogue_ms": round(f["epilogue"], 3),
            "routes": sorted(f["routes"]),
        }

    agg = {"count": 0, "total": 0.0, "rotate": 0.0, "matmul": 0.0, "epilogue": 0.0}
    for f in fam.values():
        agg["count"] += f["count"]
        agg["total"] += f["total"]
        agg["rotate"] += f["rotate"]
        agg["matmul"] += f["matmul"]
        agg["epilogue"] += f["epilogue"]
    out["aggregate"] = {
        k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items()
    }

    print(json.dumps(out, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
