#!/usr/bin/env python3
"""Build the P-ARCH-13 K2 family table from per-run aggregate JSONs.

Usage:
    escha-parch13-table.py --baseline <b.agg.json>... --profile <p.agg.json>...

Each input is produced by escha-profile-aggregate.py. Baseline files are the
128x128 control captures, profile files are the K2 128x64 captures. Per-family
medians across runs are compared against the retained Escha trace anchors
(verified independently from the P-ARCH-06 trace by escha_trace_k2_audit.py).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

ESCHA_ANCHORS = {
    "2|5120|1024": 4.270,
    "2|5120|6144": 23.220,
    "2|5120|10240": 36.338,
    "2|5120|12288": 14.896,
    "2|5120|17408": 77.704,
    "2|6144|5120": 29.297,
}
FAMILY_ORDER = ["2|5120|1024", "2|5120|6144", "2|5120|10240",
                "2|5120|12288", "2|5120|17408", "2|6144|5120"]
HISTORICAL_K2_RESIDUAL = 257.137  # P-ARCH-12 (442.862 - 185.725)


def medians(paths: list[str]) -> dict[str, dict]:
    fam: dict[str, dict] = {}
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        for key, f in d["families"].items():
            if key not in fam:
                fam[key] = {"total": [], "matmul": [], "rotate": [], "epilogue": []}
            fam[key]["total"].append(f["total_ms"])
            fam[key]["matmul"].append(f["matmul_ms"])
            fam[key]["rotate"].append(f["rotate_ms"])
            fam[key]["epilogue"].append(f["epilogue_ms"])
    return {k: {m: statistics.median(vs) for m, vs in f.items()} for k, f in fam.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True, metavar="AGG_JSON",
                    help="128x128 control aggregate JSONs")
    ap.add_argument("--profile", nargs="+", required=True, metavar="AGG_JSON",
                    help="K2 128x64 experiment aggregate JSONs")
    args = ap.parse_args()

    base = medians(args.baseline)
    prof = medians(args.profile)

    print(f"{'K2 family':14s} {'128x128 ms':>11s} {'128x64 ms':>10s} "
          f"{'Escha ms':>10s} {'Delta rem':>10s} {'MMA-body':>12s} {'Rot+Epi':>12s}")
    print("-" * 78)

    t_base = t_prof = 0.0
    m_base = m_prof = 0.0
    r_base = r_prof = 0.0
    for key in FAMILY_ORDER:
        b, p = base[key], prof[key]
        e = ESCHA_ANCHORS[key]
        delta = b["total"] - p["total"]
        t_base += b["total"]
        t_prof += p["total"]
        m_base += b["matmul"]
        m_prof += p["matmul"]
        r_base += b["rotate"] + b["epilogue"]
        r_prof += p["rotate"] + p["epilogue"]
        print(f"{key:14s} {b['total']:11.3f} {p['total']:10.3f} {e:10.3f} "
              f"{delta:10.3f} {p['matmul'] - b['matmul']:+12.3f} "
              f"{(p['rotate'] + p['epilogue']) - (b['rotate'] + b['epilogue']):+12.3f}")

    e_total = sum(ESCHA_ANCHORS.values())
    print("-" * 78)
    print(f"{'K2 total':14s} {t_base:11.3f} {t_prof:10.3f} {e_total:10.3f} "
          f"{t_base - t_prof:10.3f} {m_prof - m_base:+12.3f} {r_prof - r_base:+12.3f}")

    old_res = t_base - e_total
    new_res = t_prof - e_total
    print()
    print(f"same-instrumentation control residual (128x128 - Escha): {old_res:.3f} ms")
    print(f"experiment residual (128x64 - Escha):                   {new_res:.3f} ms")
    print(f"geometry explained (same-instrumentation):              {old_res - new_res:.3f} ms "
          f"({(old_res - new_res) / old_res * 100:.2f}% of control residual)")
    print(f"geometry explained vs historical 257.137 ms:            {(old_res - new_res):.3f} ms "
          f"({(old_res - new_res) / HISTORICAL_K2_RESIDUAL * 100:.2f}%)")
    print(f"MMA-body residual removed:                              {m_base - m_prof:.3f} ms "
          f"({(m_base - m_prof) / m_base * 100:.2f}% of control MMA body)")
    print(f"rotate+epilogue residual removed:                       {r_base - r_prof:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
