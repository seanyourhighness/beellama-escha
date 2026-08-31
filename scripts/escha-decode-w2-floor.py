#!/usr/bin/env python3
"""Compute the per-family decode W2 kernel-time floor from an ESCHA_PROFILE
decode capture (rows=1, gen=1).  The min event-elapsed per family approaches
the uninstrumented kernel duration; the mean includes launch/sync gaps."""

import re
import sys
from collections import defaultdict

pat = re.compile(
    r"^ESCHA_PROFILE k=(\d+) ic=(\d+) oc=(\d+) rows=(\d+) gen=(\d+) route=(\S+) "
    r"total_ms=([\d.]+) rotate_ms=([\d.]+) matmul_ms=([\d.]+) epilogue_ms=([\d.]+)$"
)


def main() -> int:
    fam: dict = defaultdict(lambda: {"n": 0, "mm_min": 9e9, "mm_sum": 0.0,
                                     "rot_min": 9e9, "epi_min": 9e9})
    with open(sys.argv[1], errors="replace") as fh:
        for line in fh:
            m = pat.match(line.rstrip("\n"))
            if m and m.group(5) == "1":
                key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                mm = float(m.group(9)); rot = float(m.group(8)); epi = float(m.group(10))
                f = fam[key]
                f["n"] += 1
                f["mm_min"] = min(f["mm_min"], mm)
                f["mm_sum"] += mm
                f["rot_min"] = min(f["rot_min"], rot)
                f["epi_min"] = min(f["epi_min"], epi)

    print("family n mm_min rot_min epi_min mm_avg")
    for key in sorted(fam):
        f = fam[key]
        print(f"{key} n={f['n']} mm_min={f['mm_min']:.4f} "
              f"rot_min={f['rot_min']:.4f} epi_min={f['epi_min']:.4f} "
              f"mm_avg={f['mm_sum']/f['n']:.4f}")

    n_steps = sum(f["n"] for f in fam.values()) // 400
    tot = sum((f["n"]/n_steps) * (f["mm_min"] + f["rot_min"] + f["epi_min"])
              for f in fam.values())
    print(f"steps_in_capture={n_steps}")
    print(f"per-step lower bound (min kernel durations summed): {tot:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
