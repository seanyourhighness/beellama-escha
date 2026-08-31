#!/usr/bin/env python3
"""Summarize the nsys sqlite: which CUDA/host calls happened and what ran long."""
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1] if len(sys.argv) > 1 else "/tmp/prof.sqlite")
cur = con.cursor()
tabs = [r[0] for r in cur.execute("select name from sqlite_master where type='table'").fetchall()]
print("tables:", tabs)

for t in tabs:
    up = t.upper()
    if "KERNEL" in up or "RUNTIME" in up or "API" in up or "SYNC" in up:
        try:
            n = cur.execute(f"select count(*) from {t}").fetchone()[0]
        except Exception:
            continue
        cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})").fetchall()]
        print(f"\n== {t} rows={n}")
        print("  cols:", cols[:18])

# top durations across runtime/api tables
for t in tabs:
    up = t.upper()
    if "RUNTIME" in up or "API" in up or "SYNC" in up:
        cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})").fetchall()]
        if "start" in cols and "end" in cols:
            namecol = "name" if "name" in cols else None
            try:
                rows = cur.execute(
                    f"select {namecol + ',' if namecol else ''} (end-start) as d from {t} order by d desc limit 8"
                ).fetchall()
                print(f"\n== longest in {t}:")
                for r in rows:
                    print("   ", r)
            except Exception as e:
                print("  query failed:", e)

# decode runtime call names
print("\n== runtime call durations (top 20):")
for row in cur.execute(
    "select s.value, r.end - r.start as d from CUPTI_ACTIVITY_KIND_RUNTIME r "
    "join StringIds s on r.nameId = s.id order by d desc limit 20"
).fetchall():
    print(f"  {row[1]/1e6:9.1f} ms  {row[0]}")

print("\n== last 25 runtime calls chronologically:")
for row in cur.execute(
    "select s.value, r.start, r.end, r.end - r.start as d from CUPTI_ACTIVITY_KIND_RUNTIME r "
    "join StringIds s on r.nameId = s.id order by r.end desc limit 25"
).fetchall():
    print(f"  end={row[2]/1e9:9.3f}s dur={row[3]/1e6:9.1f}ms  {row[0]}")

for t in ["CUDA_GRAPH_EVENTS", "SCHED_EVENTS", "COMPOSITE_EVENTS"]:
    try:
        cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})").fetchall()]
        rows = cur.execute(f"select * from {t} limit 8").fetchall()
        n = cur.execute(f"select count(*) from {t}").fetchone()[0]
        print(f"\n== {t} rows={n} cols={cols}")
        for r in rows:
            print("   ", r)
    except Exception as e:
        print(t, "err", e)
