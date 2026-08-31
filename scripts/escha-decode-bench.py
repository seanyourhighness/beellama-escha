#!/usr/bin/env python3
"""Matched decode baseline client for D-ARCH-01.

Runs `concurrency` simultaneous streaming chat-completions requests against an
OpenAI-compatible endpoint (llama-server or Escha/SGLang), each with the same
fixed prompt and max_tokens, temperature 0. Reports aggregate and per-stream
tok/s, step latency, first-token latency, and median GPU util/power.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def gpu_sampler(samples: list[list[float]], stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            row = subprocess.check_output(
                ["/usr/lib/wsl/lib/nvidia-smi",
                 "--query-gpu=utilization.gpu,power.draw,memory.used",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=3).strip()
            samples.append([float(p.strip()) for p in row.split(",")])
        except Exception:
            pass
        time.sleep(0.25)


def one_stream(base: str, model: str, prompt: str, tokens: int) -> dict:
    t0 = time.monotonic()
    first = None
    last = None
    got = 0
    r = requests.post(
        f"{base}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": tokens, "stream": True},
        timeout=900, stream=True)
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        if line[6:] == "[DONE]":
            break
        ev = json.loads(line[6:])
        ch = ev.get("choices") or []
        if not ch:
            continue
        delta = ch[0].get("delta") or {}
        text = delta.get("content") or delta.get("reasoning_content")
        if text:
            now = time.monotonic()
            if first is None:
                first = now
            last = now
            got += 1
    wall = time.monotonic() - t0
    return {
        "tokens": got,
        "wall_s": wall,
        "tok_s": got / wall if wall > 0 else 0.0,
        "ttft_s": (first - t0) if first else None,
        "step_ms": ((last - first) / max(got - 1, 1)) * 1000 if got > 1 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8096")
    ap.add_argument("--model", default="bee")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    if args.prompt_file:
        prompt = open(args.prompt_file, encoding="utf-8").read()
    else:
        prompt = ("Write a detailed multi-paragraph essay about the history of "
                  "computing from the 1940s to the present day.")

    report: dict = {"prompt_len": len(prompt), "max_tokens": args.tokens,
                    "concurrency_runs": {}}
    for c in args.concurrency:
        rows = []
        for run in range(args.runs):
            samples: list[list[float]] = []
            stop = threading.Event()
            th = threading.Thread(target=gpu_sampler, args=(samples, stop),
                                  daemon=True)
            th.start()
            with ThreadPoolExecutor(max_workers=c) as pool:
                futs = [pool.submit(one_stream, args.base, args.model, prompt,
                                    args.tokens) for _ in range(c)]
                results = [f.result() for f in futs]
            stop.set()
            th.join(timeout=3)
            agg_tok = sum(r["tokens"] for r in results)
            agg_wall = max(r["wall_s"] for r in results)
            row = {
                "run": run + 1,
                "streams": c,
                "aggregate_tok_s": agg_tok / agg_wall if agg_wall > 0 else 0.0,
                "per_stream": results,
                "median_step_ms": statistics.median(
                    [r["step_ms"] for r in results if r["step_ms"] is not None]),
                "median_ttft_s": statistics.median(
                    [r["ttft_s"] for r in results if r["ttft_s"] is not None]),
                "gpu": {
                    "util_median": statistics.median([s[0] for s in samples]) if samples else None,
                    "power_median": statistics.median([s[1] for s in samples]) if samples else None,
                    "mem_median": statistics.median([s[2] for s in samples]) if samples else None,
                },
            }
            rows.append(row)
            print(f"c={c} run={run + 1}: aggregate {row['aggregate_tok_s']:.2f} tok/s "
                  f"step {row['median_step_ms']:.2f} ms "
                  f"gpu {row['gpu']['util_median']}% / {row['gpu']['power_median']}W")
        report["concurrency_runs"][str(c)] = rows
        med = statistics.median(r["aggregate_tok_s"] for r in rows)
        print(f"c={c}: median aggregate {med:.2f} tok/s")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
