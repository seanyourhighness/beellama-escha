#!/usr/bin/env python3
"""Probe the top-1 tokens for the first N generated positions on one backend.

Usage:
  probe_top_tokens.py --backend cuda|cpu --prompt-file p2.txt --n-predict 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import os
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
MODEL_GGUF = Path(os.environ.get("ESCHA_MODEL", "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf"))
PORT = 8096
BASE = f"http://127.0.0.1:{PORT}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cuda", "cpu"], required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--n-predict", type=int, default=8)
    ap.add_argument("--n-probs", type=int, default=5)
    ap.add_argument("--model", default=str(MODEL_GGUF),
                    help="GGUF to probe (defaults to compact Escha × LowGPU)")
    ap.add_argument("--skip-server", action="store_true")
    args = ap.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    log = Path(f"/tmp/probe-{args.backend}.log")
    proc = None
    if not args.skip_server:
        cmd = [
            str(REPO / f"build-{args.backend}/bin/llama-server"),
            "-m", args.model, "-ngl", "99", "-c", "4096",
            "--port", str(PORT), "--host", "127.0.0.1",
            "--temp", "0", "--seed", "42", "--skip-chat-parsing",
        ]
        log_fh = open(log, "wb")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        deadline = time.time() + 900
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}, see {log}")
            try:
                if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(3)
        else:
            raise RuntimeError("server did not become ready")

    try:
        r = requests.post(f"{BASE}/completion", json={
            "prompt": prompt,
            "n_predict": args.n_predict,
            "temperature": 0,
            "seed": 42,
            "n_probs": args.n_probs,
            "cache_prompt": False,
            "stream": False,
        }, timeout=1800)
        r.raise_for_status()
        data = r.json()
        out = {
            "backend": args.backend,
            "content": data["content"],
            "completion_probabilities": data.get("completion_probabilities"),
            "timings": data.get("timings"),
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
