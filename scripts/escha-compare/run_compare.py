#!/usr/bin/env python3
"""Fixed-prompt quality comparison: Beellama native Escha port vs reference SGLang (e3).

Method (mirrors results/prompts.json):
  1. Render each prompt exactly as the reference SGLang server did: the patched
     no-think chat template + deterministic long-context generator (seed 42).
  2. Serve the Beellama GGUF once (llama-server), then POST raw-prompt /completion
     requests with temperature 0 / seed 42.
  3. Compare against the stored e3 reference outputs at token level with the same
     tokenizer, and report decode speed.

Run from WSL with the escha venv python:
  <escha-venv-python> scripts/escha-compare/run_compare.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]  # <repo>/scripts/escha-compare -> repo root
PROJ = Path(os.environ.get("ESCHA_W2_PROJ",
            "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu"))  # external data dir; override via env
MODEL_GGUF = PROJ / "weights/escha-w2-lowgpu-mono-parity.gguf"
MODEL_DIR = PROJ / "weights/escha"          # tokenizer source (same vocab as the GGUF)
REFERENCE_SUITE = PROJ / "results/generation-results/e3/suite.json"
PROMPTS_JSON = PROJ / "results/prompts.json"
# A previous one-off benchmark wrote a patched template into /tmp.  Prefer it
# when present for byte-for-byte reproduction, but make the suite portable by
# falling back to the checkpoint's canonical template.
TEMPLATE_FILE = Path("/tmp/escha_nothink_escha-w2-lowgpu-mono.jinja")
# A separate verification build can be selected without altering the default
# developer build directory.
SERVER_BIN = Path(os.environ.get("ESCHA_SERVER_BIN", REPO / "build-cuda/bin/llama-server"))
PORT = 8095
BASE = f"http://127.0.0.1:{PORT}"

sys.path.insert(0, str(PROJ / "source"))
from run_experiment import long_context_text  # noqa: E402


def render_prompts(tokenizer) -> list[dict]:
    template_path = TEMPLATE_FILE if TEMPLATE_FILE.exists() else MODEL_DIR / "chat_template.jinja"
    template = template_path.read_text(encoding="utf-8")
    prompts = json.loads(PROMPTS_JSON.read_text(encoding="utf-8"))["prompts"]
    out = []
    for p in prompts:
        content = p["content"]
        if "{{LONG_CONTEXT_PLACEHOLDER}}" in content:
            content = long_context_text()
        messages = [{"role": p["role"], "content": content}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=template,
            enable_thinking=False,
            reasoning_effort="medium",
        )
        out.append({
            "id": p["id"],
            "max_new_tokens": p["max_new_tokens"],
            "rendered": rendered,
        })
    return out


def start_server(model: Path, ctx_size: int, log_path: Path) -> subprocess.Popen:
    cmd = [
        str(SERVER_BIN), "-m", str(model), "-ngl", "99",
        "-c", str(ctx_size), "--port", str(PORT), "--host", "127.0.0.1",
        "--temp", "0", "--seed", "42", "--skip-chat-parsing",
    ]
    log_fh = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited rc={proc.returncode}, see {log_path}")
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("llama-server did not become ready")


def complete(prompt: str, n_predict: int) -> dict:
    r = requests.post(f"{BASE}/completion", json={
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
        "stream": False,
    }, timeout=1800)
    r.raise_for_status()
    return r.json()


def tokenize(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def token_agreement(tokenizer, ref: str, got: str) -> dict:
    rt = tokenize(tokenizer, ref)
    gt = tokenize(tokenizer, got)
    n = min(len(rt), len(gt))
    agree = 0
    while agree < n and rt[agree] == gt[agree]:
        agree += 1
    return {
        "ref_tokens": len(rt),
        "got_tokens": len(gt),
        "compared_tokens": n,
        "prefix_agree_tokens": agree,
        "prefix_agree_rate": agree / len(rt) if rt else 1.0,
        "compared_agree_rate": agree / n if n else 1.0,
        "exact_token_match": rt == gt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO / "scripts/escha-compare/out"))
    ap.add_argument("--render-only", action="store_true",
                    help="render the prompts, print token counts and exit")
    ap.add_argument("--dump-prompts", action="store_true",
                    help="write rendered prompts to outdir/prompts_rendered/ and exit")
    ap.add_argument("--skip-server", action="store_true",
                    help="reuse an already running llama-server on PORT")
    ap.add_argument("--model", type=Path, default=MODEL_GGUF,
                    help="GGUF to test (default: legacy compact artifact)")
    ap.add_argument("--only", metavar="IDS",
                    help="comma-separated prompt IDs, e.g. P1,P2,P6")
    ap.add_argument("--max-new-tokens", type=int,
                    help="cap completion tokens for a fast deterministic prefix check")
    ap.add_argument("--ctx-size", type=int, default=4096,
                    help="llama-server context size (default: 4096)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    reference = json.loads(REFERENCE_SUITE.read_text(encoding="utf-8"))
    ref_by_id = {p["id"]: p for p in reference["prompts"]}
    prompts = render_prompts(tokenizer)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        unknown = wanted - {p["id"] for p in prompts}
        if unknown:
            raise ValueError(f"unknown prompt IDs: {', '.join(sorted(unknown))}")
        prompts = [p for p in prompts if p["id"] in wanted]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # sanity: P5 should be ~1544 tokens like the reference run
    for p in prompts:
        nt = len(tokenize(tokenizer, p["rendered"]))
        print(f"render {p['id']}: {nt} prompt tokens (ref {ref_by_id[p['id']].get('usage', {}).get('prompt_tokens')})")
    if args.render_only:
        return

    if args.dump_prompts:
        (outdir / "prompts_rendered").mkdir(parents=True, exist_ok=True)
        for p in prompts:
            (outdir / "prompts_rendered" / f"{p['id']}.txt").write_text(
                p["rendered"], encoding="utf-8")
        print(f"prompts written to {outdir / 'prompts_rendered'}")
        return

    proc = None
    if not args.skip_server:
        proc = start_server(args.model, args.ctx_size, outdir / "server.log")
    try:
        results = []
        for p in prompts:
            rid = p["id"]
            n_predict = p["max_new_tokens"]
            if args.max_new_tokens is not None:
                n_predict = min(n_predict, args.max_new_tokens)
            data = complete(p["rendered"], n_predict)
            got = data["content"]
            ref = ref_by_id[rid]
            ref_text = ref.get("content") or ""
            agree = token_agreement(tokenizer, ref_text, got)
            timings = data.get("timings", {})
            entry = {
                "id": rid,
                "category": ref["category"],
                "got": got,
                "reference": ref_text,
                "token_agreement": agree,
                "timings": timings,
            }
            results.append(entry)
            print(f"{rid}: {agree['compared_agree_rate']*100:.1f}% agreement over generated prefix "
                  f"({agree['prefix_agree_tokens']}/{agree['compared_tokens']}; "
                  f"reference length {agree['ref_tokens']})")
        report = {
            "method": "Beellama native escha port vs reference SGLang e3 outputs; "
                      "raw rendered prompts, temp 0, seed 42, token-level prefix agreement",
            "model": str(args.model),
            "prompts": results,
        }
        (outdir / "compare-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report: {outdir / 'compare-report.json'}")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
