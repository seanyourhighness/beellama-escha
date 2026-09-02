#!/usr/bin/env bash
# EXP-08 numeric/parity smoke: P2 + P7 deterministic generation, control vs fused candidate.
# Serves each binary on its own port, requests greedy seed-42 completions, compares token
# sequences. 16/16 token agreement per prompt = parity PASS.
set -uo pipefail
REPO="/mnt/d/CODEX WORKSPACE/beellama-escha"
MODEL="/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf"
OUT="${1:-$REPO/evidence/EXP-08-fusedfinalize/2026-09-01}"
mkdir -p "$OUT/smoke"
PY=$(command -v python3)
CONTROL_BIN="$REPO/build-cuda-base01/bin/llama-server"
CAND_BIN="$REPO/build-cuda-exp08-fusedfin/bin/llama-server"

run_arm() {
  local bin="$1" port="$2" label="$3"
  "$bin" -m "$MODEL" --port "$port" -c 4096 --no-webui \
    -ngl 99 -ctk f16 -ctv f16 -fa on \
    >"$OUT/smoke/$label.server.log" 2>&1 &
  local pid=$!
  # wait for /health
  for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then break; fi
    sleep 2
  done
  echo "$pid"
}

kill_arm() { local pid="$1"; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }

CONTROL_PID=$(run_arm "$CONTROL_BIN" 18351 control)
CAND_PID=$(run_arm "$CAND_BIN" 18352 candidate)
echo "servers up: control=$CONTROL_PID cand=$CAND_PID"

python3 - "$OUT" <<'PY'
import json, sys, urllib.request, time
out = sys.argv[1]
d = json.load(open("/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/results/prompts.json"))
ps = {p["id"]: p for p in d.get("prompts", [])}

def complete(port, prompt, max_tokens=16):
    payload = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0, "seed": 42,
               "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(10):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception:
            time.sleep(5)
    return {"error": "timeout"}

report = {"p2": {}, "p7": {}}
for pid in ("P2-factual", "P7-tool-call"):
    p = ps[pid]
    content = p.get("content")
    prompt = content if isinstance(content, str) else json.dumps(content) if content is not None else ""
    c = complete(18351, prompt)
    k = complete(18352, prompt)
    ct = "".join(c.get("content", [c.get("content", "")])) if isinstance(c.get("content"), list) else c.get("content", "")
    kt = "".join(k.get("content", [k.get("content", "")])) if isinstance(k.get("content"), list) else k.get("content", "")
    # tokenize content roughly by chars; for parity use full string compare of tokens via /tokenize
    def toks(port, text):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize",
                data=json.dumps({"content": text}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r).get("tokens", [])
        except Exception:
            return []
    c_toks = toks(18351, ct)
    k_toks = toks(18352, kt)
    agree = c_toks == k_toks
    report["p2" if pid.startswith("P2") else "p7"] = {
        "control_tokens": c_toks, "candidate_tokens": k_toks,
        "control_len": len(c_toks), "candidate_len": len(k_toks),
        "agree": agree, "control_error": c.get("error"), "candidate_error": k.get("error"),
    }
    print(f"{pid}: control_len={len(c_toks)} candidate_len={len(k_toks)} agree={agree}")
with open(f"{out}/smoke/parity.json", "w") as f:
    json.dump(report, f, indent=2)
print("WROTE", f"{out}/smoke/parity.json")
PY

kill_arm "$CONTROL_PID"
kill_arm "$CAND_PID"
echo "SMOKE DONE"
