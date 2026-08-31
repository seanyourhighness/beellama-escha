# Definition of Done — Escha club-3090 medium 5-pack benchmark

Scope: 4 model GGUFs scored on the beellama-escha custom runtime (build-cuda-qwen35-gated after escha_version gating fix), club-3090 medium suite (5 non-Docker packs, 75 scenarios, thinking force-off).

Criteria (each must be verifiable from evidence files via deterministic commands):
1. Four final result JSONs exist in evidence dir: base-escha-w2.json, p-arch-23.json, p-arch-23g.json, original-lowgpu.json.
2. Each JSON: mode=="medium", thinking_enabled==false, thinking_mode=="force-off", totals.total==75, and exactly the 5 packs {toolcall-15, instructfollow-15, structoutput-15, dataextract-15, reasonmath-15} with 15 scenarios each; no sandbox/Docker pack present.
3. Each model has baseline pass count /75 and equivalent_score_150 reported from the JSON (pass@1 baseline, not retry/pass@k).
4. Base Escha W2 artifact used = escha-w2-lowgpu-mono-parity.gguf (sha256 e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d), served with chat parsing enabled (no --skip-chat-parsing), F16 KV, single slot.
5. original-lowgpu artifact = Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf, scored on the GATED build with coherent output (no degenerate loops; preflight chat + tool-call HTTP 200 with real content).
6. Evidence dir contains: sha256sums.txt, llama-server-version.txt, git-head.txt, benchlocal-version.txt, four .server.log journals with no HTTP 500 / PEG-native errors / garbage-token signatures for the scored models.
7. sha256sum -c sha256sums.txt passes; each result JSON parses with python3 json.load.
