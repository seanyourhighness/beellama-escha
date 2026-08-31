Escha club-3090 medium 5-pack — gated beellama runtime (build-cuda-qwen35-gated; qwen35.cpp gated on escha_version)

base-escha-w2: mode=medium think_force_off=True think_enabled_false=True total75=True exact5packs=True all15=True passed=65 equiv150=130
p-arch-23: mode=medium think_force_off=True think_enabled_false=True total75=True exact5packs=True all15=True passed=65 equiv150=130
p-arch-23g: mode=medium think_force_off=True think_enabled_false=True total75=True exact5packs=True all15=True passed=65 equiv150=130
original-lowgpu: mode=medium think_force_off=True think_enabled_false=True total75=True exact5packs=True all15=True passed=66 equiv150=132
base-escha-w2: server_log_real_errors=none
p-arch-23: server_log_real_errors=none
p-arch-23g: server_log_real_errors=none
original-lowgpu: server_log_real_errors=none
base_artifact_sha256_match=True
llama-server-version.txt_present=True_nonempty=True
git-head.txt_present=True_nonempty=True
benchlocal-version.txt_present=True_nonempty=True
sha256sums.txt_present=True_nonempty=True
no_sandbox_packs=True packs=['dataextract-15', 'instructfollow-15', 'reasonmath-15', 'structoutput-15', 'toolcall-15']
base-escha-w2_decode_tps=40.6
p-arch-23_decode_tps=60.0
p-arch-23g_decode_tps=60.6
original-lowgpu_decode_tps=80.8

sha256sum_c_all_OK=True (9/9 artifacts verified)
base-escha-w2_preflight_http200=True
p-arch-23_preflight_http200=True
p-arch-23g_preflight_http200=True
original-lowgpu_preflight_http200=True




plain_chat: finish=stop content='4' tool_calls=False
tool_call: finish=tool_calls tool_calls=True content=''
COHERENT_OK=True

## Server launch lines (from captured server journals — proves benchmark launch configuration)
base-escha-w2_launch: escha-gated-base.service - "/mnt/d/CODEX WORKSPACE/beellama-escha/build-cuda-qwen35-gated/bin/llama-server" -m "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf" --alias base-escha-w2 --host 127.0.0.1 --port 18111 -ngl all -c 327
base-escha-w2_flags: ngl=all ctx=32768 single_slot_np1=1 flash_attn=on kv_f16=f16/f16 jinja=yes thinking_off=yes chat_parsing=ENABLED (no --skip-chat-parsing)
p-arch-23_launch: escha-gated-p23.service - "/mnt/d/CODEX WORKSPACE/beellama-escha/build-cuda-qwen35-gated/bin/llama-server" -m "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity-standard-ffn-gdn-q2k.gguf" --alias p-arch-23 --host 127.0.0.1 --port 18112
p-arch-23_flags: ngl=all ctx=32768 single_slot_np1=1 flash_attn=on kv_f16=f16/f16 jinja=yes thinking_off=yes chat_parsing=ENABLED (no --skip-chat-parsing)
p-arch-23g_launch: escha-gated-p23g.service - "/mnt/d/CODEX WORKSPACE/beellama-escha/build-cuda-qwen35-gated/bin/llama-server" -m "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity-standard-ffn-gdn-q2k-embedq4.gguf" --alias p-arch-23g --host 127.0.0.1 --
p-arch-23g_flags: ngl=all ctx=32768 single_slot_np1=1 flash_attn=on kv_f16=f16/f16 jinja=yes thinking_off=yes chat_parsing=ENABLED (no --skip-chat-parsing)
original-lowgpu_launch: escha-gated-orig.service - "/mnt/d/CODEX WORKSPACE/beellama-escha/build-cuda-qwen35-gated/bin/llama-server" -m "/mnt/d/CODEX WORKSPACE/beellama-release/models/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf" --alias original-lowgpu --host 127.0.0.1 --port 18114 -ngl all
original-lowgpu_flags: ngl=all ctx=32768 single_slot_np1=1 flash_attn=on kv_f16=f16/f16 jinja=yes thinking_off=yes chat_parsing=ENABLED (no --skip-chat-parsing)

## Deterministic python3 json.load verification of the four result JSONs
base-escha-w2: json.load=OK packs=['toolcall-15', 'instructfollow-15', 'structoutput-15', 'dataextract-15', 'reasonmath-15'] mode=medium thinking_mode=force-off thinking_enabled=False totals_total=75 totals_passed=65 equivalent_score_150=130
p-arch-23: json.load=OK packs=['toolcall-15', 'instructfollow-15', 'structoutput-15', 'dataextract-15', 'reasonmath-15'] mode=medium thinking_mode=force-off thinking_enabled=False totals_total=75 totals_passed=65 equivalent_score_150=130
p-arch-23g: json.load=OK packs=['toolcall-15', 'instructfollow-15', 'structoutput-15', 'dataextract-15', 'reasonmath-15'] mode=medium thinking_mode=force-off thinking_enabled=False totals_total=75 totals_passed=65 equivalent_score_150=130
original-lowgpu: json.load=OK packs=['toolcall-15', 'instructfollow-15', 'structoutput-15', 'dataextract-15', 'reasonmath-15'] mode=medium thinking_mode=force-off thinking_enabled=False totals_total=75 totals_passed=66 equivalent_score_150=132
all_four_json_load_exit=0

## Deterministic evidence-directory listing (proves the four result JSONs and four .server.log journals exist)
file base-escha-w2.json 1168939
file base-escha-w2.preflight.txt 851
file base-escha-w2.server.log 125909
file base-escha-w2.stderr.log 3589
file base-escha-w2.stdout.json 1168940
file benchlocal-version.txt 135
file git-head.txt 173
file llama-server-version.txt 115
file original-lowgpu.json 1149646
file original-lowgpu.preflight.txt 824
file original-lowgpu.server.log 115696
file original-lowgpu.stderr.log 3580
file original-lowgpu.stdout.json 1149647
file p-arch-23.json 1106602
file p-arch-23.preflight.txt 863
file p-arch-23.server.log 119061
file p-arch-23.stderr.log 3586
file p-arch-23.stdout.json 1106603
file p-arch-23g.json 1102672
file p-arch-23g.preflight.txt 848
file p-arch-23g.server.log 118943
file p-arch-23g.stderr.log 3586
file p-arch-23g.stdout.json 1102673
file sha256sums.txt 1361

required_json_present=PASS,PASS,PASS,PASS
required_server_log_present=PASS,PASS,PASS,PASS
