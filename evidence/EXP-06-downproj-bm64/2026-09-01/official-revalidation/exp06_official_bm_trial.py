#!/usr/bin/env python3
"""EXP-06 Phase 1: fresh-process official down-proj BM revalidation.
One invocation of this script is one fresh process/config. Tensors are seeded,
fixed-shape synthetic direct-op evidence only; records selected kernel, resources,
grid/block and timing. Optionally saves output for cross-BM numerical comparison.
"""
import hashlib, json, os, sys, time
import torch, escha

label = sys.argv[1]                 # bm128 or bm64
out_file = sys.argv[2] if len(sys.argv) > 2 else None
M, K, IC, OC, ACC = 2048, 3, 17408, 5120, 0
assert label in ("bm128", "bm64")
expected = "128" if label == "bm128" else "64"
assert os.environ.get("ESCHAM_GEMM_BM", "128") == expected

torch.manual_seed(20260901)
torch.cuda.manual_seed_all(20260901)
dev = torch.device("cuda:0")
# Exact shapes used by fused direct op; same random tensor generation per process.
code = torch.randint(-8, 8, (16*K, OC//16, IC//16), dtype=torch.int16, device=dev)
rin = torch.rand(IC, dtype=torch.float16, device=dev); rout = torch.rand(OC, dtype=torch.float16, device=dev)
x = torch.rand(M, IC, dtype=torch.float16, device=dev); s_in = torch.rand(M, dtype=torch.float32, device=dev); s_out = torch.rand(M, dtype=torch.float32, device=dev)
def call(): return torch.ops.escha.escham_code_gemm(x, code, rin, rout, s_in, s_out, OC, K, True, False, ACC)
# fixed warmup per fresh process
for _ in range(8): y = call()
torch.cuda.synchronize()
# event timing: only direct op batch; 50 calls reduces timer quantization
start, end = torch.cuda.Event(True), torch.cuda.Event(True)
start.record()
for _ in range(50): y = call()
end.record(); torch.cuda.synchronize()
per_call_ms = start.elapsed_time(end) / 50
# one profiled op for selected symbol/timing
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False) as prof:
    y = call(); torch.cuda.synchronize()
entries = [e for e in prof.key_averages() if "code_gemm_kernel" in e.key]
assert len(entries) == 1, [e.key for e in entries]
e = entries[0]
# Trace export gives grid/block/regs/shared mem. Extract only matching kernel event.
trace = f"/tmp/escha-wheel/exp06-{label}-{os.getpid()}.trace.json"
prof.export_chrome_trace(trace)
trace_json = json.load(open(trace))
match = [ev for ev in trace_json.get("traceEvents", []) if "code_gemm_kernel" in ev.get("name", "")]
args = match[-1].get("args", {}) if match else {}
# Compare/save only the first pair output, not benchmark outputs.
y_cpu = y.detach().float().cpu().contiguous()
raw = y_cpu.numpy().tobytes()
if out_file: torch.save(y_cpu, out_file)
result = {
 "label": label, "env": {k:v for k,v in os.environ.items() if k.startswith("ESCHAM_GEMM_")},
 "shape": {"M":M,"K":K,"IC":IC,"OC":OC,"acc_mode":ACC},
 "warmup_calls":8,"timed_calls":50,"per_call_ms":round(per_call_ms,6),
 "profile_kernel":e.key,"profile_kernel_ms":round(e.self_device_time_total/1000,6),
 "trace_args":args,"output_sha256":hashlib.sha256(raw).hexdigest(),
 "output_sum":float(y_cpu.sum()),"output_l2":float(torch.linalg.vector_norm(y_cpu)),
 "cuda":torch.version.cuda,"torch":torch.__version__,"trace":trace,
}
print(json.dumps(result,sort_keys=True))
