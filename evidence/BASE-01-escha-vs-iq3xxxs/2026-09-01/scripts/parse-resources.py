import re, sys, json

txt = open(sys.argv[1], errors="replace").read()
out = {}
# parse blocks: "Function <name>:" followed by attribute lines until next Function
blocks = re.split(r"(?=Function )", txt)
for b in blocks:
    m = re.match(r"Function (\S+):", b)
    if not m:
        continue
    name = m.group(1)
    # extract key resource lines
    regs = re.search(r"REG:(\d+)", b)
    stk = re.search(r"STACK:(\d+)", b)
    loc = re.search(r"LOCAL:(\d+)", b)
    shr = re.search(r"SHARED:(\d+)", b)
    if regs or stk or shr:
        out[name] = {"regs": int(regs.group(1)) if regs else None,
                     "stack": int(stk.group(1)) if stk else None,
                     "local": int(loc.group(1)) if loc else None,
                     "shared": int(shr.group(1)) if shr else None}

want = ["escha_matmul_dense_tiled_mmaILi2ELi128ELi128ELb1",  # K2 fp16 acc
        "escha_matmul_dense_tiled_mmaILi2ELi128ELi128ELb0",  # K2 fp32
        "escha_matmul_dense_tiled_mmaILi3ELi128ELi128ELb1",  # K3 fp16
        "escha_matmul_dense_tiled_mmaILi3ELi128ELi128ELb0",  # K3 fp32
        "escha_matmul_dense_tiledILi2", "escha_matmul_dense_tiledILi3",
        "escha_rotate_in_dense", "escha_finalize_dense", "escha_matmul_partial",
        "mul_mat_q", "mmq", "iq3"]
for name, r in sorted(out.items()):
    if any(w in name for w in want):
        print(f"{name}\n  regs={r['regs']} stack={r['stack']} local={r['local']} shared={r['shared']}")

with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1)
print("WROTE", sys.argv[2])
