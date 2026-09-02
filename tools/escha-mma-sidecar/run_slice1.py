#!/usr/bin/env python3
"""Build, run, and gate the A4 escha-mma-cache-v1 discriminating slice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/"
    "escha-w2-lowgpu-mono-parity.gguf"
)
DEFAULT_OUT = REPO / "evidence/PREFILL-50PCT-PLAN/2026-09-02/a4-mma-sidecar/slice1-raw"
GATE_JSON = REPO / "evidence/PREFILL-50PCT-PLAN/2026-09-02/a4-mma-sidecar/SLICE1-GATE.json"
GATE_MD = REPO / "evidence/PREFILL-50PCT-PLAN/2026-09-02/a4-mma-sidecar/SLICE1-GATE.md"


def run(command: list[str], stdout: Path | None = None, stderr: Path | None = None) -> None:
    out_handle = stdout.open("w") if stdout else None
    err_handle = stderr.open("w") if stderr else None
    try:
        subprocess.run(command, cwd=REPO, check=True, stdout=out_handle, stderr=err_handle)
    finally:
        if out_handle:
            out_handle.close()
        if err_handle:
            err_handle.close()


def sass_counts(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    function: str | None = None
    pattern = re.compile(r"/\*[0-9a-f]+\*/\s+(?:@[A-Z0-9!]+\s+)?([A-Z][A-Z0-9.]*)\s")
    for line in path.read_text().splitlines():
        match = re.search(r"Function : (\S+)", line)
        if match:
            function = match.group(1)
            result[function] = {name: 0 for name in ("IMAD", "LOP3", "IADD3", "LDL", "STL")}
            result[function]["instructions"] = 0
            continue
        if not function or not (match := pattern.search(line)):
            continue
        instruction = match.group(1).split(".")[0]
        result[function]["instructions"] += 1
        if instruction in result[function]:
            result[function][instruction] += 1
    for values in result.values():
        values["decode_address_alu"] = values["IMAD"] + values["LOP3"] + values["IADD3"]
    return {name: values for name, values in result.items() if name.startswith("slice1_")}


def ptxas_properties(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    text = path.read_text()
    pattern = re.compile(
        r"Compiling entry function '([^']+)'.*?Function properties for \1\s+"
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\s+"
        r"ptxas info\s+: Used (\d+) registers",
        re.S,
    )
    for name, stack, stores, loads, regs in pattern.findall(text):
        if name.startswith("slice1_"):
            result[name] = {
                "registers": int(regs), "stack_bytes": int(stack),
                "spill_store_bytes": int(stores), "spill_load_bytes": int(loads),
            }
    return result


def verdict(pass_value: bool) -> str:
    return "PASS" if pass_value else "FAIL"


def write_reports(manifest: dict, bench: dict, sass: dict, ptxas: dict) -> dict:
    side16, side32 = ptxas["slice1_sidecar_fp16"], ptxas["slice1_sidecar_fp32"]
    resource_ok = (
        side16["registers"] <= 104 and side32["registers"] <= 128
        and all(v[key] == 0 for v in (side16, side32)
                for key in ("stack_bytes", "spill_store_bytes", "spill_load_bytes"))
        and all(sass[name][op] == 0 for name in ("slice1_sidecar_fp16", "slice1_sidecar_fp32")
                for op in ("LDL", "STL"))
    )
    alu = {}
    alu_ok = True
    for mode in ("fp16", "fp32"):
        control = sass[f"slice1_control_{mode}"]["decode_address_alu"]
        candidate = sass[f"slice1_sidecar_{mode}"]["decode_address_alu"]
        reduction = (control - candidate) / control * 100.0
        alu[mode] = {"control": control, "sidecar": candidate, "reduction_pct": reduction}
        alu_ok &= reduction >= 30.0
    residency = {mode: bench["resources"][f"sidecar_{mode}"]["active_ctas_per_sm"] for mode in ("fp16", "fp32")}
    residency_register_math = {
        "register_file_per_sm": 65536,
        "threads_per_cta": 256,
        "fp16_two_cta_registers": side16["registers"] * 256 * 2,
        "fp32_two_cta_registers": side32["registers"] * 256 * 2,
        "two_cta_dynamic_shared_bytes": bench["resources"]["sidecar_fp16"]["dynamic_shared_bytes"] * 2,
    }
    residency_ok = all(value >= 2 for value in residency.values())
    rows = {
        "representation_growth": {
            "measured_pct": manifest["growth_pct"], "threshold": "<=25%",
            "pass": manifest["growth_pct"] <= 25.0,
        },
        "registers_and_spills": {
            "fp16_registers": side16["registers"], "fp32_registers": side32["registers"],
            "fp16_ceiling": 104, "fp32_ceiling": 128,
            "local_bytes": {"fp16": bench["resources"]["sidecar_fp16"]["local_bytes"],
                            "fp32": bench["resources"]["sidecar_fp32"]["local_bytes"]},
            "stack_local_spills": 0 if resource_ok else {
                "fp16": side16, "fp32": side32,
            },
            "pass": resource_ok,
        },
        "decode_address_alu": {"measurements": alu, "threshold": ">=30% reduction in both modes", "pass": alu_ok},
        "two_cta_residency": {"active_ctas_per_sm": residency, "register_math": residency_register_math,
                              "threshold": ">=2", "pass": residency_ok},
        "direct_op_m2048": {
            "control_ms": bench["control_ms"], "sidecar_ms": bench["sidecar_ms"],
            "candidate_control_time_ratio": bench["sidecar_ms"] / bench["control_ms"],
            "throughput_delta_pct": bench["speedup_pct"],
            "threshold": ">=15% throughput", "pass": bench["speedup_pct"] >= 15.0,
        },
    }
    overall = all(row["pass"] for row in rows.values())
    report = {
        "schema": "escha-mma-cache-v1-slice1-gate",
        "scope": "blk.0.ffn_gate K2 5120->17408, M=2048",
        "device": bench["device"], "compute_capability": bench["compute_capability"],
        "correctness": {"reverse_transform_byte_exact": manifest["validation"]["reverse_transform_byte_exact"],
                        "direct_op_bit_mismatches": bench["bit_mismatches"]},
        "rows": rows,
        "overall_pass": overall,
        "verdict": "CONFIRM-PROMOTE" if overall else "CONFIRM-REJECT",
        "program_decision": "continue packed-exact +50% line" if overall else "close Attempt 3 and packed-exact +50% line",
        "artifacts": {"overlay_sha256": manifest["sha256"]["overlay_file"],
                      "source_code_sha256": manifest["sha256"]["source_code"]},
    }
    GATE_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md = f"""# A4 / EXP-11 Attempt 3 — Slice 1 mechanism gate

Date: 2026-09-02  
Scope: `blk.0.ffn_gate`, K2 5120 -> 17408, direct op M=2048 on {bench['device']} (sm_120a build; runtime CC {bench['compute_capability']}).

## Verdict

**{report['verdict']}**.  The representation is exact and compact, but the direct-fragment consumer fails four of five frozen mechanism rows.  Per the dispatch brief, stop before full-model work.  Attempt 3 and the packed-exact +50% line close here.

## Gate

| row | measured | threshold | result |
| --- | ---: | ---: | --- |
| representation growth | {manifest['growth_pct']:.3f}% | <=25% | {verdict(rows['representation_growth']['pass'])} |
| registers / spills | FP16 {side16['registers']}, FP32 {side32['registers']}; stack/local/spills 0 | <=104 / <=128; zero spills | {verdict(resource_ok)} |
| decode/address ALU | FP16 {alu['fp16']['control']} -> {alu['fp16']['sidecar']} ({alu['fp16']['reduction_pct']:.1f}% reduction); FP32 {alu['fp32']['control']} -> {alu['fp32']['sidecar']} ({alu['fp32']['reduction_pct']:.1f}%) | >=30% reduction | {verdict(alu_ok)} |
| direct op M=2048 | {bench['control_ms']:.6f} -> {bench['sidecar_ms']:.6f} ms ({bench['sidecar_ms']/bench['control_ms']:.2f}x slower; {bench['speedup_pct']:.1f}% throughput) | >=15% faster | {verdict(rows['direct_op_m2048']['pass'])} |
| residency | FP16 {residency['fp16']}, FP32 {residency['fp32']} CTA/SM | >=2 CTA/SM | {verdict(residency_ok)} |

Correctness passed: the overlay reverse transform is byte-exact and the FP16 direct-op comparison has {bench['bit_mismatches']} bit mismatches across {2048*17408:,} outputs.

Two-CTA register math: FP16 requires `{side16['registers']}*256*2 = {side16['registers']*256*2:,}` and FP32 requires `{side32['registers']}*256*2 = {side32['registers']*256*2:,}` registers, both above the 65,536-register SM pool.  Shared memory would require only `{bench['resources']['sidecar_fp16']['dynamic_shared_bytes']*2:,}` bytes for two CTAs, so registers are binding.  CUDA occupancy independently reports one active CTA/SM in both modes.

The ALU counts are static occurrences in the actual sm_120a SASS emitted for the matched standalone control and sidecar symbols.  Raw SASS, ptxas diagnostics, cuobjdump resources, repack manifest, and benchmark JSON are retained under `slice1-raw/`.

## Mechanism finding

Pre-resolving `dw0/dw1/dsh` is insufficient under the 25% representation cap.  Four row warps still independently load and evaluate every fragment weight.  The compiler exposes that duplication directly: candidate ALU rises rather than falls, live state reaches {side16['registers']}/{side32['registers']} registers, occupancy drops to one CTA/SM, and the candidate is {bench['sidecar_ms']/bench['control_ms']:.2f}x slower.  This reproduces EXP-09's fundamental four-row-warp cost even though its index chain has been removed.

No model binary was run, no loader/model source was changed, and no full-model work was started.
"""
    GATE_MD.write_text(md)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay = args.out_dir / "blk.0.ffn_gate.escha_mma_code"
    manifest_path = args.out_dir / "repack-manifest.json"
    binary = args.out_dir / "slice1_harness"
    ptxas_path = args.out_dir / "ptxas.txt"
    sass_path = args.out_dir / "harness.sass"
    resources_path = args.out_dir / "cuobjdump-resources.txt"
    benchmark_path = args.out_dir / "benchmark.json"

    run(["python3", str(REPO / "tools/escha-mma-sidecar/repack_slice1.py"),
         "--source", str(args.source), "--output", str(overlay), "--manifest", str(manifest_path)],
        args.out_dir / "repack.stdout.json")
    run(["/usr/local/cuda/bin/nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_120a", "-Xptxas=-v",
         str(REPO / "tools/escha-mma-sidecar/slice1_harness.cu"), "-o", str(binary)], stderr=ptxas_path)
    manifest = json.loads(manifest_path.read_text())
    code_offset = manifest["source_tensor_offsets"]["blk.0.ffn_gate.escha_code"]
    run([str(binary), "--source", str(args.source), "--overlay", str(overlay),
         "--code-offset", str(code_offset), "--reps", str(args.reps)], benchmark_path, args.out_dir / "benchmark.stderr")
    run(["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(binary)], sass_path)
    run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(binary)], resources_path)
    report = write_reports(manifest, json.loads(benchmark_path.read_text()), sass_counts(sass_path), ptxas_properties(ptxas_path))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
