#!/usr/bin/env python3
"""Build, run, and gate the collective A4 sidecar V2 Slice 1 consumer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from run_slice1 import ptxas_properties, sass_counts


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/"
    "escha-w2-lowgpu-mono-parity.gguf"
)
EVIDENCE = REPO / "evidence/PREFILL-50PCT-PLAN/2026-09-02/a4-mma-sidecar"
DEFAULT_OUT = EVIDENCE / "v2-raw"
GATE_JSON = EVIDENCE / "V2-GATE.json"
GATE_MD = EVIDENCE / "V2-GATE.md"


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


def verdict(value: bool) -> str:
    return "PASS" if value else "FAIL"


def alu_change(reduction_pct: float) -> str:
    if reduction_pct >= 0:
        return f"{reduction_pct:.1f}% reduction"
    return f"{-reduction_pct:.1f}% increase"


def write_reports(manifest: dict, bench: dict, sass: dict, ptxas: dict) -> dict:
    v16 = ptxas["slice1_v2_fp16"]
    v32 = ptxas["slice1_v2_fp32"]
    resource_ok = (
        v16["registers"] <= 104
        and v32["registers"] <= 128
        and all(
            item[key] == 0
            for item in (v16, v32)
            for key in ("stack_bytes", "spill_store_bytes", "spill_load_bytes")
        )
        and all(
            sass[name][op] == 0
            for name in ("slice1_v2_fp16", "slice1_v2_fp32")
            for op in ("LDL", "STL")
        )
        and bench["resources"]["v2_fp16"]["local_bytes"] == 0
        and bench["resources"]["v2_fp32"]["local_bytes"] == 0
    )

    alu = {}
    alu_ok = True
    for mode in ("fp16", "fp32"):
        control = sass[f"slice1_control_{mode}"]["decode_address_alu"]
        candidate = sass[f"slice1_v2_{mode}"]["decode_address_alu"]
        reduction = (control - candidate) / control * 100.0
        alu[mode] = {
            "control": control,
            "v2": candidate,
            "reduction_pct": reduction,
        }
        alu_ok &= reduction >= 30.0

    residency = {
        mode: bench["resources"][f"v2_{mode}"]["active_ctas_per_sm"]
        for mode in ("fp16", "fp32")
    }
    residency_ok = all(value >= 2 for value in residency.values())
    correctness_ok = (
        manifest["validation"]["reverse_transform_byte_exact"]
        and bench["bit_mismatches"] == 0
    )
    speedup = bench["speedup_pct"]

    rows = {
        "representation_growth": {
            "measured_pct": manifest["growth_pct"],
            "threshold": "<=25%",
            "pass": manifest["growth_pct"] <= 25.0,
        },
        "correctness_bit_compare": {
            "reverse_transform_byte_exact": manifest["validation"]["reverse_transform_byte_exact"],
            "direct_op_bit_mismatches": bench["bit_mismatches"],
            "compared_outputs": 2048 * 17408,
            "threshold": "byte-exact reverse and zero bit mismatches",
            "pass": correctness_ok,
        },
        "registers_and_spills": {
            "fp16_registers": v16["registers"],
            "fp32_registers": v32["registers"],
            "fp16_ceiling": 104,
            "fp32_ceiling": 128,
            "stack_bytes": {"fp16": v16["stack_bytes"], "fp32": v32["stack_bytes"]},
            "local_bytes": {
                "fp16": bench["resources"]["v2_fp16"]["local_bytes"],
                "fp32": bench["resources"]["v2_fp32"]["local_bytes"],
            },
            "spill_bytes": {
                "fp16_load": v16["spill_load_bytes"],
                "fp16_store": v16["spill_store_bytes"],
                "fp32_load": v32["spill_load_bytes"],
                "fp32_store": v32["spill_store_bytes"],
            },
            "sass_ldl_stl": {
                "fp16": [sass["slice1_v2_fp16"]["LDL"], sass["slice1_v2_fp16"]["STL"]],
                "fp32": [sass["slice1_v2_fp32"]["LDL"], sass["slice1_v2_fp32"]["STL"]],
            },
            "pass": resource_ok,
        },
        "decode_address_alu": {
            "measurements": alu,
            "threshold": ">=30% reduction in both modes",
            "pass": alu_ok,
        },
        "two_cta_residency": {
            "active_ctas_per_sm": residency,
            "dynamic_shared_bytes_per_cta": 45056,
            "register_math": {
                "register_file_per_sm": 65536,
                "threads_per_cta": 256,
                "fp16_two_cta_registers": v16["registers"] * 256 * 2,
                "fp32_two_cta_registers": v32["registers"] * 256 * 2,
                "two_cta_dynamic_shared_bytes": 45056 * 2,
            },
            "threshold": ">=2",
            "pass": residency_ok,
        },
        "direct_op_m2048": {
            "control_ms": bench["control_ms"],
            "v2_ms": bench["v2_ms"],
            "candidate_control_time_ratio": bench["v2_ms"] / bench["control_ms"],
            "throughput_delta_pct": speedup,
            "threshold": ">=15% throughput",
            "pass": speedup >= 15.0,
        },
    }
    overall = all(row["pass"] for row in rows.values())
    report = {
        "schema": "escha-mma-cache-v1-collective-v2-slice1-gate",
        "scope": "blk.0.ffn_gate K2 5120->17408, M=2048",
        "device": bench["device"],
        "compute_capability": bench["compute_capability"],
        "consumer": {
            "cta": "128x128, 256 threads",
            "bands": 2,
            "warps_per_band": 4,
            "descriptors_per_thread": 4,
            "decoded_values_per_thread": 8,
            "decoded_values_per_cta_tile": 2048,
            "a4_v1_decoded_values_per_cta_tile": 8192,
            "shared_b_publications_per_band_tile": 1,
            "dynamic_shared_bytes": 45056,
            "barriers": {
                "entry_descriptor_publish_cta": 1,
                "per_tile_input_publish_cta": 1,
                "per_tile_decoded_b_publish_band_group": 1,
                "per_tile_reuse_cta": 1,
            },
        },
        "rows": rows,
        "overall_pass": overall,
        "verdict": "CONFIRM-PROMOTE" if overall else "CONFIRM-REJECT",
        "program_decision": (
            "sidecar V2 viable; continue packed-exact line"
            if overall
            else "sidecar V2 not viable; close packed-exact line"
        ),
        "artifacts": {
            "overlay_sha256": manifest["sha256"]["overlay_file"],
            "source_code_sha256": manifest["sha256"]["source_code"],
        },
    }
    GATE_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    time_ratio = rows["direct_op_m2048"]["candidate_control_time_ratio"]
    md = f"""# A4 MMA sidecar V2 — collective-consumer Slice 1 gate

Date: 2026-09-02  
Scope: `blk.0.ffn_gate`, K2 5120 -> 17408, direct op M=2048 on {bench['device']} (sm_120a; runtime CC {bench['compute_capability']}).

## Verdict

**{report['verdict']}**. V2 proves that the collective architecture fixes A4's duplication, register pressure, residency, and correctness failures, but it does not beat the packed shared-B control. The pre-resolved descriptor/record consumer has more decode/address ALU and is {time_ratio:.2f}x slower at M=2048. Per the V2 terminal gate, the exact-packed sidecar line closes; use the 23/23I hybrid delivery path.

## Mechanism gate

| row | measured | threshold | result |
| --- | ---: | ---: | --- |
| representation growth | {manifest['growth_pct']:.3f}% | <=25% | {verdict(rows['representation_growth']['pass'])} |
| correctness bit-compare | byte-exact reverse; {bench['bit_mismatches']} / {2048*17408:,} mismatches | exact; zero | {verdict(correctness_ok)} |
| registers / spills | FP16 {v16['registers']}, FP32 {v32['registers']}; stack/local/spills 0 | <=104 / <=128; zero | {verdict(resource_ok)} |
| decode/address ALU | FP16 {alu['fp16']['control']} -> {alu['fp16']['v2']} ({alu_change(alu['fp16']['reduction_pct'])}); FP32 {alu['fp32']['control']} -> {alu['fp32']['v2']} ({alu_change(alu['fp32']['reduction_pct'])}) | >=30% reduction | {verdict(alu_ok)} |
| direct op M=2048 | {bench['control_ms']:.6f} -> {bench['v2_ms']:.6f} ms ({time_ratio:.2f}x time; {speedup:.1f}% throughput) | >=15% faster | {verdict(rows['direct_op_m2048']['pass'])} |
| residency | FP16 {residency['fp16']}, FP32 {residency['fp32']} CTA/SM; 45,056 B/CTA | >=2 CTA/SM | {verdict(residency_ok)} |

## Mechanism result

The collective coverage is exact: each four-warp band executes 128 threads x 4 descriptors x 2 values = 1024 unique decodes, publishes one `[64][16]` tile, and all four row warps consume the same bytes via `ldmatrix`. The CTA therefore performs 2,048 codebook evaluations per K16 tile versus A4 V1's 8,192. The fully unrolled MMA seam avoids EXP-10 accumulator homing: FP16/FP32 use {v16['registers']}/{v32['registers']} registers with no local traffic and both retain two-CTA residency.

That structural repair is insufficient. Against the same packed control symbol, static `IMAD+LOP3+IADD3` rises rather than falling, and the directly measured FP16 operator loses {abs(speedup):.1f}% throughput. The remaining cost is the descriptor-driven adjacent-pair access and publication path, not four-row-warp duplication. This is the previously untested collective consumer of the pre-resolved stream, and it fails the two gates that determine whether the sidecar can outrun current packed decode.

Raw repack, compiler, SASS, resource, and benchmark evidence is in `v2-raw/`. No llama model binary ran; no `ggml/` or loader source was modified.
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
    binary = args.out_dir / "slice1_v2_harness"
    ptxas_path = args.out_dir / "ptxas.txt"
    sass_path = args.out_dir / "harness.sass"
    resources_path = args.out_dir / "cuobjdump-resources.txt"
    benchmark_path = args.out_dir / "benchmark.json"

    run(
        [
            "python3", str(REPO / "tools/escha-mma-sidecar/repack_slice1.py"),
            "--source", str(args.source), "--output", str(overlay),
            "--manifest", str(manifest_path),
        ],
        args.out_dir / "repack.stdout.json",
    )
    run(
        [
            "/usr/local/cuda/bin/nvcc", "-std=c++17", "-O3", "-lineinfo",
            "-arch=sm_120a", "-Xptxas=-v",
            str(REPO / "tools/escha-mma-sidecar/slice1_v2_harness.cu"),
            "-o", str(binary),
        ],
        stderr=ptxas_path,
    )
    manifest = json.loads(manifest_path.read_text())
    code_offset = manifest["source_tensor_offsets"]["blk.0.ffn_gate.escha_code"]
    run(
        [
            str(binary), "--source", str(args.source), "--overlay", str(overlay),
            "--code-offset", str(code_offset), "--reps", str(args.reps),
        ],
        benchmark_path,
        args.out_dir / "benchmark.stderr",
    )
    run(["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(binary)], sass_path)
    run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(binary)], resources_path)

    report = write_reports(
        manifest,
        json.loads(benchmark_path.read_text()),
        sass_counts(sass_path),
        ptxas_properties(ptxas_path),
    )
    (args.out_dir / "run.stdout.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
