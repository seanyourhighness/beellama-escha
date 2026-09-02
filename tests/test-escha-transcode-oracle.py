#!/usr/bin/env python3
"""Golden and cache-oracle checks for EXP-11 Attempt-1 Slice 1."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import types
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gguf-py"))
sys.path.insert(0, str(REPO / "conversion" / "escha"))

import gguf
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

import transcode_oracle as oracle
from transcode_cache_schema import FFN_DIMENSIONS, sidecar_tensor_name

REFERENCE_TOOLS = Path("/home/sean/research/escha-refs/yaniss/tools/escha")
CHECKPOINT = Path(
    "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono"
)
CONVERTER_GDN_Q2_GOLDEN = {
    "source_sha256": "ac01fe3fd9c098e90eb9c4ac8524911ae5bd27ebb215259d6bd84143aafe61ac",
    "payload_sha256": "03cc407c46fee4561aff978a8b505afc1ec2d003abb2fbe4d57ec0c13fffa7a4",
    "payload_bytes": 10321920,
}
QUANT_GOLDENS = {
    "q2_k": "b34768a5c2d06eadd84bd962b0b4705f76ed02e4ec9bfba11fc7308a5511d7ed",
    "q4_k": "5f9804b208bfa404edc8755b44cbf4d430d28ae0d95e0478f9daacadcd96050f",
    "q6_k": "5c8734e412edc2370bd460454e0f1e8ba0e2c11aa0402a4de26bc902919ee3d0",
}


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value).view(np.uint8)
    return hashlib.sha256(memoryview(array)).hexdigest()


def mean_abs_error(restored: np.ndarray, source: np.ndarray, rows: int = 128) -> float:
    total = np.float64(0.0)
    count = 0
    for row0 in range(0, source.shape[0], rows):
        row1 = min(row0 + rows, source.shape[0])
        total += np.sum(np.abs(restored[row0:row1] - source[row0:row1]), dtype=np.float64)
        count += (row1 - row0) * source.shape[1]
    return float(total / count)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quant_golden_checks() -> dict[str, Any]:
    rng = np.random.default_rng(0xE5CA)
    source = rng.standard_normal((3, 512), dtype=np.float32)
    checks = {}
    for name, quantize in (
        ("q2_k", oracle.quantize_q2_k),
        ("q4_k", oracle.quantize_q4_k),
        ("q6_k", oracle.quantize_q6_k),
    ):
        payload = quantize(source)
        digest = sha256_array(payload)
        checks[name] = {
            "sha256": digest,
            "expected_sha256": QUANT_GOLDENS[name],
            "byte_equal": digest == QUANT_GOLDENS[name],
        }
    return checks


def reconstruction_reference_checks() -> dict[str, Any]:
    sys.path.insert(0, str(REFERENCE_TOOLS))
    import escham_cpu

    rng = np.random.default_rng(0x11E5CA)
    checks = {}
    for k in (2, 3):
        code = rng.integers(-32768, 32768, size=(8, 8, 16 * k), dtype=np.int16)
        rin = rng.standard_normal(128, dtype=np.float32).astype(np.float16)
        rout = rng.standard_normal(128, dtype=np.float32).astype(np.float16)
        expected = escham_cpu.reconstruct_deploy_weight(
            code, rin, rout, 128, 128, k, True, False
        )
        actual = oracle.reconstruct_deploy_weight(
            code, rin, rout, 128, 128, k, True, False
        )
        checks[f"k{k}"] = {
            "reference_sha256": sha256_array(expected),
            "shared_sha256": sha256_array(actual),
            "bitwise_equal": bool(np.array_equal(expected, actual)),
        }
    return checks


def converter_golden_check() -> dict[str, Any]:
    # The converter imports safetensors for its CLI, but this golden only needs
    # its shared function bindings.  Keep the test runnable in a NumPy-only
    # environment by supplying a non-callable import stub when the package is
    # absent and reading the three fixture arrays from the simple file format.
    try:
        import safetensors  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("safetensors")
        stub.safe_open = None
        sys.modules["safetensors"] = stub
    converter = load_module(REPO / "convert_escha_to_gguf.py", "exp11_converter")
    function_identity = all((
        converter.quantize_q2_k is oracle.quantize_q2_k,
        converter.quantize_q4_k is oracle.quantize_q4_k,
        converter.quantize_q6_k is oracle.quantize_q6_k,
        converter.reconstruct_deploy_weight is oracle.reconstruct_deploy_weight,
    ))
    def find(name: str) -> np.ndarray:
        dtypes = {"F16": "<f2", "F32": "<f4", "I16": "<i2"}
        for path in sorted(CHECKPOINT.glob("*.safetensors")):
            with path.open("rb") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_size))
            if name not in header:
                continue
            info = header[name]
            start, end = info["data_offsets"]
            dtype = np.dtype(dtypes[info["dtype"]])
            expected_bytes = int(np.prod(info["shape"])) * dtype.itemsize
            if end - start != expected_bytes:
                raise ValueError(f"invalid safetensors span for {name}")
            return np.memmap(
                path, mode="r", dtype=dtype,
                offset=8 + header_size + start,
                shape=tuple(info["shape"]),
            )
        raise KeyError(name)

    prefix = "model.language_model.layers.0.linear_attn.in_proj_z"
    code = find(prefix + ".escha_code")
    rin = find(prefix + ".escha_rin")
    rout = find(prefix + ".escha_rout")
    weight = converter.reconstruct_deploy_weight(
        code, rin, rout, 5120, 6144, code.shape[2] // 16, True, False
    )
    source = np.ascontiguousarray(weight.T.astype(np.float32))
    del weight
    payload = converter.quantize_q2_k(source)
    result = {
        "case": "layer-0 --standard-gdn-gate-quant q2_k",
        "function_identity_shared": function_identity,
        "source_sha256": sha256_array(source),
        "expected_source_sha256": CONVERTER_GDN_Q2_GOLDEN["source_sha256"],
        "payload_sha256": sha256_array(payload),
        "expected_payload_sha256": CONVERTER_GDN_Q2_GOLDEN["payload_sha256"],
        "payload_bytes": payload.nbytes,
        "expected_payload_bytes": CONVERTER_GDN_Q2_GOLDEN["payload_bytes"],
    }
    result["byte_equal"] = all((
        result["function_identity_shared"],
        result["source_sha256"] == result["expected_source_sha256"],
        result["payload_sha256"] == result["expected_payload_sha256"],
        result["payload_bytes"] == result["expected_payload_bytes"],
    ))
    return result


def find_cache_entry(root: Path) -> Path:
    manifests = sorted(root.glob("*/*/manifest.json"))
    valid = [path.parent for path in manifests if (path.parent / "complete").is_file()]
    if len(valid) != 1:
        raise ValueError(f"expected one complete cache under {root}, found {len(valid)}")
    return valid[0]


def cache_oracle_checks(source: Path, cache_root: Path) -> list[dict[str, Any]]:
    entry_dir = find_cache_entry(cache_root)
    manifest = json.loads((entry_dir / "manifest.json").read_text(encoding="utf-8"))
    source_reader = gguf.GGUFReader(str(source), "r")
    source_tensors = {tensor.name: tensor for tensor in source_reader.tensors}
    overlay_reader = gguf.GGUFReader(str(entry_dir / "overlay.gguf"), "r")
    overlay_tensors = {tensor.name: tensor for tensor in overlay_reader.tensors}
    checks = []
    for entry in manifest["entries"]:
        key = entry["key"]
        role, layer = key["tensor_role"], int(key["layer"])
        ic, oc = FFN_DIMENSIONS[role]
        code = source_tensors[sidecar_tensor_name(layer, role, "escha_code")].data
        rin = source_tensors[sidecar_tensor_name(layer, role, "escha_rin")].data
        rout = source_tensors[sidecar_tensor_name(layer, role, "escha_rout")].data
        source_fp32 = oracle.reconstruct_standard_weight(code, rin, rout, ic, oc)
        expected = oracle.quantize_q2_k(source_fp32)
        cached = overlay_tensors[entry["output_tensor_name"]].data
        restored = dequantize(cached, GGMLQuantizationType.Q2_K)
        mae = mean_abs_error(restored, source_fp32)
        checks.append({
            "name": entry["output_tensor_name"],
            "payload_sha256": sha256_array(cached),
            "expected_payload_sha256": sha256_array(expected),
            "payload_bitwise_equal": bool(np.array_equal(cached, expected)),
            "dequantized_mae": mae,
            "manifest_mae": entry["source_fp32_mae"],
            "mae_equal_to_manifest": mae == entry["source_fp32_mae"],
        })
        del source_fp32, expected, restored
        gc.collect()
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("ffn",), required=True)
    parser.add_argument("--quant", choices=("q2_k",), required=True)
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "numpy_version": np.__version__,
        "quantizer_golden": quant_golden_checks(),
        "reconstruction_reference": reconstruction_reference_checks(),
        "converter_regression": converter_golden_check(),
        "cache_tensors": cache_oracle_checks(args.source.resolve(), args.cache_dir.resolve()),
    }
    failures = []
    failures.extend(
        f"quantizer {name} differs" for name, check in result["quantizer_golden"].items()
        if not check["byte_equal"]
    )
    failures.extend(
        f"reconstruction {name} differs" for name, check in result["reconstruction_reference"].items()
        if not check["bitwise_equal"]
    )
    if not result["converter_regression"]["byte_equal"]:
        failures.append("converter golden differs")
    if args.require_bitwise:
        failures.extend(
            f"cache tensor {check['name']} differs" for check in result["cache_tensors"]
            if not check["payload_bitwise_equal"]
        )
    result["status"] = "ok" if not failures else "failed"
    result["failures"] = failures
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
