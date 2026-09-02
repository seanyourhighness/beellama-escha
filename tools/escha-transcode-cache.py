#!/usr/bin/env python3
"""EXP-11 Attempt-1, Slice-1 one-layer Escha FFN overlay cache tool."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gguf-py"))
sys.path.insert(0, str(REPO / "conversion" / "escha"))

import gguf
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

from transcode_cache_schema import (
    CACHE_FORMAT_VERSION,
    COMPLETE_FILENAME,
    FFN_DIMENSIONS,
    FFN_MASK_SUFFIXES,
    FFN_ROLES,
    LOCK_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    OVERLAY_FILENAME,
    OVERLAY_SHA_FILENAME,
    TENSOR_LAYOUT_VERSION,
    EntryKey,
    Manifest,
    ManifestEntry,
    RecipeSpec,
    canonical_json_bytes,
    entry_directory,
    sidecar_tensor_name,
    standard_tensor_name,
)
from transcode_oracle import (
    ORACLE_ABI,
    QUANTIZER_ABI,
    quantize_q2_k,
    reconstruct_standard_weight,
)

CANONICAL_SOURCE = Path(
    "/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/"
    "escha-w2-lowgpu-mono-parity.gguf"
)
CANONICAL_SOURCE_SHA256 = (
    "e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d"
)
SUPPORTED_QUANT = "q2_k"
QTYPE = GGMLQuantizationType.Q2_K


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value).view(np.uint8)
    return hashlib.sha256(memoryview(array)).hexdigest()


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        first_text, last_text = value.split("..", 1)
        first, last = int(first_text), int(last_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("layers must use FIRST..LAST syntax") from error
    if first < 0 or last < first:
        raise argparse.ArgumentTypeError("invalid layer range")
    layers = tuple(range(first, last + 1))
    if layers != (0,):
        raise argparse.ArgumentTypeError(
            "EXP-11 Attempt-1 Slice 1 is intentionally limited to --layers 0..0"
        )
    return layers


def parse_quant(value: str) -> tuple[tuple[str, str], ...]:
    parsed: dict[str, str] = {}
    for item in value.split(","):
        try:
            role, quant = item.split("=", 1)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid quant item: {item}") from error
        if role in parsed:
            raise argparse.ArgumentTypeError(f"duplicate quant role: {role}")
        parsed[role] = quant
    if tuple(parsed) != FFN_ROLES:
        raise argparse.ArgumentTypeError(
            "quant roles must be ordered ffn_gate,ffn_up,ffn_down"
        )
    if any(value != SUPPORTED_QUANT for value in parsed.values()):
        raise argparse.ArgumentTypeError("Slice 1 supports q2_k only")
    return tuple(parsed.items())


def require_canonical_source(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != CANONICAL_SOURCE.resolve():
        raise ValueError(
            "Slice 1 accepts only the funded canonical GGUF: " + str(CANONICAL_SOURCE)
        )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def field_value(reader: gguf.GGUFReader, name: str) -> Any:
    field = reader.get_field(name)
    if field is None:
        raise ValueError(f"missing canonical metadata: {name}")
    return field.contents()


def validate_canonical_metadata(reader: gguf.GGUFReader) -> None:
    architecture = field_value(reader, "general.architecture")
    escha_version = int(field_value(reader, "qwen35.escha.version"))
    if architecture != "qwen35" or escha_version != 1:
        raise ValueError(
            f"ineligible source metadata: architecture={architecture!r}, "
            f"escha_version={escha_version}"
        )


def oracle_module_sha256() -> str:
    module_path = REPO / "conversion" / "escha" / "transcode_oracle.py"
    return sha256_file(module_path)


def make_recipe(source_sha256: str, quant: tuple[tuple[str, str], ...]) -> RecipeSpec:
    return RecipeSpec(
        format_version=CACHE_FORMAT_VERSION,
        source_sha256=source_sha256,
        architecture="qwen35",
        escha_version=1,
        scope="ffn",
        ordered_roles=FFN_ROLES,
        target_quant=quant,
        oracle_module_sha256=oracle_module_sha256(),
        numpy_version=np.__version__,
        quantizer_abi=QUANTIZER_ABI,
        endianness=sys.byteorder,
        tensor_layout_version=TENSOR_LAYOUT_VERSION,
    )


def mean_abs_error(restored: np.ndarray, source: np.ndarray, rows: int = 128) -> float:
    total = np.float64(0.0)
    count = 0
    for row0 in range(0, source.shape[0], rows):
        row1 = min(row0 + rows, source.shape[0])
        total += np.sum(
            np.abs(restored[row0:row1] - source[row0:row1]), dtype=np.float64
        )
        count += (row1 - row0) * source.shape[1]
    return float(total / count)


def tensor_map(reader: gguf.GGUFReader) -> dict[str, Any]:
    return {tensor.name: tensor for tensor in reader.tensors}


def validate_source_triplet(tensors: dict[str, Any], layer: int, role: str) -> tuple[Any, Any, Any, int, int, int]:
    names = [sidecar_tensor_name(layer, role, suffix) for suffix in ("escha_code", "escha_rin", "escha_rout")]
    missing = [name for name in names if name not in tensors]
    if missing:
        raise ValueError("missing source sidecars: " + ", ".join(missing))
    code, rin, rout = (tensors[name] for name in names)
    ic, oc = FFN_DIMENSIONS[role]
    expected_code_prefix = (ic // 16, oc // 16)
    if code.data.shape[:2] != expected_code_prefix or code.data.shape[2] not in (32, 48):
        raise ValueError(f"invalid {names[0]} data shape: {code.data.shape}")
    if rin.data.shape != (ic,) or rout.data.shape != (oc,):
        raise ValueError(f"invalid scale shapes for layer {layer} {role}")
    return code, rin, rout, code.data.shape[2] // 16, ic, oc


def write_overlay(path: Path, products: list[dict[str, Any]]) -> None:
    writer = gguf.GGUFWriter(path, "qwen35", use_temp_file=True)
    for product in products:
        writer.add_tensor(
            product["name"], product["payload"],
            raw_shape=product["payload"].shape, raw_dtype=QTYPE,
        )
    writer.write_header_to_file(path=path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    fsync_file(path)


def validate_overlay_file(path: Path, products: list[dict[str, Any]]) -> dict[str, Any]:
    reader = gguf.GGUFReader(str(path), "r")
    actual = tensor_map(reader)
    expected_names = [product["name"] for product in products]
    if list(actual) != expected_names:
        raise ValueError(f"overlay allowlist/order mismatch: {list(actual)} != {expected_names}")
    result: dict[str, Any] = {}
    for product in products:
        tensor = actual[product["name"]]
        if tensor.tensor_type != QTYPE:
            raise ValueError(f"wrong qtype for {tensor.name}: {tensor.tensor_type}")
        if tuple(int(item) for item in tensor.shape) != (product["ic"], product["oc"]):
            raise ValueError(f"wrong logical shape for {tensor.name}: {tensor.shape}")
        payload_sha = sha256_array(tensor.data)
        if tensor.n_bytes != product["payload"].nbytes or payload_sha != product["payload_sha256"]:
            raise ValueError(f"payload mismatch for {tensor.name}")
        result[tensor.name] = {
            "data_offset": tensor.data_offset,
            "byte_count": tensor.n_bytes,
            "payload_sha256": payload_sha,
        }
    return result


def prepare(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    source = require_canonical_source(args.source)
    if args.scope != "ffn" or args.jobs != 1:
        raise ValueError("Slice 1 requires --scope ffn --jobs 1")

    hash_started = time.perf_counter()
    source_sha = sha256_file(source)
    source_hash_seconds = time.perf_counter() - hash_started
    if source_sha != CANONICAL_SOURCE_SHA256:
        raise ValueError(
            f"canonical source SHA mismatch: got {source_sha}, expected {CANONICAL_SOURCE_SHA256}"
        )
    recipe = make_recipe(source_sha, args.quant)
    recipe_id = recipe.recipe_id()
    cache_entry = entry_directory(args.cache_dir.resolve(), source_sha, recipe_id)
    cache_entry.mkdir(parents=True, exist_ok=True)

    lock_path = cache_entry / LOCK_FILENAME
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        complete = cache_entry / COMPLETE_FILENAME
        if complete.exists():
            raise FileExistsError(
                f"complete cache already exists at {cache_entry}; use a distinct probe root"
            )

        reader_started = time.perf_counter()
        source_reader = gguf.GGUFReader(str(source), "r")
        validate_canonical_metadata(source_reader)
        source_tensors = tensor_map(source_reader)
        reader_seconds = time.perf_counter() - reader_started

        products: list[dict[str, Any]] = []
        tensor_reports: list[dict[str, Any]] = []
        reconstruction_seconds = 0.0
        quantization_seconds = 0.0
        validation_seconds = 0.0

        for layer in args.layers:
            for role, target_quant in args.quant:
                code, rin, rout, k, ic, oc = validate_source_triplet(
                    source_tensors, layer, role
                )
                code_sha = sha256_array(code.data)
                rin_sha = sha256_array(rin.data)
                rout_sha = sha256_array(rout.data)

                phase = time.perf_counter()
                fp32 = reconstruct_standard_weight(code.data, rin.data, rout.data, ic, oc)
                reconstruct_elapsed = time.perf_counter() - phase
                reconstruction_seconds += reconstruct_elapsed
                fp32_sha = sha256_array(fp32)

                phase = time.perf_counter()
                payload = quantize_q2_k(fp32)
                quantize_elapsed = time.perf_counter() - phase
                quantization_seconds += quantize_elapsed
                payload_sha = sha256_array(payload)

                phase = time.perf_counter()
                restored = dequantize(payload, QTYPE)
                mae = mean_abs_error(restored, fp32)
                del restored
                validation_elapsed = time.perf_counter() - phase
                validation_seconds += validation_elapsed

                product = {
                    "layer": layer,
                    "role": role,
                    "target_quant": target_quant,
                    "name": standard_tensor_name(layer, role),
                    "k": k,
                    "ic": ic,
                    "oc": oc,
                    "code_sha256": code_sha,
                    "rin_sha256": rin_sha,
                    "rout_sha256": rout_sha,
                    "source_fp32_sha256": fp32_sha,
                    "payload": payload,
                    "payload_sha256": payload_sha,
                    "mae": mae,
                }
                products.append(product)
                tensor_reports.append({
                    "name": product["name"],
                    "k": k,
                    "logical_shape_oc_ic": [oc, ic],
                    "source_fp32_bytes": fp32.nbytes,
                    "payload_bytes": payload.nbytes,
                    "source_fp32_sha256": fp32_sha,
                    "payload_sha256": payload_sha,
                    "dequantized_mae": mae,
                    "reconstruction_seconds": reconstruct_elapsed,
                    "quantization_seconds": quantize_elapsed,
                    "validation_seconds": validation_elapsed,
                    "oracle_payload_byte_equal": True,
                })
                del fp32
                gc.collect()

        pid = os.getpid()
        overlay_tmp = cache_entry / f"{OVERLAY_FILENAME}.tmp.{pid}"
        manifest_tmp = cache_entry / f"{MANIFEST_FILENAME}.tmp.{pid}"
        sha_tmp = cache_entry / f"{OVERLAY_SHA_FILENAME}.tmp.{pid}"
        complete_tmp = cache_entry / f"{COMPLETE_FILENAME}.tmp.{pid}"

        write_started = time.perf_counter()
        write_overlay(overlay_tmp, products)
        overlay_sha = sha256_file(overlay_tmp)
        overlay_size = overlay_tmp.stat().st_size
        validated = validate_overlay_file(overlay_tmp, products)
        write_seconds = time.perf_counter() - write_started

        entries: list[ManifestEntry] = []
        for product in products:
            overlay_info = validated[product["name"]]
            key = EntryKey(
                source_gguf_sha256=source_sha,
                layer=product["layer"],
                tensor_role=product["role"],
                source_code_sha256=product["code_sha256"],
                source_rin_sha256=product["rin_sha256"],
                source_rout_sha256=product["rout_sha256"],
                k=product["k"], ic=product["ic"], oc=product["oc"],
                target_quant=product["target_quant"],
                oracle_abi=ORACLE_ABI,
                layout_version=TENSOR_LAYOUT_VERSION,
            )
            entries.append(ManifestEntry(
                key=key,
                output_tensor_name=product["name"],
                ggml_type=QTYPE.name,
                shape=(product["oc"], product["ic"]),
                byte_count=overlay_info["byte_count"],
                data_offset=overlay_info["data_offset"],
                payload_sha256=overlay_info["payload_sha256"],
                source_fp32_sha256=product["source_fp32_sha256"],
                source_fp32_mae=product["mae"],
            ))

        scalable_layer_seconds = (
            reconstruction_seconds + quantization_seconds + validation_seconds + write_seconds
        )
        projected_all64_seconds = source_hash_seconds + reader_seconds + scalable_layer_seconds * 64
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest = Manifest(
            manifest_schema=MANIFEST_SCHEMA,
            cache_format_version=CACHE_FORMAT_VERSION,
            source_path=str(source), source_sha256=source_sha,
            source_size=source.stat().st_size,
            recipe_id=recipe_id, recipe=recipe.to_dict(), scope="ffn",
            layers=args.layers,
            masking={
                "atomic_family": "per-layer ffn_gate+ffn_up+ffn_down",
                "shadow_standard_names": [product["name"] for product in products],
                "exclude_sidecar_suffixes": list(FFN_MASK_SUFFIXES),
                "bias_correction_applied": False,
            },
            overlay_filename=OVERLAY_FILENAME, overlay_sha256=overlay_sha,
            overlay_size=overlay_size, entries=tuple(entries),
            timing={
                "source_hash_seconds": source_hash_seconds,
                "reader_seconds": reader_seconds,
                "reconstruction_seconds": reconstruction_seconds,
                "quantization_seconds": quantization_seconds,
                "dequant_validation_seconds": validation_seconds,
                "overlay_write_hash_validate_seconds": write_seconds,
                "projected_all64_seconds": projected_all64_seconds,
            },
            resources={
                "max_rss_kb_internal": peak_rss_kb,
                "projected_all64_max_rss_kb": peak_rss_kb,
                "attempt1_target_max_rss_gb": 2.5,
                "attempt1_hard_max_rss_gb": 3.0,
                "attempt1_hard_wall_seconds": 120.0,
            },
            generator={
                "tool": "tools/escha-transcode-cache.py",
                "slice": "EXP-11 ATTEMPT-1 SLICE-1",
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "oracle_module_sha256": recipe.oracle_module_sha256,
            },
        )
        write_fsync(manifest_tmp, canonical_json_bytes(manifest.to_dict()))
        write_fsync(sha_tmp, f"{overlay_sha}  {OVERLAY_FILENAME}\n".encode("ascii"))

        os.replace(overlay_tmp, cache_entry / OVERLAY_FILENAME)
        os.replace(manifest_tmp, cache_entry / MANIFEST_FILENAME)
        os.replace(sha_tmp, cache_entry / OVERLAY_SHA_FILENAME)
        write_fsync(complete_tmp, b"")
        os.replace(complete_tmp, complete)
        fsync_directory(cache_entry)

    elapsed = time.perf_counter() - started
    report = {
        "command": "prepare",
        "status": "complete",
        "cache_entry": str(cache_entry),
        "source_sha256": source_sha,
        "recipe_id": recipe_id,
        "overlay_sha256": overlay_sha,
        "overlay_bytes": overlay_size,
        "wall_seconds_internal": elapsed,
        "max_rss_kb_internal": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "tensor_reports": tensor_reports,
        "projection": {
            "all64_wall_seconds": projected_all64_seconds,
            "all64_max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "wall_budget_seconds": 120.0,
            "rss_target_gb": 2.5,
            "rss_hard_gb": 3.0,
            "wall_budget_holds": projected_all64_seconds <= 120.0,
            "rss_target_holds": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 2.5 * 1024 * 1024,
            "rss_hard_holds": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 3.0 * 1024 * 1024,
            "attempt2_trigger": projected_all64_seconds > 120.0 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 3.0 * 1024 * 1024,
        },
    }
    write_fsync(args.report, json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def load_single_complete(cache_root: Path) -> tuple[Path, dict[str, Any]]:
    manifests = sorted(cache_root.glob(f"*/*/{MANIFEST_FILENAME}"))
    valid = [path for path in manifests if (path.parent / COMPLETE_FILENAME).is_file()]
    if len(valid) != 1:
        raise ValueError(f"expected exactly one complete cache entry under {cache_root}, found {len(valid)}")
    return valid[0].parent, json.loads(valid[0].read_text(encoding="utf-8"))


def verify(args: argparse.Namespace) -> int:
    source = require_canonical_source(args.source)
    source_sha = sha256_file(source)
    entry_dir, manifest = load_single_complete(args.cache_dir.resolve())
    errors: list[str] = []
    if args.require_complete and not (entry_dir / COMPLETE_FILENAME).is_file():
        errors.append("missing complete marker")
    if manifest.get("manifest_schema") != MANIFEST_SCHEMA:
        errors.append("manifest schema mismatch")
    if manifest.get("source_sha256") != source_sha:
        errors.append("source SHA mismatch")
    overlay = entry_dir / OVERLAY_FILENAME
    actual_overlay_sha = sha256_file(overlay)
    if actual_overlay_sha != manifest.get("overlay_sha256"):
        errors.append("whole-overlay SHA mismatch")
    sha_line = (entry_dir / OVERLAY_SHA_FILENAME).read_text(encoding="ascii").strip()
    if sha_line != f"{actual_overlay_sha}  {OVERLAY_FILENAME}":
        errors.append("overlay.sha256 mismatch")

    source_reader = gguf.GGUFReader(str(source), "r")
    validate_canonical_metadata(source_reader)
    source_tensors = tensor_map(source_reader)
    overlay_reader = gguf.GGUFReader(str(overlay), "r")
    overlay_tensors = tensor_map(overlay_reader)
    expected_names = [entry["output_tensor_name"] for entry in manifest["entries"]]
    if list(overlay_tensors) != expected_names:
        errors.append("overlay tensor allowlist/order mismatch")

    checks = []
    for raw_entry in manifest["entries"]:
        key = raw_entry["key"]
        role, layer = key["tensor_role"], int(key["layer"])
        code, rin, rout, k, ic, oc = validate_source_triplet(source_tensors, layer, role)
        tensor = overlay_tensors.get(raw_entry["output_tensor_name"])
        if tensor is None:
            errors.append(f"missing overlay tensor {raw_entry['output_tensor_name']}")
            continue
        local_errors = []
        component_hashes = {
            "source_code_sha256": sha256_array(code.data),
            "source_rin_sha256": sha256_array(rin.data),
            "source_rout_sha256": sha256_array(rout.data),
        }
        for field, value in component_hashes.items():
            if key.get(field) != value:
                local_errors.append(field + " mismatch")
        payload_sha = sha256_array(tensor.data)
        if payload_sha != raw_entry["payload_sha256"]:
            local_errors.append("payload SHA mismatch")

        mae = None
        oracle_equal = None
        if args.deep:
            fp32 = reconstruct_standard_weight(code.data, rin.data, rout.data, ic, oc)
            oracle_payload = quantize_q2_k(fp32)
            oracle_equal = bool(np.array_equal(oracle_payload, tensor.data))
            if not oracle_equal:
                local_errors.append("oracle payload bytes differ")
            restored = dequantize(tensor.data, QTYPE)
            mae = mean_abs_error(restored, fp32)
            del fp32, oracle_payload, restored
            gc.collect()
        errors.extend(f"{tensor.name}: {item}" for item in local_errors)
        checks.append({
            "name": tensor.name,
            "k": k,
            "payload_sha256": payload_sha,
            "oracle_payload_byte_equal": oracle_equal,
            "dequantized_mae": mae,
            "errors": local_errors,
        })

    report = {
        "command": "verify", "status": "ok" if not errors else "failed",
        "cache_entry": str(entry_dir), "overlay_sha256": actual_overlay_sha,
        "deep": bool(args.deep), "checks": checks, "errors": errors,
    }
    write_fsync(args.report, json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


def compare(args: argparse.Namespace) -> int:
    entry_a, manifest_a = load_single_complete(args.cache_a.resolve())
    entry_b, manifest_b = load_single_complete(args.cache_b.resolve())
    overlay_a = sha256_file(entry_a / OVERLAY_FILENAME)
    overlay_b = sha256_file(entry_b / OVERLAY_FILENAME)
    entries_a = {item["output_tensor_name"]: item["payload_sha256"] for item in manifest_a["entries"]}
    entries_b = {item["output_tensor_name"]: item["payload_sha256"] for item in manifest_b["entries"]}
    overlay_equal = overlay_a == overlay_b
    entries_equal = entries_a == entries_b
    errors = []
    if args.require_overlay_sha_equal and not overlay_equal:
        errors.append("overlay SHA values differ")
    if args.require_entry_sha_equal and not entries_equal:
        errors.append("entry SHA maps differ")
    report = {
        "command": "compare", "status": "ok" if not errors else "failed",
        "overlay_sha_a": overlay_a, "overlay_sha_b": overlay_b,
        "overlay_sha_equal": overlay_equal, "entry_sha_equal": entries_equal,
        "entry_sha_a": entries_a, "entry_sha_b": entries_b, "errors": errors,
    }
    write_fsync(args.report, json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, default=CANONICAL_SOURCE)
    prep.add_argument("--cache-dir", type=Path, required=True)
    prep.add_argument("--scope", choices=("ffn",), required=True)
    prep.add_argument("--layers", type=parse_layers, default=parse_layers("0..0"))
    prep.add_argument("--quant", type=parse_quant, required=True)
    prep.add_argument("--jobs", type=int, choices=(1,), required=True)
    prep.add_argument("--report", type=Path, required=True)
    prep.set_defaults(run=prepare)

    check = sub.add_parser("verify")
    check.add_argument("--source", type=Path, default=CANONICAL_SOURCE)
    check.add_argument("--cache-dir", type=Path, required=True)
    check.add_argument("--scope", choices=("ffn",), required=True)
    check.add_argument("--deep", action="store_true")
    check.add_argument("--require-complete", action="store_true")
    check.add_argument("--report", type=Path, required=True)
    check.set_defaults(run=verify)

    diff = sub.add_parser("compare")
    diff.add_argument("--cache-a", type=Path, required=True)
    diff.add_argument("--cache-b", type=Path, required=True)
    diff.add_argument("--require-overlay-sha-equal", action="store_true")
    diff.add_argument("--require-entry-sha-equal", action="store_true")
    diff.add_argument("--report", type=Path, required=True)
    diff.set_defaults(run=compare)
    return root


def main() -> int:
    args = parser().parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
