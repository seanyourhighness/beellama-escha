#!/usr/bin/env python3
"""Produce the deterministic P-ARCH-01 K2/K3 packing evidence.

This is a correctness capture, not a benchmark.  It compares one production
16x16 payload at every representation boundary and writes compact, auditable
CSV/JSON evidence outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gguf-py"))

import escha  # noqa: E402,F401  Registers torch.ops.escha.
import gguf  # noqa: E402


PROJECTIONS = {
    2: {
        "gguf": "blk.0.ffn_gate.escha_code",
        "ckpt": "model.language_model.layers.0.mlp.gate_proj",
    },
    3: {
        "gguf": "blk.0.ffn_up.escha_code",
        "ckpt": "model.language_model.layers.0.mlp.up_proj",
    },
}
IC = 5120
OC = 17408
TILE = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor(reader: gguf.GGUFReader, name: str) -> np.ndarray:
    item = next(candidate for candidate in reader.tensors if candidate.name == name)
    return np.asarray(item.data)


def codebook_bits(indices: np.ndarray) -> np.ndarray:
    indices = indices.astype(np.uint32, copy=False)
    word = ((indices * np.uint32(0xCBAC1FED)) & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
    low = (word & np.uint32(0xFFFF)).astype("<u2").view(np.float16)
    high = (word >> np.uint32(16)).astype("<u2").view(np.float16)
    return (low.astype(np.float32) + high.astype(np.float32)).astype(np.float16).view("<u2")


def dep_pi(row: int) -> int:
    return (row & 1) | (((row >> 3) & 1) << 1) | (((row >> 1) & 3) << 3)


def mirror8(column: int) -> int:
    within = column & 7
    return column if within == 0 else (column & ~7) + (8 - within)


def decode_tile(payload_i16: np.ndarray, K: int):
    words = np.ascontiguousarray(payload_i16).view("<u4")
    nwords = 8 * K
    nbits = 256 * K
    indices = np.empty((TILE, TILE), dtype=np.uint16)  # [k, out]
    metadata: dict[tuple[int, int], dict[str, int]] = {}
    for k in range(TILE):
        for out in range(TILE):
            t = dep_pi(k) + 32 * out + 4 * (out >> 3)
            start_bit = ((32 - K) - K * t) % nbits
            group = start_bit >> 5
            word0 = nwords - group if group else 0
            word1 = word0 - 1 if word0 else nwords - 1
            shift = start_bit & 31
            pair = (int(words[word1]) << 32) | int(words[word0])
            index = (pair >> shift) & 0xFFFF
            indices[k, out] = index
            metadata[(k, out)] = {
                "t": t,
                "start_bit": start_bit,
                "word0": word0,
                "word1": word1,
                "shift": shift,
                "codebook_index": index,
            }
    return codebook_bits(indices), metadata


def expected_b_fragments(logical_bits: np.ndarray) -> np.ndarray:
    """Pack a logical W[k,out] tile according to Bee's tile<8,8,half2> ABI."""
    fragments = np.empty((2, 32, 2), dtype="<u4")
    for out_half in range(2):
        for lane in range(32):
            local_out = lane // 4
            half2_column_low = lane % 4
            for register in range(2):
                k0 = 2 * (register * 4 + half2_column_low)
                out = out_half * 8 + local_out
                lo = int(logical_bits[k0, out])
                hi = int(logical_bits[k0 + 1, out])
                fragments[out_half, lane, register] = lo | (hi << 16)
    return fragments


def bit_sources(meta: dict[str, int]) -> list[int]:
    sources = []
    for bit in range(16):
        offset = meta["shift"] + bit
        if offset < 32:
            sources.append(32 * meta["word0"] + offset)
        else:
            sources.append(32 * meta["word1"] + offset - 32)
    return sources


def find_checkpoint(shards: list[Path], name: str) -> np.ndarray:
    for shard in shards:
        with safe_open(str(shard), framework="np") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError(name)


def write_mapping(path: Path, K: int, logical_bits: np.ndarray, metadata, fragments: np.ndarray) -> None:
    fields = [
        "format", "dep_pi_k", "dep_row_direct", "dep_row_mirror8", "trellis_t", "packed_word0", "packed_byte0",
        "packed_word1", "packed_byte1", "start_bit", "shift",
        "packed_bit_indices_lsb_to_msb",
        "logical_k", "logical_out", "codebook_index", "logical_fp16_hex",
        "bee_fma_staging_index", "bee_mma_staging_index", "fragment_tile",
        "fragment_lane", "fragment_register", "fragment_half", "fragment_u32_hex",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for k in range(TILE):
            for out in range(TILE):
                meta = metadata[(k, out)]
                fragment_tile = out // 8
                lane = 4 * (out % 8) + ((k >> 1) & 3)
                register = k >> 3
                half = k & 1
                writer.writerow({
                    "format": f"K{K}",
                    "dep_pi_k": dep_pi(k),
                    "dep_row_direct": k * 16 + out,
                    "dep_row_mirror8": k * 16 + mirror8(out),
                    "trellis_t": meta["t"],
                    "packed_word0": meta["word0"],
                    "packed_byte0": 4 * meta["word0"],
                    "packed_word1": meta["word1"],
                    "packed_byte1": 4 * meta["word1"],
                    "start_bit": meta["start_bit"],
                    "shift": meta["shift"],
                    "packed_bit_indices_lsb_to_msb": ",".join(map(str, bit_sources(meta))),
                    "logical_k": k,
                    "logical_out": out,
                    "codebook_index": meta["codebook_index"],
                    "logical_fp16_hex": f"0x{int(logical_bits[k, out]):04x}",
                    "bee_fma_staging_index": k * 128 + out,
                    "bee_mma_staging_index": out * 16 + k,
                    "fragment_tile": fragment_tile,
                    "fragment_lane": lane,
                    "fragment_register": register,
                    "fragment_half": half,
                    "fragment_u32_hex": f"0x{int(fragments[fragment_tile, lane, register]):08x}",
                })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    args = parser.parse_args()
    out = args.capture_dir
    reader = gguf.GGUFReader(str(args.gguf))
    summary: dict[str, object] = {
        "question": "Did this make the data presented to the kernel match the control?",
        "gguf": {"path": str(args.gguf), "sha256": sha256(args.gguf)},
        "tile_coordinate": [0, 0],
        "geometry": {"IC": IC, "OC": OC, "tile": [16, 16], "activation": [512, IC]},
        "formats": {},
    }

    activation_hashes: dict[int, str] = {}
    for K, names in PROJECTIONS.items():
        prefix = f"k{K}"
        payload_path = out / f"{prefix}_tile_0_0_payload.i16.bin"
        dep_path = out / f"{prefix}_dep.i16.bin"
        rin_path = out / f"{prefix}_rin.f16.bin"
        rout_path = out / f"{prefix}_rout.f16.bin"
        activation_path = out / f"{prefix}_activation_512x5120.f32.bin"
        fragment_path = out / f"{prefix}_bee_b_fragments.u32.bin"
        paths = (payload_path, dep_path, rin_path, rout_path, activation_path, fragment_path)
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)

        payload = np.fromfile(payload_path, dtype="<i2")
        captured_fragments = np.fromfile(fragment_path, dtype="<u4").reshape(2, 32, 2)
        gguf_code = tensor(reader, names["gguf"])
        gguf_payload = np.ascontiguousarray(gguf_code.reshape(-1, 16 * K)[0])
        ckpt_code = find_checkpoint(args.checkpoint, names["ckpt"] + ".escha_code")
        ckpt_payload = np.ascontiguousarray(ckpt_code[0, 0, :])

        logical_bits, metadata = decode_tile(payload, K)
        reference = torch.ops.escha.escham_reconstruct(
            torch.from_numpy(np.ascontiguousarray(ckpt_code)).cuda(), IC, OC, K, True, False
        )
        if tuple(reference.shape) == (IC, OC):
            reference_tile = reference[:TILE, :TILE]
        elif tuple(reference.shape) == (OC, IC):
            reference_tile = reference[:TILE, :TILE].T
        else:
            raise AssertionError(f"unexpected reconstruct shape: {tuple(reference.shape)}")
        reference_bits = reference_tile.contiguous().cpu().numpy().astype(np.float16, copy=False).view("<u2")
        expected_fragments = expected_b_fragments(reference_bits)
        del reference
        torch.cuda.empty_cache()

        # Invoke the production reference prefill op exactly once.  Retain only
        # the selected CUDA kernel name and an output content hash; timings are
        # intentionally excluded because P-ARCH-01 is not a benchmark.
        activation = np.fromfile(activation_path, dtype="<f4").reshape(512, IC)
        rin_ckpt = find_checkpoint(args.checkpoint, names["ckpt"] + ".escha_rin")
        rout_ckpt = find_checkpoint(args.checkpoint, names["ckpt"] + ".escha_rout")
        s_in_ckpt = find_checkpoint(args.checkpoint, names["ckpt"] + ".escha_s_in")
        s_out_ckpt = find_checkpoint(args.checkpoint, names["ckpt"] + ".escha_s_out")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            reference_output = torch.ops.escha.escham_code_gemm(
                torch.from_numpy(activation).cuda().half().contiguous(),
                torch.from_numpy(np.ascontiguousarray(ckpt_code)).cuda(),
                torch.from_numpy(np.ascontiguousarray(rin_ckpt)).cuda().half().contiguous(),
                torch.from_numpy(np.ascontiguousarray(rout_ckpt)).cuda().half().contiguous(),
                torch.from_numpy(np.ascontiguousarray(s_in_ckpt)).cuda().contiguous(),
                torch.from_numpy(np.ascontiguousarray(s_out_ckpt)).cuda().contiguous(),
                OC, K, True, False, 1,
            )
            torch.cuda.synchronize()
        reference_kernel_names = sorted({
            event.name for event in profiler.events()
            if "escham_code_gemm_kernel" in event.name
        })
        reference_output_bits = reference_output.contiguous().cpu().numpy().view(np.uint16)
        reference_output_sha256 = array_sha256(reference_output_bits)
        del reference_output
        torch.cuda.empty_cache()

        dep_gguf = tensor(reader, f"escha_dep_k{K}")
        rin_gguf = tensor(reader, names["gguf"].replace("escha_code", "escha_rin"))
        rout_gguf = tensor(reader, names["gguf"].replace("escha_code", "escha_rout"))
        dep_capture = np.fromfile(dep_path, dtype="<i2")
        rin_capture = np.fromfile(rin_path, dtype="<f2")
        rout_capture = np.fromfile(rout_path, dtype="<f2")

        payload_checkpoint_equal = np.array_equal(payload, ckpt_payload)
        payload_gguf_equal = np.array_equal(payload, gguf_payload)
        logical_reference_equal = np.array_equal(logical_bits, reference_bits)
        fragment_reference_equal = np.array_equal(captured_fragments, expected_fragments)
        dep_equal = np.array_equal(dep_capture, np.ascontiguousarray(dep_gguf).reshape(-1))
        dep_matrix = dep_capture.reshape(256, 16)
        dep_direct_equal = all(
            np.array_equal(dep_matrix[k * 16 + out], np.asarray(bit_sources(metadata[(k, out)]), dtype=np.int16))
            for k in range(16) for out in range(16)
        )
        dep_mirror8_equal = all(
            np.array_equal(dep_matrix[k * 16 + mirror8(out)],
                           np.asarray(bit_sources(metadata[(k, out)]), dtype=np.int16))
            for k in range(16) for out in range(16)
        )
        dep_mapping = "direct-k-major-out-minor" if dep_direct_equal else (
            "k-major-mirror8(out)-minor" if dep_mirror8_equal else "unrecognized")
        rin_equal = np.array_equal(rin_capture.view("<u2"), np.ascontiguousarray(rin_gguf).reshape(-1).view("<u2"))
        rout_equal = np.array_equal(rout_capture.view("<u2"), np.ascontiguousarray(rout_gguf).reshape(-1).view("<u2"))
        rin_checkpoint_folded = (
            rin_ckpt.astype(np.float32) * s_in_ckpt.astype(np.float32)).astype(np.float16)
        rout_checkpoint_folded = (
            rout_ckpt.astype(np.float32) * s_out_ckpt.astype(np.float32)).astype(np.float16)
        rin_checkpoint_equal = np.array_equal(
            rin_capture.view("<u2"), np.ascontiguousarray(rin_checkpoint_folded).reshape(-1).view("<u2"))
        rout_checkpoint_equal = np.array_equal(
            rout_capture.view("<u2"), np.ascontiguousarray(rout_checkpoint_folded).reshape(-1).view("<u2"))

        mapping_path = out / f"{prefix}_packed_logical_fragment_mapping.csv"
        write_mapping(mapping_path, K, logical_bits, metadata, captured_fragments)
        activation_hashes[K] = sha256(activation_path)
        checks = {
            "captured_payload_equals_checkpoint": payload_checkpoint_equal,
            "captured_payload_equals_gguf": payload_gguf_equal,
            "independent_logical_decode_equals_escha_reconstruct_fp16_bits": logical_reference_equal,
            "captured_bee_ldmatrix_fragments_equal_expected_hmma_b_abi": fragment_reference_equal,
            "captured_dep_equals_gguf": dep_equal,
            "captured_dep_bit_sources_have_recognized_closed_form_mapping": dep_mapping != "unrecognized",
            "captured_rin_equals_gguf_fp16_bits": rin_equal,
            "captured_rout_equals_gguf_fp16_bits": rout_equal,
            "captured_rin_equals_checkpoint_rin_times_s_in_fp16_bits": rin_checkpoint_equal,
            "captured_rout_equals_checkpoint_rout_times_s_out_fp16_bits": rout_checkpoint_equal,
        }
        summary["formats"][f"K{K}"] = {
            "tensor": names["gguf"],
            "checkpoint_tensor": names["ckpt"] + ".escha_code",
            "checkpoint_code_shape": list(ckpt_code.shape),
            "reference_reconstruct_shape": list(reference_bits.shape),
            "reference_code_gemm_kernel_names": reference_kernel_names,
            "reference_code_gemm_output_fp16_bits_sha256": reference_output_sha256,
            "checks": checks,
            "captured_dep_logical_row_mapping": dep_mapping,
            "all_packing_checks_pass": all(checks.values()),
            "files": {path.name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in paths},
            "mapping": {"path": str(mapping_path), "sha256": sha256(mapping_path), "rows": 256},
            "logical_fp16_bits_sha256": array_sha256(logical_bits),
            "reference_fp16_bits_sha256": array_sha256(reference_bits),
            "expected_fragment_u32_sha256": array_sha256(expected_fragments),
        }

    summary["activation_captures_byte_identical"] = activation_hashes[2] == activation_hashes[3]
    summary["answer"] = all(item["all_packing_checks_pass"] for item in summary["formats"].values())
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["answer"]:
        raise SystemExit("P-ARCH-01 mismatch: see summary.json")


if __name__ == "__main__":
    main()
