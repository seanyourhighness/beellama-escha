#!/usr/bin/env python3
"""Repack blk.0.ffn_gate K2 into the descriptor-free V3 BK64 layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gguf-py"))
import gguf  # noqa: E402

MAGIC = b"ESCHA-MMA-V3\0\0\0\0"
VERSION = 3
HEADER_BYTES = 256
K = 2
IC = 5120
OC = 17408
K_STAGE = 16
STAGES_PER_SUPER = 4
CTA_N = 128
BANDS = 2
TILES_PER_BAND = 4
WORDS_PER_TILE = 16
N_CTA = OC // CTA_N
N_STAGE = IC // K_STAGE
N_SUPER = N_STAGE // STAGES_PER_SUPER
SUPER_WORDS = BANDS * STAGES_PER_SUPER * TILES_PER_BAND * WORDS_PER_TILE
SUPER_BYTES = SUPER_WORDS * 4
N_RECORDS = N_CTA * N_SUPER
PAYLOAD_OFFSET = HEADER_BYTES
PAYLOAD_BYTES = N_RECORDS * SUPER_BYTES
TOTAL_BYTES = PAYLOAD_OFFSET + PAYLOAD_BYTES
TENSOR = "blk.0.ffn_gate"


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_map(reader: gguf.GGUFReader) -> dict[str, object]:
    return {tensor.name: tensor for tensor in reader.tensors}


def metadata(reader: gguf.GGUFReader, name: str):
    field = reader.get_field(name)
    if field is None:
        raise ValueError(f"missing metadata {name}")
    return field.contents()


def make_header(code_sha: str, rin_sha: str, rout_sha: str) -> bytes:
    values = (
        VERSION, HEADER_BYTES, K, IC, OC, K_STAGE, STAGES_PER_SUPER,
        CTA_N, BANDS, TILES_PER_BAND, WORDS_PER_TILE, N_CTA, N_STAGE,
        N_SUPER, SUPER_BYTES,
    )
    prefix = struct.pack(
        "<16s15I4Q", MAGIC, *values,
        0, PAYLOAD_OFFSET, PAYLOAD_BYTES, TOTAL_BYTES,
    )
    hashes = b"".join(bytes.fromhex(x) for x in (code_sha, rin_sha, rout_sha))
    header = prefix + hashes
    if len(header) > HEADER_BYTES:
        raise AssertionError(len(header))
    return header.ljust(HEADER_BYTES, b"\0")


def iter_super_records(code: np.ndarray):
    words = np.ascontiguousarray(code).view("<u4").reshape(N_STAGE, OC // 16, WORDS_PER_TILE)
    for cta in range(N_CTA):
        for super_stage in range(N_SUPER):
            record = np.empty(
                (BANDS, STAGES_PER_SUPER, TILES_PER_BAND, WORDS_PER_TILE),
                dtype="<u4",
            )
            for band in range(BANDS):
                for stage4 in range(STAGES_PER_SUPER):
                    stage = super_stage * STAGES_PER_SUPER + stage4
                    first_tile = cta * 8 + band * TILES_PER_BAND
                    record[band, stage4] = words[
                        stage, first_tile:first_tile + TILES_PER_BAND
                    ]
            yield record.tobytes()


def reverse_payload(payload: memoryview) -> np.ndarray:
    restored = np.empty((N_STAGE, OC // 16, WORDS_PER_TILE), dtype="<u4")
    pos = 0
    for cta in range(N_CTA):
        for super_stage in range(N_SUPER):
            record = np.frombuffer(
                payload[pos:pos + SUPER_BYTES], dtype="<u4"
            ).reshape(BANDS, STAGES_PER_SUPER, TILES_PER_BAND, WORDS_PER_TILE)
            pos += SUPER_BYTES
            for band in range(BANDS):
                for stage4 in range(STAGES_PER_SUPER):
                    stage = super_stage * STAGES_PER_SUPER + stage4
                    first_tile = cta * 8 + band * TILES_PER_BAND
                    restored[stage, first_tile:first_tile + TILES_PER_BAND] = record[band, stage4]
    if pos != len(payload):
        raise AssertionError((pos, len(payload)))
    return restored.view("<i2").reshape(N_STAGE, OC // 16, 32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    reader = gguf.GGUFReader(str(args.source), "r")
    if metadata(reader, "general.architecture") != "qwen35" or int(
        metadata(reader, "qwen35.escha.version")
    ) != 1:
        raise ValueError("source is not metadata-gated qwen35 Escha v1")
    tensors = tensor_map(reader)
    names = [f"{TENSOR}.escha_{suffix}" for suffix in ("code", "rin", "rout")]
    if any(name not in tensors for name in names):
        raise ValueError(f"missing source tensor(s): {names}")
    code_t, rin_t, rout_t = (tensors[name] for name in names)
    code = np.ascontiguousarray(code_t.data)
    rin = np.ascontiguousarray(rin_t.data)
    rout = np.ascontiguousarray(rout_t.data)
    if code.shape != (N_STAGE, OC // 16, 32) or code.dtype != np.dtype("int16"):
        raise ValueError(f"unexpected code shape/dtype: {code.shape} {code.dtype}")
    if rin.shape != (IC,) or rin.dtype != np.dtype("float16"):
        raise ValueError(f"unexpected rin shape/dtype: {rin.shape} {rin.dtype}")
    if rout.shape != (OC,) or rout.dtype != np.dtype("float16"):
        raise ValueError(f"unexpected rout shape/dtype: {rout.shape} {rout.dtype}")

    code_bytes = memoryview(code).cast("B")
    rin_bytes = memoryview(rin).cast("B")
    rout_bytes = memoryview(rout).cast("B")
    code_sha, rin_sha, rout_sha = map(sha256_bytes, (code_bytes, rin_bytes, rout_bytes))
    header = make_header(code_sha, rin_sha, rout_sha)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.output.parent, prefix=args.output.name + ".", delete=False
    ) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(header)
        for record in iter_super_records(code):
            tmp.write(record)
        tmp.flush()
        os.fsync(tmp.fileno())
    if temp_path.stat().st_size != TOTAL_BYTES:
        temp_path.unlink(missing_ok=True)
        raise ValueError("overlay size mismatch")
    os.replace(temp_path, args.output)

    overlay = args.output.read_bytes()
    if overlay[:HEADER_BYTES] != header:
        raise ValueError("overlay header reopen mismatch")
    restored = reverse_payload(memoryview(overlay)[PAYLOAD_OFFSET:])
    if not np.array_equal(restored, code):
        raise ValueError("overlay reverse transform is not byte-exact")

    canonical_bytes = code.nbytes + rin.nbytes + rout.nbytes
    effective_bytes = len(overlay) + rin.nbytes + rout.nbytes
    report = {
        "format": "escha-mma-cache-v3-pipe4",
        "scope": TENSOR,
        "source": str(args.source.resolve()),
        "source_size_bytes": args.source.stat().st_size,
        "source_tensor_offsets": {
            names[0]: int(code_t.data_offset),
            names[1]: int(rin_t.data_offset),
            names[2]: int(rout_t.data_offset),
        },
        "shape": {"k": K, "ic": IC, "oc": OC},
        "layout": {
            "order": ["output_cta", "bk64", "band_64", "k16", "tile_16", "word"],
            "records": N_RECORDS,
            "record_bytes": SUPER_BYTES,
            "header_bytes": HEADER_BYTES,
            "payload_offset": PAYLOAD_OFFSET,
            "descriptor_count": 0,
            "stored_word_copies_per_canonical_word": 1,
            "u16_indices_per_weight": 0,
            "fp16_values_per_weight": 0,
        },
        "bytes": {
            "canonical_code": code.nbytes,
            "canonical_rin": rin.nbytes,
            "canonical_rout": rout.nbytes,
            "canonical_code_plus_rotations": canonical_bytes,
            "overlay_file": len(overlay),
            "overlay_payload": PAYLOAD_BYTES,
            "effective_overlay_plus_reused_rotations": effective_bytes,
        },
        "growth_pct": (effective_bytes / canonical_bytes - 1.0) * 100.0,
        "sha256": {
            "source_code": code_sha,
            "source_rin": rin_sha,
            "source_rout": rout_sha,
            "overlay_payload": sha256_bytes(memoryview(overlay)[PAYLOAD_OFFSET:]),
            "overlay_file": sha256_bytes(overlay),
        },
        "validation": {
            "metadata_gate": True,
            "reverse_transform_byte_exact": True,
            "no_descriptor_table": True,
            "no_u16_or_fp16_per_weight_stream": True,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
