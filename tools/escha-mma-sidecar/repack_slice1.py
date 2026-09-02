#!/usr/bin/env python3
"""Repack exactly blk.0.ffn_gate K2 into escha-mma-cache-v1 Slice 1."""
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

MAGIC = b"ESCHA-MMA-V1\0\0\0\0"
VERSION = 1
HEADER_BYTES = 256
K = 2
IC = 5120
OC = 17408
K_STAGE = 16
CTA_N = 128
BANDS = 2
TILES_PER_BAND = 4
CANONICAL_WORDS = 16
RING_WORDS = 17
RECORD_BYTES = TILES_PER_BAND * RING_WORDS * 4
DESC_COUNT = 8 * 2 * 32
DESC_BYTES = 16
DESC_OFFSET = HEADER_BYTES
PAYLOAD_OFFSET = DESC_OFFSET + DESC_COUNT * DESC_BYTES
N_CTA = OC // CTA_N
N_STAGE = IC // K_STAGE
N_RECORDS = N_CTA * N_STAGE * BANDS
PAYLOAD_BYTES = N_RECORDS * RECORD_BYTES
TOTAL_BYTES = PAYLOAD_OFFSET + PAYLOAD_BYTES
TENSOR = "blk.0.ffn_gate"


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def dep_pi(row: int) -> int:
    return (row & 1) | (((row >> 3) & 1) << 1) | (((row >> 1) & 3) << 3)


def chain(row: int, col: int) -> tuple[int, int, int]:
    nb = 32 * 8 * K
    sp = ((32 - K) - K * (dep_pi(row) + 32 * col + 4 * (col >> 3))) % nb
    dw0 = (8 * K - (sp >> 5)) if (sp >> 5) else 0
    dw1 = dw0 - 1 if dw0 else 8 * K - 1
    return dw0, dw1, sp & 31


def ring_position(dw0: int) -> int:
    return 0 if dw0 == 0 else CANONICAL_WORDS - dw0


def build_descriptors() -> bytes:
    out = bytearray()
    for frag_j in range(8):
        for slot in range(2):
            for lane in range(32):
                frag_col = lane // 4
                c_band = 8 * frag_j + frag_col
                tile = c_band // 16
                ccl = c_band & 15
                q = 4 * slot + (lane & 3)
                row0, row1 = 2 * q, 2 * q + 1
                dw00, dw10, sh0 = chain(row0, ccl)
                dw01, dw11, sh1 = chain(row1, ccl)
                off0 = tile * RING_WORDS + ring_position(dw00)
                off1 = tile * RING_WORDS + ring_position(dw01)
                assert 0 <= off0 < 68 and 0 <= off1 < 68
                runtime = off0 | (sh0 << 7) | (off1 << 12) | (sh1 << 19)
                out += struct.pack(
                    "<12BI",
                    row0, row1, frag_col, slot, dep_pi(row0), dep_pi(row1),
                    dw00, dw10, dw01, dw11, sh0, sh1, runtime,
                )
    assert len(out) == DESC_COUNT * DESC_BYTES
    return bytes(out)


def tensor_map(reader: gguf.GGUFReader) -> dict[str, object]:
    return {tensor.name: tensor for tensor in reader.tensors}


def metadata(reader: gguf.GGUFReader, name: str):
    field = reader.get_field(name)
    if field is None:
        raise ValueError(f"missing metadata {name}")
    return field.contents()


def make_header(code_sha: str, rin_sha: str, rout_sha: str, desc_sha: str) -> bytes:
    values = (
        VERSION, HEADER_BYTES, K, IC, OC, K_STAGE, CTA_N, BANDS,
        TILES_PER_BAND, CANONICAL_WORDS, RING_WORDS, RECORD_BYTES,
        DESC_COUNT, DESC_BYTES, N_RECORDS,
    )
    prefix = struct.pack("<16s15I4Q", MAGIC, *values, DESC_OFFSET, PAYLOAD_OFFSET,
                         PAYLOAD_BYTES, TOTAL_BYTES)
    hashes = b"".join(bytes.fromhex(x) for x in (code_sha, rin_sha, rout_sha, desc_sha))
    header = prefix + hashes
    if len(header) > HEADER_BYTES:
        raise AssertionError(len(header))
    return header.ljust(HEADER_BYTES, b"\0")


def iter_records(code: np.ndarray):
    words = np.ascontiguousarray(code).view("<u4").reshape(N_STAGE, OC // 16, 16)
    for cta in range(N_CTA):
        for stage in range(N_STAGE):
            for band in range(BANDS):
                record: list[np.ndarray] = []
                for tile in range(TILES_PER_BAND):
                    canonical = words[stage, cta * 8 + band * 4 + tile]
                    ring = np.concatenate((canonical[:1], canonical[:0:-1], canonical[:1]))
                    assert ring.shape == (RING_WORDS,)
                    record.append(ring)
                yield np.concatenate(record).astype("<u4", copy=False).tobytes()


def reverse_payload(payload: memoryview) -> np.ndarray:
    restored = np.empty((N_STAGE, OC // 16, CANONICAL_WORDS), dtype="<u4")
    pos = 0
    for cta in range(N_CTA):
        for stage in range(N_STAGE):
            for band in range(BANDS):
                rec = np.frombuffer(payload[pos:pos + RECORD_BYTES], dtype="<u4")
                pos += RECORD_BYTES
                for tile in range(TILES_PER_BAND):
                    ring = rec[tile * RING_WORDS:(tile + 1) * RING_WORDS]
                    canonical = np.empty(CANONICAL_WORDS, dtype="<u4")
                    canonical[0] = ring[0]
                    canonical[1:] = ring[CANONICAL_WORDS - 1:0:-1]
                    assert ring[-1] == canonical[0]
                    restored[stage, cta * 8 + band * 4 + tile] = canonical
    assert pos == len(payload)
    return restored.view("<i2").reshape(N_STAGE, OC // 16, 32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    reader = gguf.GGUFReader(str(args.source), "r")
    if metadata(reader, "general.architecture") != "qwen35" or int(metadata(reader, "qwen35.escha.version")) != 1:
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
    descriptors = build_descriptors()
    desc_sha = sha256_bytes(descriptors)
    header = make_header(code_sha, rin_sha, rout_sha, desc_sha)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=args.output.name + ".", delete=False) as tmp:
        temp_path = Path(tmp.name)
        tmp.write(header)
        tmp.write(descriptors)
        for record in iter_records(code):
            tmp.write(record)
        tmp.flush()
        os.fsync(tmp.fileno())
    if temp_path.stat().st_size != TOTAL_BYTES:
        temp_path.unlink(missing_ok=True)
        raise ValueError("overlay size mismatch")
    os.replace(temp_path, args.output)

    overlay = args.output.read_bytes()
    if overlay[:HEADER_BYTES] != header or overlay[DESC_OFFSET:PAYLOAD_OFFSET] != descriptors:
        raise ValueError("overlay header/descriptor reopen mismatch")
    restored = reverse_payload(memoryview(overlay)[PAYLOAD_OFFSET:])
    if not np.array_equal(restored, code):
        raise ValueError("overlay reverse transform is not byte-exact")

    canonical_bytes = code.nbytes + rin.nbytes + rout.nbytes
    effective_bytes = len(overlay) + rin.nbytes + rout.nbytes
    report = {
        "format": "escha-mma-cache-v1",
        "scope": TENSOR,
        "source": str(args.source.resolve()),
        "source_size_bytes": args.source.stat().st_size,
        "source_tensor_offsets": {
            names[0]: int(code_t.data_offset), names[1]: int(rin_t.data_offset), names[2]: int(rout_t.data_offset),
        },
        "shape": {"k": K, "ic": IC, "oc": OC},
        "layout": {
            "order": ["output_cta", "k_stage", "band_64", "warp_publication_record"],
            "records": N_RECORDS, "record_bytes": RECORD_BYTES,
            "descriptor_count": DESC_COUNT, "descriptor_bytes": DESC_BYTES,
            "descriptor_offset": DESC_OFFSET, "payload_offset": PAYLOAD_OFFSET,
        },
        "bytes": {
            "canonical_code": code.nbytes, "canonical_rin": rin.nbytes,
            "canonical_rout": rout.nbytes, "canonical_code_plus_rotations": canonical_bytes,
            "overlay_file": len(overlay), "overlay_payload": PAYLOAD_BYTES,
            "effective_overlay_plus_reused_rotations": effective_bytes,
        },
        "growth_pct": (effective_bytes / canonical_bytes - 1.0) * 100.0,
        "sha256": {
            "source_code": code_sha, "source_rin": rin_sha, "source_rout": rout_sha,
            "descriptors": desc_sha, "overlay_payload": sha256_bytes(memoryview(overlay)[PAYLOAD_OFFSET:]),
            "overlay_file": sha256_bytes(overlay),
        },
        "validation": {"metadata_gate": True, "reverse_transform_byte_exact": True},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
