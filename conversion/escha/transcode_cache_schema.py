"""Frozen EXP-11 v1 transcode-cache schema and naming contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_FORMAT_VERSION = 1
MANIFEST_SCHEMA = "beellama.escha-transcode.manifest.v1"
TENSOR_LAYOUT_VERSION = "gguf-standard-weight-oc-ic-v1"
DEFAULT_ROOT_SUFFIX = Path("escha-transcode") / "v1"
OVERLAY_FILENAME = "overlay.gguf"
MANIFEST_FILENAME = "manifest.json"
OVERLAY_SHA_FILENAME = "overlay.sha256"
COMPLETE_FILENAME = "complete"
LOCK_FILENAME = ".lock"
TEMP_PATTERN = "{filename}.tmp.{pid}"

FFN_ROLES = ("ffn_gate", "ffn_up", "ffn_down")
FFN_DIMENSIONS = {
    "ffn_gate": (5120, 17408),
    "ffn_up": (5120, 17408),
    "ffn_down": (17408, 5120),
}
FFN_MASK_SUFFIXES = ("escha_code", "escha_rin", "escha_rout", "bias")


def standard_tensor_name(layer: int, role: str) -> str:
    """Version-1 overlay name, matching convert_escha_to_gguf.py."""
    if role not in FFN_ROLES:
        raise ValueError(f"unsupported FFN role: {role}")
    return f"blk.{layer}.{role}.weight"


def sidecar_tensor_name(layer: int, role: str, suffix: str) -> str:
    if role not in FFN_ROLES or suffix not in FFN_MASK_SUFFIXES:
        raise ValueError((layer, role, suffix))
    return f"blk.{layer}.{role}.{suffix}"


@dataclass(frozen=True)
class RecipeSpec:
    format_version: int
    source_sha256: str
    architecture: str
    escha_version: int
    scope: str
    ordered_roles: tuple[str, ...]
    target_quant: tuple[tuple[str, str], ...]
    oracle_module_sha256: str
    numpy_version: str
    quantizer_abi: str
    endianness: str
    tensor_layout_version: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["ordered_roles"] = list(self.ordered_roles)
        value["target_quant"] = {key: quant for key, quant in self.target_quant}
        return value

    def canonical_json(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def recipe_id(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


@dataclass(frozen=True)
class EntryKey:
    source_gguf_sha256: str
    layer: int
    tensor_role: str
    source_code_sha256: str
    source_rin_sha256: str
    source_rout_sha256: str
    k: int
    ic: int
    oc: int
    target_quant: str
    oracle_abi: str
    layout_version: str


@dataclass(frozen=True)
class ManifestEntry:
    key: EntryKey
    output_tensor_name: str
    ggml_type: str
    shape: tuple[int, ...]
    byte_count: int
    data_offset: int
    payload_sha256: str
    source_fp32_sha256: str
    source_fp32_mae: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        return value


@dataclass(frozen=True)
class Manifest:
    manifest_schema: str
    cache_format_version: int
    source_path: str
    source_sha256: str
    source_size: int
    recipe_id: str
    recipe: dict[str, Any]
    scope: str
    layers: tuple[int, ...]
    masking: dict[str, Any]
    overlay_filename: str
    overlay_sha256: str
    overlay_size: int
    entries: tuple[ManifestEntry, ...]
    timing: dict[str, Any]
    resources: dict[str, Any]
    generator: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["layers"] = list(self.layers)
        value["entries"] = [entry.to_dict() for entry in self.entries]
        return value


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical recipe/manifest JSON: UTF-8, sorted keys, no whitespace."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def entry_directory(root: Path, source_sha256: str, recipe_id: str) -> Path:
    """Return <root>/<source-sha256>/<recipe-id>/ exactly."""
    return root / source_sha256 / recipe_id


CACHE_CONTRACT = {
    "root_layout": "<root>/<source-sha256>/<recipe-id>/",
    "files": {
        "overlay": OVERLAY_FILENAME,
        "manifest": MANIFEST_FILENAME,
        "overlay_sha256": OVERLAY_SHA_FILENAME,
        "complete_marker": COMPLETE_FILENAME,
        "lock": LOCK_FILENAME,
        "temporary": TEMP_PATTERN,
    },
    "publication": (
        "exclusive lock; write PID-scoped temporary overlay and manifest; close+fsync; "
        "validate as a new reader; atomic rename; write zero-length complete marker last"
    ),
    "selection": "read complete entries only; never read .tmp files",
    "replacement": "retain the previous complete entry until replacement validates",
    "tensor_layout_version": TENSOR_LAYOUT_VERSION,
    "overlay_names": "blk.{layer}.{ffn_gate|ffn_up|ffn_down}.weight",
    "mask_sidecars": list(FFN_MASK_SUFFIXES),
}
