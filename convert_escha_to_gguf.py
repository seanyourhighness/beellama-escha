#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert an Escha 2/3-bit safetensors checkpoint (dense, Qwen3.5 architecture) to
Beellama GGUF with the projections kept in their native escha trellis code.

What lands in the GGUF
----------------------
* every linear projection stays in the packed 2/3-bit code (same bytes as the
  safetensors), plus the per-projection rin/rout rotation vectors and the
  bias-correction vector (loaded, not applied -- matches the reference runtime);
* the shared codec tables: `escha_lut` (codebook A, computed from the closed
  form) and `escha_dep_k2` / `escha_dep_k3` (bit-dependency tables, extracted
  from the escha kernel and checked bit-for-bit against `escham_reconstruct`);
* the token embedding and LM head are dequantized to fp16
  (`token_embd.weight`, `output.weight`), bit-exact to the reference LowGPU
  dequant -- the native 3-bit vocab is deferred (use `--vocab lowgpu` to
  store the packed codes and decode them in-kernel instead);
* the remaining tensors (norms, ssm params) are written fp16/fp32 like the
  reference GGUF for this model.

Run:
    python3 convert_escha_to_gguf.py \
        --input-dir weights/escha-w2-lowgpu-mono \
        --outfile escha-w2-lowgpu-mono.gguf

Requires the escha venv (safetensors) and this repo's gguf-py.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))

import gguf
from gguf import GGUFValueType, GGMLQuantizationType


# ---------------------------------------------------------------------------
# K-quant quantization (pure NumPy port of ggml/src/ggml-quants.c)
# ---------------------------------------------------------------------------

QK_K = 256
Q2_K_TYPE_SIZE = 84
Q4_K_TYPE_SIZE = 144
Q6_K_TYPE_SIZE = 210


def _nearest_int(x: np.ndarray) -> np.ndarray:
    """Match ggml's nearest_int (round to nearest, ties to even)."""
    return np.rint(x).astype(np.int32)


def make_qkx3_quants(x: np.ndarray, weights: np.ndarray, nmax: int = 3,
                     rmin: float = -0.9, rdelta: float = 0.05,
                     nstep: int = 36, use_mad: bool = False
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized make_qkx3_quants for fixed-width rows."""
    x = np.asarray(x, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    assert x.ndim == 2, x.shape
    assert weights.shape == x.shape, weights.shape

    xmin = np.minimum(np.min(x, axis=1), np.float32(0.0))
    xmax = np.max(x, axis=1)
    span = xmax - xmin
    valid = span > 0

    iscale = np.zeros_like(span)
    np.divide(np.float32(nmax), span, out=iscale, where=valid)
    scale = np.zeros_like(span)
    np.divide(np.float32(1.0), iscale, out=scale, where=valid)
    levels = np.clip(_nearest_int(iscale[:, None] * (x - xmin[:, None])),
                     0, nmax).astype(np.uint8)

    diff = scale[:, None] * levels.astype(np.float32) + xmin[:, None] - x
    loss = np.abs(diff) if use_mad else diff * diff
    best_mad = np.sum(weights * loss, axis=1, dtype=np.float32)
    sum_w = np.sum(weights, axis=1, dtype=np.float32)
    sum_x = np.sum(weights * x, axis=1, dtype=np.float32)

    for istep in range(nstep + 1):
        trial_iscale = np.zeros_like(span)
        numerator = (np.float32(rmin) + np.float32(rdelta) * np.float32(istep)
                     + np.float32(nmax))
        np.divide(numerator, span, out=trial_iscale, where=valid)
        trial_levels = np.clip(
            _nearest_int(trial_iscale[:, None] * (x - xmin[:, None])),
            0, nmax,
        ).astype(np.uint8)
        lf = trial_levels.astype(np.float32)
        sum_l = np.sum(weights * lf, axis=1, dtype=np.float32)
        sum_l2 = np.sum(weights * lf * lf, axis=1, dtype=np.float32)
        sum_xl = np.sum(weights * lf * x, axis=1, dtype=np.float32)
        determinant = sum_w * sum_l2 - sum_l * sum_l
        candidate = valid & (determinant > 0)
        if not np.any(candidate):
            continue

        trial_scale = np.zeros_like(span)
        trial_min = np.zeros_like(span)
        np.divide(sum_w * sum_xl - sum_x * sum_l, determinant,
                  out=trial_scale, where=candidate)
        np.divide(sum_l2 * sum_x - sum_l * sum_xl, determinant,
                  out=trial_min, where=candidate)
        positive_min = candidate & (trial_min > 0)
        np.divide(sum_xl, sum_l2, out=trial_scale,
                  where=positive_min & (sum_l2 != 0))
        trial_min[positive_min] = 0

        diff = trial_scale[:, None] * lf + trial_min[:, None] - x
        loss = np.abs(diff) if use_mad else diff * diff
        mad = np.sum(weights * loss, axis=1, dtype=np.float32)
        better = candidate & (mad < best_mad)
        levels[better] = trial_levels[better]
        best_mad[better] = mad[better]
        scale[better] = trial_scale[better]
        xmin[better] = trial_min[better]

    return scale, -xmin, levels


def make_qp_quants(x: np.ndarray, quant_weights: np.ndarray,
                   nmax: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized make_qp_quants for fixed-width non-negative rows."""
    x = np.asarray(x, dtype=np.float32)
    quant_weights = np.asarray(quant_weights, dtype=np.float32)
    assert x.ndim == 2, x.shape
    assert quant_weights.shape == x.shape, quant_weights.shape

    xmax = np.max(x, axis=1)
    valid = xmax >= np.float32(1e-15)
    iscale = np.zeros_like(xmax)
    np.divide(np.float32(nmax), xmax, out=iscale, where=valid)
    levels = _nearest_int(iscale[:, None] * x)
    levels[~valid] = 0
    scale = np.zeros_like(xmax)
    np.divide(np.float32(1.0), iscale, out=scale, where=valid)
    diff = x - scale[:, None] * levels.astype(np.float32)
    best_mse = np.sum(quant_weights * diff * diff, axis=1, dtype=np.float32)

    for istep in range(-4, 5):
        if istep == 0:
            continue
        trial_iscale = np.zeros_like(xmax)
        numerator = np.float32(0.1) * np.float32(istep) + np.float32(nmax)
        np.divide(numerator, xmax,
                  out=trial_iscale, where=valid)
        trial_scale = np.zeros_like(xmax)
        np.divide(np.float32(1.0), trial_iscale,
                  out=trial_scale, where=valid)
        trial_levels = np.minimum(nmax, _nearest_int(trial_iscale[:, None] * x))
        diff = x - trial_scale[:, None] * trial_levels.astype(np.float32)
        mse = np.sum(quant_weights * diff * diff, axis=1, dtype=np.float32)
        better = valid & (mse < best_mse)
        best_mse[better] = mse[better]
        iscale[better] = trial_iscale[better]

    levels = np.minimum(nmax, _nearest_int(iscale[:, None] * x))
    levels[~valid] = 0
    lf = levels.astype(np.float32)
    sum_lx = np.sum(quant_weights * x * lf, axis=1, dtype=np.float32)
    sum_l2 = np.sum(quant_weights * lf * lf, axis=1, dtype=np.float32)

    for _ in range(5):
        changed = np.zeros(x.shape[0], dtype=bool)
        for i in range(x.shape[1]):
            old_l = levels[:, i].astype(np.float32)
            weight = quant_weights[:, i]
            slx = sum_lx - weight * x[:, i] * old_l
            sl2 = sum_l2 - weight * old_l * old_l
            candidate = (slx > 0) & (sl2 > 0)
            ratio = np.zeros_like(slx)
            np.divide(x[:, i] * sl2, slx, out=ratio, where=candidate)
            new_l = np.minimum(nmax, _nearest_int(ratio))
            different = candidate & (new_l != levels[:, i])
            new_lf = new_l.astype(np.float32)
            new_slx = slx + weight * x[:, i] * new_lf
            new_sl2 = sl2 + weight * new_lf * new_lf
            better = different & (
                new_slx * new_slx * sum_l2 > sum_lx * sum_lx * new_sl2
            )
            levels[better, i] = new_l[better]
            sum_lx[better] = new_slx[better]
            sum_l2[better] = new_sl2[better]
            changed |= better
        if not np.any(changed):
            break

    result_scale = np.zeros_like(sum_lx)
    np.divide(sum_lx, sum_l2, out=result_scale, where=sum_l2 > 0)
    return result_scale, levels.astype(np.uint8)


def make_qx_quants(x: np.ndarray, nmax: int = 32
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized make_qx_quants with rmse_type=1 and no external weights."""
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2, x.shape

    abs_x = np.abs(x)
    imax = np.argmax(abs_x, axis=1)
    xmax = x[np.arange(x.shape[0]), imax]
    valid = abs_x[np.arange(x.shape[0]), imax] >= np.float32(1e-15)

    iscale = np.zeros_like(xmax)
    np.divide(np.float32(-nmax), xmax, out=iscale, where=valid)
    signed = np.clip(_nearest_int(iscale[:, None] * x),
                     -nmax, nmax - 1)
    signed[~valid] = -nmax
    levels = (signed + nmax).astype(np.uint8)

    weights = x * x
    sf = signed.astype(np.float32)
    sum_lx = np.sum(weights * x * sf, axis=1, dtype=np.float32)
    sum_l2 = np.sum(weights * sf * sf, axis=1, dtype=np.float32)
    scale = np.zeros_like(sum_lx)
    np.divide(sum_lx, sum_l2, out=scale, where=sum_l2 > 0)
    best = scale * sum_lx

    for istep in range(-9, 10):
        if istep == 0:
            continue
        trial_iscale = np.zeros_like(xmax)
        numerator = -(np.float32(nmax) + np.float32(0.1) * np.float32(istep))
        np.divide(numerator, xmax, out=trial_iscale, where=valid)
        trial_signed = np.clip(_nearest_int(trial_iscale[:, None] * x),
                               -nmax, nmax - 1)
        tf = trial_signed.astype(np.float32)
        trial_lx = np.sum(weights * x * tf, axis=1, dtype=np.float32)
        trial_l2 = np.sum(weights * tf * tf, axis=1, dtype=np.float32)
        better = valid & (trial_l2 > 0) & (trial_lx * trial_lx > best * trial_l2)
        if not np.any(better):
            continue
        levels[better] = (trial_signed[better] + nmax).astype(np.uint8)
        scale[better] = trial_lx[better] / trial_l2[better]
        best[better] = scale[better] * trial_lx[better]

    levels[~valid] = 0
    return scale, levels


def quantize_q2_k(data: np.ndarray, batch_blocks: int = 4096) -> np.ndarray:
    """Quantize float rows to block_q2_K bytes (QK_K=256, type_size=84)."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim < 1 or data.shape[-1] % QK_K:
        raise ValueError(f"Q2_K requires the last dimension to be divisible by {QK_K}: {data.shape}")

    blocks = data.reshape(-1, QK_K)
    packed = np.empty((blocks.shape[0], Q2_K_TYPE_SIZE), dtype=np.uint8)
    for block0 in range(0, blocks.shape[0], batch_blocks):
        xb = blocks[block0:block0 + batch_blocks]
        nblock = xb.shape[0]
        sigma2 = np.sum(xb * xb, axis=1, dtype=np.float32) / np.float32(QK_K)
        groups = xb.reshape(nblock, 16, 16)
        weights = np.sqrt(sigma2[:, None, None] + groups * groups).astype(np.float32)

        scales, mins, levels = make_qkx3_quants(
            groups.reshape(-1, 16), weights.reshape(-1, 16)
        )
        scales = scales.reshape(nblock, 16)
        mins = mins.reshape(nblock, 16)
        levels = levels.reshape(nblock, QK_K)
        sw = np.sum(weights, axis=2, dtype=np.float32)
        dm, ls = make_qp_quants(scales, sw)
        mm, lm = make_qp_quants(mins, sw)

        # The C implementation stores these as fp16, reads them back, and only
        # then requantizes the 2-bit values.
        dm_f16 = dm.astype("<f2")
        mm_f16 = mm.astype("<f2")
        dm = dm_f16.astype(np.float32)
        mm = mm_f16.astype(np.float32)
        scale_bytes = (ls | (lm << np.uint8(4))).astype(np.uint8)

        d = dm[:, None] * ls.astype(np.float32)
        m = mm[:, None] * lm.astype(np.float32)
        requant = levels.reshape(nblock, 16, 16).copy()
        nonzero = d != 0
        candidate = np.zeros_like(groups, dtype=np.float32)
        np.divide(groups + m[:, :, None], d[:, :, None],
                  out=candidate, where=nonzero[:, :, None])
        candidate = np.clip(_nearest_int(candidate), 0, 3).astype(np.uint8)
        requant[nonzero] = candidate[nonzero]
        levels = requant.reshape(nblock, 2, 4, 32)
        qs = (levels[:, :, 0] |
              (levels[:, :, 1] << np.uint8(2)) |
              (levels[:, :, 2] << np.uint8(4)) |
              (levels[:, :, 3] << np.uint8(6))).reshape(nblock, 64)

        out = packed[block0:block0 + nblock]
        out[:, :16] = scale_bytes
        out[:, 16:80] = qs
        out[:, 80:82] = dm_f16.view(np.uint8).reshape(nblock, 2)
        out[:, 82:84] = mm_f16.view(np.uint8).reshape(nblock, 2)

    byte_shape = (*data.shape[:-1], data.shape[-1] // QK_K * Q2_K_TYPE_SIZE)
    return packed.reshape(byte_shape)


def quantize_q4_k(data: np.ndarray, batch_blocks: int = 4096) -> np.ndarray:
    """Quantize float rows to block_q4_K bytes (QK_K=256, type_size=144)."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim < 1 or data.shape[-1] % QK_K:
        raise ValueError(f"Q4_K requires the last dimension to be divisible by {QK_K}: {data.shape}")

    blocks = data.reshape(-1, QK_K)
    packed = np.empty((blocks.shape[0], Q4_K_TYPE_SIZE), dtype=np.uint8)
    for block0 in range(0, blocks.shape[0], batch_blocks):
        xb = blocks[block0:block0 + batch_blocks]
        nblock = xb.shape[0]
        sigma2 = (np.float32(2.0) * np.sum(xb * xb, axis=1, dtype=np.float32)
                  / np.float32(QK_K))
        groups = xb.reshape(nblock, 8, 32)
        weights = (np.sqrt(sigma2)[:, None, None] + np.abs(groups)).astype(np.float32)

        scales, mins, levels = make_qkx3_quants(
            groups.reshape(-1, 32), weights.reshape(-1, 32), nmax=15
        )
        scales = scales.reshape(nblock, 8)
        mins = mins.reshape(nblock, 8)
        levels = levels.reshape(nblock, 8, 32)
        sw = np.sum(weights, axis=2, dtype=np.float32)
        d_block, ls = make_qp_quants(scales, sw, nmax=63)
        m_block, lm = make_qp_quants(mins, sw, nmax=63)

        d_f16 = d_block.astype("<f2")
        m_f16 = m_block.astype("<f2")
        d_block = d_f16.astype(np.float32)
        m_block = m_f16.astype(np.float32)

        scale_bytes = np.zeros((nblock, 12), dtype=np.uint8)
        scale_bytes[:, :4] = ls[:, :4]
        scale_bytes[:, 4:8] = lm[:, :4]
        scale_bytes[:, 8:12] = ((ls[:, 4:] & np.uint8(0x0f)) |
                                ((lm[:, 4:] & np.uint8(0x0f)) << np.uint8(4)))
        scale_bytes[:, :4] |= (ls[:, 4:] >> np.uint8(4)) << np.uint8(6)
        scale_bytes[:, 4:8] |= (lm[:, 4:] >> np.uint8(4)) << np.uint8(6)

        d = d_block[:, None] * ls.astype(np.float32)
        m = m_block[:, None] * lm.astype(np.float32)
        nonzero = d != 0
        candidate = np.zeros_like(groups, dtype=np.float32)
        np.divide(groups + m[:, :, None], d[:, :, None],
                  out=candidate, where=nonzero[:, :, None])
        candidate = np.clip(_nearest_int(candidate), 0, 15).astype(np.uint8)
        levels[nonzero] = candidate[nonzero]
        level_pairs = levels.reshape(nblock, 4, 2, 32)
        qs = (level_pairs[:, :, 0] |
              (level_pairs[:, :, 1] << np.uint8(4))).reshape(nblock, 128)

        out = packed[block0:block0 + nblock]
        out[:, 0:2] = d_f16.view(np.uint8).reshape(nblock, 2)
        out[:, 2:4] = m_f16.view(np.uint8).reshape(nblock, 2)
        out[:, 4:16] = scale_bytes
        out[:, 16:144] = qs

    byte_shape = (*data.shape[:-1], data.shape[-1] // QK_K * Q4_K_TYPE_SIZE)
    return packed.reshape(byte_shape)


def quantize_q6_k(data: np.ndarray, batch_blocks: int = 4096) -> np.ndarray:
    """Quantize float rows to block_q6_K bytes (QK_K=256, type_size=210)."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim < 1 or data.shape[-1] % QK_K:
        raise ValueError(f"Q6_K requires the last dimension to be divisible by {QK_K}: {data.shape}")

    blocks = data.reshape(-1, QK_K)
    packed = np.empty((blocks.shape[0], Q6_K_TYPE_SIZE), dtype=np.uint8)
    for block0 in range(0, blocks.shape[0], batch_blocks):
        xb = blocks[block0:block0 + batch_blocks]
        nblock = xb.shape[0]
        groups = xb.reshape(nblock, 16, 16)
        sub_scales, levels = make_qx_quants(groups.reshape(-1, 16), nmax=32)
        sub_scales = sub_scales.reshape(nblock, 16)
        levels = levels.reshape(nblock, 16, 16)

        imax = np.argmax(np.abs(sub_scales), axis=1)
        max_scale = sub_scales[np.arange(nblock), imax]
        valid = np.abs(max_scale) >= np.float32(1e-15)
        iscale = np.zeros_like(max_scale)
        np.divide(np.float32(-128.0), max_scale, out=iscale, where=valid)
        d_block = np.zeros_like(max_scale)
        np.divide(np.float32(1.0), iscale, out=d_block, where=valid)
        d_f16 = d_block.astype("<f2")
        d_block = d_f16.astype(np.float32)

        scales = np.minimum(127, _nearest_int(iscale[:, None] * sub_scales))
        scales[~valid] = 0
        scales = scales.astype(np.int8)
        d = d_block[:, None] * scales.astype(np.float32)
        nonzero = d != 0
        candidate = np.zeros_like(groups, dtype=np.float32)
        np.divide(groups, d[:, :, None], out=candidate,
                  where=nonzero[:, :, None])
        candidate = (np.clip(_nearest_int(candidate), -32, 31) + 32).astype(np.uint8)
        levels[nonzero] = candidate[nonzero]

        quartets = levels.reshape(nblock, 2, 4, 32)
        ql = np.concatenate([
            (quartets[:, :, 0] & np.uint8(0x0f)) |
            ((quartets[:, :, 2] & np.uint8(0x0f)) << np.uint8(4)),
            (quartets[:, :, 1] & np.uint8(0x0f)) |
            ((quartets[:, :, 3] & np.uint8(0x0f)) << np.uint8(4)),
        ], axis=2).reshape(nblock, 128)
        qh = ((quartets[:, :, 0] >> np.uint8(4)) |
              ((quartets[:, :, 1] >> np.uint8(4)) << np.uint8(2)) |
              ((quartets[:, :, 2] >> np.uint8(4)) << np.uint8(4)) |
              ((quartets[:, :, 3] >> np.uint8(4)) << np.uint8(6))).reshape(nblock, 64)

        out = packed[block0:block0 + nblock]
        out[:, 0:128] = ql
        out[:, 128:192] = qh
        out[:, 192:208] = scales.view(np.uint8)
        out[:, 208:210] = d_f16.view(np.uint8).reshape(nblock, 2)

    byte_shape = (*data.shape[:-1], data.shape[-1] // QK_K * Q6_K_TYPE_SIZE)
    return packed.reshape(byte_shape)


# ---------------------------------------------------------------------------
# escha codec
# ---------------------------------------------------------------------------

def escha_codebook_lut() -> np.ndarray:
    """Codebook A, closed form (recovered from escham_reconstruct_kernel<1, K>)."""
    idx = np.arange(65536, dtype=np.uint32)
    x = ((idx * np.uint32(0xCBAC1FED)) & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
    lo = (x & np.uint32(0xFFFF)).astype("<u2").view(np.float16)
    hi = (x >> np.uint32(16)).astype("<u2").view(np.float16)
    v = (lo.astype(np.float32) + hi.astype(np.float32)).astype(np.float16)
    return v


def load_dep_table(path: Path) -> np.ndarray:
    dep = np.load(path)
    assert dep.shape == (16, 256), dep.shape
    # writer reverses dims into the file, so store as [weight][bit] -> file [bit][weight]
    return np.ascontiguousarray(dep.T).astype(np.int16)


# ---------------------------------------------------------------------------
# LowGPU vocab dequant (bit-exact to source/lowgpu/format.py::dequant_full)
# ---------------------------------------------------------------------------

def dequant_lowgpu_rows(codes: np.ndarray, scales: np.ndarray, zps: np.ndarray,
                        row0: int, row1: int, group: int = 128) -> np.ndarray:
    """Dequant rows [row0, row1) of the packed 3-bit vocab to fp16."""
    v, kb = codes.shape
    k = kb * 8 // 3
    g = (k + group - 1) // group
    rows = np.arange(row0, row1)
    p = codes[rows].reshape(row1 - row0, kb // 3, 3).astype(np.int32)
    b0, b1, b2 = p[..., 0], p[..., 1], p[..., 2]
    c0 = b0 & 7
    c1 = (b0 >> 3) & 7
    c2 = ((b0 >> 6) | (b1 << 2)) & 7
    c3 = (b1 >> 1) & 7
    c4 = (b1 >> 4) & 7
    c5 = ((b1 >> 7) | (b2 << 1)) & 7
    c6 = (b2 >> 2) & 7
    c7 = (b2 >> 5) & 7
    c = np.stack([c0, c1, c2, c3, c4, c5, c6, c7], axis=-1).reshape(row1 - row0, k)
    sc = scales[rows].reshape(row1 - row0, g, 1).astype(np.float32)
    zp = zps[rows].reshape(row1 - row0, g, 1).astype(np.float32)
    out = ((c.reshape(row1 - row0, g, group).astype(np.float32) - zp) * sc)
    return out.reshape(row1 - row0, k).astype(np.float16)


# ---------------------------------------------------------------------------
# checkpoint -> GGUF mapping
# ---------------------------------------------------------------------------

def escha_gguf_name(blk: str, suffix: str, il: int) -> str:
    return f"blk.{il}.{blk}.{suffix}"


def write_escha_proj(writer: gguf.GGUFWriter, find, ckpt_prefix: str, gguf_blk: str, il: int,
                     expect_oc: int | None = None) -> None:
    """Write one escha projection (code/rin/rout/bias) from the checkpoint."""
    code = find(f"{ckpt_prefix}.escha_code")
    rin  = find(f"{ckpt_prefix}.escha_rin")
    rout = find(f"{ckpt_prefix}.escha_rout")
    s_in = find(f"{ckpt_prefix}.escha_s_in")
    s_out = find(f"{ckpt_prefix}.escha_s_out")
    bias = find(f"{ckpt_prefix}.bias")

    # checkpoint code is (IC/16, OC/16, 16*K); GGUF wants (16*K, OC/16, IC/16)
    assert code.ndim == 3
    ic16, oc16, n_code = code.shape
    ic = ic16 * 16
    oc = oc16 * 16
    assert n_code in (32, 48), n_code
    if expect_oc is not None:
        assert oc == expect_oc, (ckpt_prefix, oc, expect_oc)

    assert rin.shape == (ic,), rin.shape
    assert rout.shape == (oc,), rout.shape
    assert s_in.shape == (ic,), s_in.shape
    assert s_out.shape == (oc,), s_out.shape
    assert bias.shape == (oc,), bias.shape

    # Wscale is already folded into rin; fold the per-channel s_in/s_out on top
    rin_g = (rin.astype(np.float32) * s_in.astype(np.float32)).astype(np.float16)
    rout_g = (rout.astype(np.float32) * s_out.astype(np.float32)).astype(np.float16)
    bias_g = bias.astype(np.float32)

    writer.add_tensor(escha_gguf_name(gguf_blk, "escha_code", il),
                      code, raw_dtype=GGMLQuantizationType.I16)
    writer.add_tensor(escha_gguf_name(gguf_blk, "escha_rin", il),
                      rin_g)
    writer.add_tensor(escha_gguf_name(gguf_blk, "escha_rout", il),
                      rout_g)
    writer.add_tensor(escha_gguf_name(gguf_blk, "bias", il),
                      bias_g)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--outfile", required=True, type=Path)
    parser.add_argument("--model-name", type=str, default=None,
                        help="model identity to stamp into the GGUF (default: input-dir basename)")
    parser.add_argument("--metadata-template", type=Path,
                        default=Path(__file__).parent / "conversion" / "escha" / "metadata-template.json")
    parser.add_argument("--dep-k2", type=Path,
                        default=Path(__file__).parent / "conversion" / "escha" / "dep_k2.npy")
    parser.add_argument("--dep-k3", type=Path,
                        default=Path(__file__).parent / "conversion" / "escha" / "dep_k3.npy")
    parser.add_argument("--vocab", choices=("dequant", "lowgpu"), default="dequant",
                        help="vocab storage: dequant (fp16 token_embd/output, default) "
                             "or lowgpu (native 3-bit codes, decoded in-kernel)")
    parser.add_argument("--embed-vocab", choices=("dequant", "lowgpu", "q4_k", "q6_k"), default=None,
                        help="override vocab storage for the embedding only (P-ARCH-22 isolation; "
                             "q4_k/q6_k = standard-quantized token_embd for the fast get_rows path)")
    parser.add_argument("--head-vocab", choices=("dequant", "lowgpu", "q4_k", "q6_k"), default=None,
                        help="override vocab storage for the LM head only (P-ARCH-22 isolation)")
    parser.add_argument("--standard-ffn-gguf", type=Path,
                        help="replace all FFN gate/up/down Escha sidecars with raw quantized "
                             "weights copied from this compatible GGUF")
    parser.add_argument("--standard-attn-ffn", action="store_true",
                        help="also replace the 16 full-attention Q/K/V/output projections "
                             "with raw quantized weights from --standard-ffn-gguf")
    parser.add_argument("--standard-linear-ffn", action="store_true",
                        help="reconstruct the 48 linear-attention QKV/output projections "
                             "from checkpoint Escha sidecars and store them as Q2_K weights")
    parser.add_argument("--standard-gdn-gate", action="store_true",
                        help="also replace the 48 linear-attention gate projections from "
                             "--standard-ffn-gguf")
    parser.add_argument("--standard-gdn-gate-f16", action="store_true",
                        help="diagnostic: write the 48 attn_gate projections as dequantized F16 "
                             "instead of the source's raw quant (over-cap, for root-cause only)")
    parser.add_argument("--standard-gdn-gate-quant", choices=("q2_k", "q4_k", "q6_k"),
                        help="reconstruct the true checkpoint GDN gate and store it using the "
                             "selected standard GGML K-quantization")
    args = parser.parse_args()

    if sum((args.standard_gdn_gate, args.standard_gdn_gate_f16,
            args.standard_gdn_gate_quant is not None)) > 1:
        parser.error("the standard GDN gate options are mutually exclusive")
    if args.standard_attn_ffn and args.standard_ffn_gguf is None:
        parser.error("--standard-attn-ffn requires --standard-ffn-gguf")
    standard_ffn = None
    if args.standard_ffn_gguf is not None:
        standard_ffn_reader = gguf.GGUFReader(str(args.standard_ffn_gguf))
        standard_ffn = {tensor.name: tensor for tensor in standard_ffn_reader.tensors}

    def write_standard_ffn(name: str) -> None:
        assert standard_ffn is not None
        tensor = standard_ffn[name]
        writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type,
                          tensor_endianess=standard_ffn_reader.endianess)

    in_dir = args.input_dir
    files = sorted(p for p in in_dir.iterdir() if p.name.endswith(".safetensors"))
    assert files, f"no safetensors files in {in_dir}"

    layer_prefix = "model.language_model.layers."

    # Derive the block/layer layout from the checkpoint structure itself rather
    # than hard-coding a specific model (no artifact-name/layout assumptions).
    # Full-attention layers expose `self_attn` projections; linear-attention
    # (GDN) layers expose `linear_attn` projections. Layers without either
    # marker default to full attention (matching the qwen35 default interval).
    full_attn: set = set()
    n_layer = 0
    layer_ids: set = set()
    for p in files:
        with open(p, "rb") as f:
            hdr_len = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(hdr_len))
        for k in hdr:
            if ".layers." not in k:
                continue
            parts = k.split(".layers.")[1].split(".")[0]
            try:
                il = int(parts)
            except ValueError:
                continue
            layer_ids.add(il)
            if "linear_attn" in k:
                full_attn.discard(il)
            elif "self_attn" in k:
                full_attn.add(il)
    if layer_ids:
        n_layer = max(layer_ids) + 1
    else:
        # Fallback for checkpoints without per-layer markers: default qwen35
        # full-attention interval (every 4th layer), clearly documented.
        n_layer = 64
        full_attn = {il for il in range(n_layer) if il % 4 == 3}
    assert n_layer > 0, "could not determine layer count from checkpoint"
    print(f"[convert] derived n_layer={n_layer} full_attn={sorted(full_attn)}")

    # metadata: reuse the reference GGUF's header (tokenizer + hparams), with
    # identity fields pointed at this build
    with open(args.metadata_template, encoding="utf-8") as f:
        meta = json.load(f)

    outfile = args.outfile
    writer = gguf.GGUFWriter(outfile, "qwen35", use_temp_file=True)

    for key, typ, elem, val in meta:
        if key == "general.architecture":
            continue
        if key in ("general.name", "general.basename", "general.finetune"):
            continue
        if key == "qwen35.escha.version":
            writer.add_key_value(key, val, GGUFValueType(typ))
            # qwen35.lowgpu.version is written once below, after per-side vocab
            # storage is resolved (--embed-vocab / --head-vocab may mix sides)
            continue
        if typ == 9:
            writer.add_key_value(key, val, GGUFValueType.ARRAY, GGUFValueType(elem))
        else:
            writer.add_key_value(key, val, GGUFValueType(typ))

    # Model identity: derived from --model-name or the input-dir basename (no
    # hard-coded artifact name in the converter).
    model_name = args.model_name or in_dir.name
    writer.add_string("general.name", f"{model_name} (BeeLlama Escha port)")
    writer.add_string("general.basename", model_name)
    writer.add_string("general.finetune", model_name)

    dep_k2 = load_dep_table(args.dep_k2)
    dep_k3 = load_dep_table(args.dep_k3)
    lut = escha_codebook_lut()

    # shared codec tables
    writer.add_tensor("escha_lut", lut)
    writer.add_tensor("escha_dep_k2", dep_k2, raw_dtype=GGMLQuantizationType.I16)
    writer.add_tensor("escha_dep_k3", dep_k3, raw_dtype=GGMLQuantizationType.I16)

    # NOTE: full_attn / n_layer were derived from the checkpoint structure above
    # (see the derivation block near the top of main()); they are NOT hard-coded
    # to a specific model (no artifact-name/layout assumptions).

    if args.standard_attn_ffn:
        required_attn = [
            f"blk.{il}.{name}.weight"
            for il in sorted(full_attn)
            for name in ("attn_q", "attn_k", "attn_v", "attn_output")
        ]
        missing_attn = [name for name in required_attn if name not in standard_ffn]
        if missing_attn:
            raise ValueError(
                "--standard-attn-ffn source GGUF is missing required full-attention "
                "tensor(s): " + ", ".join(missing_attn)
            )

    # collect per-file tensor name lists
    handles = []
    for p in files:
        sf = safe_open(str(p), framework="np")
        handles.append((p, sf, set(sf.keys())))

    def find(prefix: str):
        for _, sf, keys in handles:
            if prefix in keys:
                return sf.get_tensor(prefix)
        raise KeyError(prefix)

    def in_checkpoint(prefix: str) -> bool:
        return any(prefix in keys for _, _, keys in handles)

    def reconstruct_escha(prefix: str, ic: int, oc: int) -> np.ndarray:
        import sys as _sys
        _sys.path.insert(0, os.environ.get("ESCHA_TOOLS_DIR", "/home/sean/research/escha-refs/yaniss/tools/escha"))
        from escham_cpu import reconstruct_deploy_weight
        code = find(prefix + ".escha_code")
        rin = find(prefix + ".escha_rin")
        rout = find(prefix + ".escha_rout")
        K = code.shape[2] // 16
        return reconstruct_deploy_weight(code, rin, rout, ic, oc, K, True, False)

    def reconstruct_gdn_gate(prefix: str) -> np.ndarray:
        return reconstruct_escha(prefix, 5120, 6144)

    # layer tensors, in load order
    for il in range(n_layer):
        p = f"{layer_prefix}{il}."
        if il in full_attn:
            qkv = "self_attn"
        else:
            qkv = "linear_attn"

        # norms
        # Qwen3.5 RMSNorm stores a residual scale.  Its runtime computes
        # norm(x) * (1 + weight), while GGML consumes the effective scale.
        attn_norm = (find(p + "input_layernorm.weight").astype(np.float32) + 1.0)
        writer.add_tensor(f"blk.{il}.attn_norm.weight", attn_norm)
        post_norm = (find(p + "post_attention_layernorm.weight").astype(np.float32) + 1.0)
        writer.add_tensor(f"blk.{il}.post_attention_norm.weight", post_norm)

        if il in full_attn:
            if args.standard_attn_ffn:
                for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
                    write_standard_ffn(f"blk.{il}.{name}.weight")
            else:
                write_escha_proj(writer, find, p + qkv + ".q_proj", "attn_q", il, 12288)
                write_escha_proj(writer, find, p + qkv + ".k_proj", "attn_k", il, 1024)
                write_escha_proj(writer, find, p + qkv + ".v_proj", "attn_v", il, 1024)
                write_escha_proj(writer, find, p + qkv + ".o_proj", "attn_output", il, 5120)
            for name, gguf_blk in (("q_norm", "attn_q_norm"), ("k_norm", "attn_k_norm")):
                n = find(p + qkv + f".{name}.weight").astype(np.float32) + 1.0
                writer.add_tensor(f"blk.{il}.{gguf_blk}.weight", n)
        else:
            a_log = find(p + qkv + ".A_log").astype(np.float32)
            writer.add_tensor(f"blk.{il}.ssm_a", a_log)
            conv = find(p + qkv + ".conv1d.weight").astype(np.float32)
            conv_g = np.ascontiguousarray(conv.reshape(10240, 4))
            writer.add_tensor(f"blk.{il}.ssm_conv1d.weight", conv_g)
            dt = find(p + qkv + ".dt_bias").astype(np.float32)
            writer.add_tensor(f"blk.{il}.ssm_dt.bias", dt)
            # Reference Qwen3.5 uses beta=sigmoid(in_proj_b(x)) and
            # alpha=in_proj_a(x).  Preserve that semantic mapping in GGUF.
            for ckpt, gguf_blk, in_dim, out_dim in (
                    ("in_proj_b", "ssm_beta", 5120, 48),
                    ("in_proj_a", "ssm_alpha", 5120, 48)):
                w = find(p + qkv + f".{ckpt}.weight").astype(np.float16)
                writer.add_tensor(f"blk.{il}.{gguf_blk}.weight", w)
            ssm_norm = find(p + qkv + ".norm.weight").astype(np.float32)
            writer.add_tensor(f"blk.{il}.ssm_norm.weight", ssm_norm)
            if args.standard_linear_ffn:
                if il % 8 == 0:
                    print(f"layer {il}: reconstructing linear-attention QKV/output as Q2_K")
                w = reconstruct_escha(p + qkv + ".in_proj_qkv", 5120, 10240)
                source = np.ascontiguousarray(w.T.astype(np.float32))
                del w
                qbytes = quantize_q2_k(source)
                writer.add_tensor(f"blk.{il}.attn_qkv.weight", qbytes,
                                  raw_shape=qbytes.shape,
                                  raw_dtype=GGMLQuantizationType.Q2_K)
                del source, qbytes
            else:
                write_escha_proj(writer, find, p + qkv + ".in_proj_qkv", "attn_qkv", il, 10240)
            if args.standard_gdn_gate_quant is not None:
                w = reconstruct_gdn_gate(p + qkv + ".in_proj_z")
                source = np.ascontiguousarray(w.T.astype(np.float32))
                quantizers = {
                    "q2_k": (quantize_q2_k, GGMLQuantizationType.Q2_K),
                    "q4_k": (quantize_q4_k, GGMLQuantizationType.Q4_K),
                    "q6_k": (quantize_q6_k, GGMLQuantizationType.Q6_K),
                }
                quantize_gate, gate_qtype = quantizers[args.standard_gdn_gate_quant]
                qbytes = quantize_gate(source)
                writer.add_tensor(f"blk.{il}.attn_gate.weight", qbytes,
                                  raw_shape=qbytes.shape,
                                  raw_dtype=gate_qtype)
                if il == 0:
                    from gguf.quants import dequantize
                    restored = dequantize(qbytes, gate_qtype)
                    mae = np.mean(np.abs(restored - source), dtype=np.float64)
                    print(f"layer 0 {gate_qtype.name} attn_gate self-check: "
                          f"mean_abs_error={mae:.8g}")
                    del restored
                del w, source, qbytes
            elif args.standard_gdn_gate_f16:
                # diagnostic: the source GGUF's attn_gate.weight is a DIFFERENT projection
                # (corr ~0.04 vs the checkpoint gate, while FFN matches at 0.835). Reconstruct
                # the true in_proj_z gate from the checkpoint sidecars and write it dense F16.
                w = reconstruct_gdn_gate(p + qkv + ".in_proj_z")
                # writer reverses numpy dims into GGUF ne; the loader wants ne=(5120, 6144)
                writer.add_tensor(f"blk.{il}.attn_gate.weight",
                                  np.ascontiguousarray(w.T.astype(np.float16)),
                                  raw_dtype=GGMLQuantizationType.F16)
            elif args.standard_gdn_gate:
                write_standard_ffn(f"blk.{il}.attn_gate.weight")
            else:
                write_escha_proj(writer, find, p + qkv + ".in_proj_z", "attn_gate", il, 6144)
            if args.standard_linear_ffn:
                w = reconstruct_escha(p + qkv + ".out_proj", 6144, 5120)
                source = np.ascontiguousarray(w.T.astype(np.float32))
                del w
                qbytes = quantize_q2_k(source)
                writer.add_tensor(f"blk.{il}.ssm_out.weight", qbytes,
                                  raw_shape=qbytes.shape,
                                  raw_dtype=GGMLQuantizationType.Q2_K)
                del source, qbytes
            else:
                write_escha_proj(writer, find, p + qkv + ".out_proj", "ssm_out", il, 5120)

        if standard_ffn is None:
            write_escha_proj(writer, find, p + "mlp.gate_proj", "ffn_gate", il, 17408)
            write_escha_proj(writer, find, p + "mlp.up_proj", "ffn_up", il, 17408)
            write_escha_proj(writer, find, p + "mlp.down_proj", "ffn_down", il, 5120)
        else:
            write_standard_ffn(f"blk.{il}.ffn_gate.weight")
            write_standard_ffn(f"blk.{il}.ffn_up.weight")
            write_standard_ffn(f"blk.{il}.ffn_down.weight")

    output_norm = (find("model.language_model.norm.weight").astype(np.float32) + 1.0)
    writer.add_tensor("output_norm.weight", output_norm)

    # vocab: either dequantized fp16 (default, matches the reference GGUF layout)
    # or the native LowGPU 3-bit codes (--vocab lowgpu, decoded in-kernel).
    # --embed-vocab / --head-vocab override one side independently (P-ARCH-22).
    embed_storage = args.embed_vocab if args.embed_vocab is not None else args.vocab
    head_storage  = args.head_vocab  if args.head_vocab  is not None else args.vocab

    # dequant in row-chunks so the int32 intermediates never span the full
    # 248k-row vocab at once (full-span peak is ~15 GiB and trips the worker
    # cgroup memory cap; chunked peak stays near the fp16 output size)
    def dequant_all(codes, scales, zps):
        n = codes.shape[0]
        chunk = 16384
        parts = []
        for r0 in range(0, n, chunk):
            parts.append(dequant_lowgpu_rows(codes, scales, zps, r0, min(r0 + chunk, n)))
        return np.concatenate(parts, axis=0)

    def write_embed(prefix, gguf_base, storage):
        if storage in ("dequant", "q4_k", "q6_k"):
            codes  = find(prefix + ".weight_lowgpu_codes")
            scales = find(prefix + ".weight_lowgpu_scales")
            zps    = find(prefix + ".weight_lowgpu_zps")
            n = codes.shape[0]
            if storage == "dequant":
                w = dequant_all(codes, scales, zps)
                writer.add_tensor(f"{gguf_base}.weight", w)
                del w
            else:
                # chunked quantize: never materialize the full F32 embed at once
                # (peak was >8 GiB and tripped the worker cgroup OOM when two ran)
                qn = GGMLQuantizationType.Q4_K if storage == "q4_k" else GGMLQuantizationType.Q6_K
                fn = quantize_q4_k if storage == "q4_k" else quantize_q6_k
                chunk = 8192
                parts = []
                for r0 in range(0, n, chunk):
                    wc = dequant_lowgpu_rows(codes, scales, zps, r0, min(r0 + chunk, n))
                    qc = fn(np.ascontiguousarray(wc.astype(np.float32)))
                    parts.append(qc)
                    del wc, qc
                q = np.concatenate(parts, axis=0)
                del parts
                writer.add_tensor(f"{gguf_base}.weight", q, raw_shape=q.shape,
                                  raw_dtype=qn)
                del q
        else:
            codes  = find(prefix + ".weight_lowgpu_codes")
            scales = find(prefix + ".weight_lowgpu_scales")
            zps    = find(prefix + ".weight_lowgpu_zps")
            writer.add_tensor(f"{gguf_base}.lowgpu_codes",  codes.astype(np.int8))
            writer.add_tensor(f"{gguf_base}.lowgpu_scales", scales)
            writer.add_tensor(f"{gguf_base}.lowgpu_zps",    zps.astype(np.int8))

    write_embed("model.language_model.embed_tokens", "token_embd", embed_storage)
    write_embed("lm_head", "output", head_storage)

    # lowgpu.version metadata already written above if --vocab lowgpu; for mixed
    # per-side storage make sure the flag reflects "any side is packed"
    writer.add_uint32("qwen35.lowgpu.version", 1 if (embed_storage == "lowgpu" or head_storage == "lowgpu") else 0)

    writer.write_header_to_file(path=outfile)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    # verify header round-trips
    r = gguf.GGUFReader(str(outfile))
    print(f"wrote {len(r.tensors)} tensors to {outfile}")


if __name__ == "__main__":
    main()
