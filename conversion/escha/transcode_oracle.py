"""Deterministic Escha reconstruction and GGML K-quant oracle.

This module is shared by the canonical converter and the EXP-11 cache builder.
Its arithmetic order, dtypes, and block layout are cache ABI.
"""
from __future__ import annotations

import numpy as np

ORACLE_ABI = "escha-reconstruct-cba-h128-fp32-v1"
QUANTIZER_ABI = "ggml-kquants-numpy-v1-batch4096"


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


MASK = np.uint32(0x8FFF8FFF)
MAGIC = np.uint32(0x3B603B60)
MUL = {0: None, 1: np.uint32(3417055213), 2: np.uint32(2212286765)}


def _codebook(cb: int, nbits: int) -> np.ndarray:
    """LUT: window value -> fp16 weight.  Size 2**nbits."""
    x = np.arange(1 << nbits, dtype=np.uint32)
    if cb == 2:
        t = (x * MUL[2]).astype(np.uint32)
    elif cb == 1:
        t = (x * MUL[1]).astype(np.uint32)
        t = (t & MASK) ^ MAGIC
    else:
        t = (x & MASK) ^ MAGIC
    lo = (t & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    hi = (t >> np.uint32(16)).astype(np.uint16).view(np.float16)
    return (lo.astype(np.float32) + hi.astype(np.float32)).astype(np.float16)


def _tile_map(K: int):
    """(lane, step) -> (row, col) inside the 16x16 tile.  Same for K=2 and K=3.

    r143 = (tid&3)*512 + warp*32 + ((lane>>1)&12); eight st.shared.u32 at byte
    offsets 0,256,2048,2304,16,272,2064,2320, each writing lane l's value then
    lane l+4's (shfl.sync.down delta 4).  Steps run high->low: offset 0 holds
    the step-7 window, offset 2320 the step-0 one.
    """
    rows = np.empty((32, 8), np.int64)
    cols = np.empty((32, 8), np.int64)
    off = {7: (0, 0), 6: (1, 0), 5: (8, 0), 4: (9, 0),
           3: (0, 8), 2: (1, 8), 1: (8, 8), 0: (9, 8)}
    for m in range(32):
        r0 = 2 * (m & 3)
        c0 = 2 * ((m >> 3) & 1) + 4 * ((m >> 4) & 1)
        d = (m >> 2) & 1
        for s in range(8):
            dr, dc = off[s]
            rows[m, s] = r0 + dr
            cols[m, s] = c0 + dc + d
    return rows, cols


def _lane_addr(K: int):
    """(cur_word, prev_word, shift) per lane, straight from the PTX prologue."""
    cur = np.empty(32, np.int64); prev = np.empty(32, np.int64)
    sh = np.empty(32, np.int64)
    for l in range(32):
        if K == 2:
            c = (l >> 1) & 15
            cur[l], prev[l], sh[l] = c, (c - 1) & 15, 16 if (l & 1) == 0 else 0
        elif K == 3:
            b = l * 24
            r86 = b + 791
            r87 = r86 & 2016
            cur[l] = (((r86 >> 3) & 252) - 96) // 4
            p = ((b + 755) >> 5) - 24
            prev[l] = 23 if l == 0 else p
            sh[l] = r87 - b - 760
        else:
            raise ValueError(K)
        assert prev[l] == (cur[l] - 1) % (16 if K == 2 else 24), (l, cur[l], prev[l])
    assert sh.min() >= 0 and sh.max() + 7 * K + 16 <= 64
    return cur, prev, sh


def _windows(words: np.ndarray, K: int) -> np.ndarray:
    """words: (T, 16*K/2) uint32 -> (T,32,8) 16-bit window values.

    Tail-biting: the window that starts near the end of a lane's slice wraps
    into the previous u32, which is why the kernel builds a 64-bit pair.
    """
    cur, prev, sh = _lane_addr(K)
    pair = ((words[:, prev].astype(np.uint64) << np.uint64(32))
            | words[:, cur].astype(np.uint64))
    steps = (sh[None, :, None] + K * np.arange(8)[None, None, :]).astype(np.uint64)
    return ((pair[:, :, None] >> steps) & np.uint64(0xFFFF)).astype(np.uint16)


def reconstruct_code(packed: np.ndarray, in_features: int, out_features: int,
                     K: int, cbA: bool, mul1: bool) -> np.ndarray:
    """packed: (IC//16, OC//16, 16*K) int16  ->  (IC, OC) float16."""
    cb = 1 if cbA else (2 if mul1 else 0)
    itc, otc = in_features // 16, out_features // 16
    assert packed.shape[:2] == (itc, otc), packed.shape
    words = np.ascontiguousarray(packed).view(np.uint32).reshape(itc * otc, 8 * K)
    win = _windows(words, K)                       # (T,32,8)
    lut = _codebook(cb, 16)
    vals = lut[win]                                # (T,32,8) f16
    rows, cols = _tile_map(K)
    tiles = np.empty((words.shape[0], 16, 16), np.float16)
    tiles[:, rows.ravel(), cols.ravel()] = vals.reshape(words.shape[0], -1)
    return (tiles.reshape(itc, otc, 16, 16).transpose(0, 2, 1, 3)
            .reshape(in_features, out_features))


_H128 = None


def _had128() -> np.ndarray:
    global _H128
    if _H128 is None:
        h = np.ones((1, 1), np.float32)
        while h.shape[0] < 128:
            h = np.block([[h, h], [h, -h]])
        _H128 = h / np.sqrt(128.0)
    return _H128


def escha_t128(x: np.ndarray) -> np.ndarray:
    """Blockwise normalised Hadamard-128 on the last axis (transform.py)."""
    s = x.shape
    return (x.reshape(-1, s[-1] // 128, 128) @ _had128()).reshape(s)


def reconstruct_deploy_weight(code, rin, rout, in_features, out_features,
                              K, cbA, mul1) -> np.ndarray:
    """linear.py:reconstruct_deploy_weight -> (IC, OC) float32."""
    w = reconstruct_code(code, in_features, out_features, K, cbA, mul1).astype(np.float32)
    w = escha_t128(w.T.copy()).T.copy()      # Hadamard along IC
    w = w * rin.astype(np.float32)[:, None]
    w = escha_t128(w)                        # Hadamard along OC
    w = w * rout.astype(np.float32)[None, :]
    return w


def reconstruct_standard_weight(code: np.ndarray, rin: np.ndarray, rout: np.ndarray,
                                in_features: int, out_features: int) -> np.ndarray:
    """Reconstruct GGUF sidecars and return contiguous stock (OC, IC) fp32."""
    if code.ndim != 3 or code.shape[2] not in (32, 48):
        raise ValueError(f"invalid Escha code shape: {code.shape}")
    k = code.shape[2] // 16
    weight = reconstruct_deploy_weight(
        code, rin, rout, in_features, out_features, k, True, False
    )
    return np.ascontiguousarray(weight.T.astype(np.float32))
