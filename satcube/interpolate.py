from __future__ import annotations

import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import tqdm

from satcube.config import DESPIKE_THRESHOLD
from satcube.logging_config import setup_logger

logger = setup_logger(__name__)


def _rolling_median3(a):
    """Window-3 median along time (axis 0), edge padded. Fast replacement for
    scipy median_filter(size=(3,1,1,1)); matches it exactly."""
    ap = np.concatenate([a[:1], a, a[-1:]], axis=0)
    x, y, z = ap[0:-2], ap[1:-1], ap[2:]
    return np.maximum(np.minimum(x, y), np.minimum(np.maximum(x, y), z))


def _interp_axis0(a):
    """Vectorized linear interpolation of NaNs along time (axis 0), per column.
    Replaces the per-pixel Python loop; same result, far faster."""
    T, M = a.shape
    valid = ~np.isnan(a)
    rows = np.arange(T)[:, None]
    idx_prev = np.where(valid, rows, -1)
    np.maximum.accumulate(idx_prev, axis=0, out=idx_prev)
    idx_next = np.where(valid, rows, T)
    idx_next = np.minimum.accumulate(idx_next[::-1], axis=0)[::-1]
    idx_prev_c = np.clip(np.where(idx_prev < 0, idx_next, idx_prev), 0, T - 1)
    idx_next_c = np.clip(np.where(idx_next >= T, idx_prev, idx_next), 0, T - 1)
    col = np.arange(M)[None, :]
    v_prev = a[idx_prev_c, col]
    v_next = a[idx_next_c, col]
    denom = idx_next_c - idx_prev_c
    w = np.where(denom == 0, 0.0, (rows - idx_prev_c) / np.where(denom == 0, 1, denom))
    out = (v_prev + (v_next - v_prev) * w).astype(np.float32)
    out[:, ~valid.any(axis=0)] = np.nan
    return out


def _process_spatial_chunk(data_chunk, despike_threshold):
    rm = _rolling_median3(data_chunk)
    dc = data_chunk.copy()
    dc[np.abs(dc - rm) > despike_threshold] = np.nan
    T, C, H, W = dc.shape
    return _interp_axis0(dc.reshape(T, -1)).reshape(T, C, H, W)


def _get_optimal_interpolate_workers():
    import os
    return min((os.cpu_count() or 4) // 2, 4)


def interpolate_fn(metadata, input_dir, output_dir="interpolated", *, despike_threshold=DESPIKE_THRESHOLD,
                   num_workers=None, chunk_size=512, cache=False, quiet=False):
    """Despike (remove temporal spikes vs 3-month rolling median) and fill gaps by
    linear interpolation along time. cache=True skips if outputs exist."""
    input_dir = pathlib.Path(input_dir).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_raw_files = sorted([f for f in input_dir.glob("*.tif") if f.is_file()])
    if not all_raw_files:
        raise FileNotFoundError(f"No .tif files found in {input_dir}")

    if cache and all((output_dir / f.name).exists() for f in all_raw_files):
        if not quiet:
            logger.info("cache hit, skipping interpolate")
        return metadata

    with rio.open(all_raw_files[0]) as src:
        profile = src.profile
    data_np = np.array([rio.open(f).read() for f in all_raw_files]).astype(np.float32) / 10000.0
    data_np[data_np == 0] = np.nan
    T, C, H, W = data_np.shape

    if num_workers is None:
        num_workers = _get_optimal_interpolate_workers()
    eff = min(chunk_size, max(H, W))
    rcs = list(range(0, H, eff)); ccs = list(range(0, W, eff))
    if len(rcs) * len(ccs) < num_workers and eff > 128:
        eff = max(128, int(min(H, W) / np.sqrt(num_workers)))
        rcs = list(range(0, H, eff)); ccs = list(range(0, W, eff))

    output_data = np.zeros_like(data_np)
    coords = [(r, c, min(r + eff, H), min(c + eff, W)) for r in rcs for c in ccs]
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_process_spatial_chunk,
                             data_chunk=data_np[:, :, r0:r1, c0:c1], despike_threshold=despike_threshold): (r0, c0, r1, c1)
                   for r0, c0, r1, c1 in coords}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Interpolating", unit="chunk", disable=quiet):
            r0, c0, r1, c1 = futures[fut]
            try:
                output_data[:, :, r0:r1, c0:c1] = fut.result()
            except Exception:
                logger.exception(f"Failed chunk ({r0},{c0})")
    output_data = np.nan_to_num(output_data, nan=0.0)
    for idx, f in enumerate(all_raw_files):
        with rio.open(output_dir / f.name, "w", **profile) as dst:
            dst.write((np.clip(output_data[idx], 0, 1.0) * 10000).astype(np.uint16))
    if not quiet:
        logger.info(f"✓ Interpolated {len(all_raw_files)} images")
    return metadata