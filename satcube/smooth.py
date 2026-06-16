from __future__ import annotations

import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio as rio
from scipy.signal import savgol_filter
from tqdm import tqdm

from satcube.logging_config import setup_logger

logger = setup_logger(__name__)


def _process_spatial_chunk_smooth(data_chunk, data_clim, data_month, window_length, polyorder):
    T, C, H, W = data_chunk.shape
    for idx, month in enumerate(data_month):
        data_chunk[idx] -= data_clim[month - 1]
    if window_length % 2 == 0:
        window_length += 1
    if window_length > T:
        window_length = T if T % 2 != 0 else T - 1
    if window_length >= 3:
        data_chunk = savgol_filter(data_chunk, window_length=window_length, polyorder=polyorder, axis=0, mode="interp")
    for idx, month in enumerate(data_month):
        data_chunk[idx] += data_clim[month - 1]
    return data_chunk.astype(np.float32)


def _get_optimal_smooth_workers():
    import os
    return min((os.cpu_count() or 4) // 2, 4)


def smooth_fn(metadata, input_dir, output_dir="smoothed", *, smooth_w=7, smooth_p=2,
              num_workers=None, chunk_size=512, cache=False, quiet=False):
    """Savitzky-Golay smoothing with monthly climatology removal, so the seasonal
    cycle is preserved. Note: with one image per calendar month it is near-identity;
    it matters with multi-year series. cache=True skips if outputs exist."""
    input_dir = pathlib.Path(input_dir).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_raw_files = [input_dir / fname for fname in metadata["outname"]]
    out_files = [output_dir / f.name for f in all_raw_files]
    if not all_raw_files[0].exists():
        raise FileNotFoundError(f"Input files not found in {input_dir}")

    new_table = pd.DataFrame({"id": metadata["outname"].apply(lambda x: pathlib.Path(x).stem),
                              "date": metadata["date"], "outname": metadata["outname"]})
    if cache and all(o.exists() for o in out_files):
        if not quiet:
            logger.info("cache hit, skipping smooth")
        return new_table

    with rio.open(all_raw_files[0]) as src:
        profile = src.profile
    data_np = np.array([rio.open(f).read() for f in all_raw_files]).astype(np.float32) / 10000.0
    T, C, H, W = data_np.shape

    data_month = pd.to_datetime(metadata["date"]).dt.month.to_numpy()
    clim = []
    for month in range(1, 13):
        m = data_month == month
        clim.append(np.nanmedian(data_np[m], axis=0) if np.any(m) else np.zeros(data_np.shape[1:], dtype=np.float32))
    data_clim = np.array(clim, dtype=np.float32)

    if num_workers is None:
        num_workers = _get_optimal_smooth_workers()
    eff = min(chunk_size, max(H, W))
    rcs = list(range(0, H, eff)); ccs = list(range(0, W, eff))
    if len(rcs) * len(ccs) < num_workers and eff > 128:
        eff = max(128, int(min(H, W) / np.sqrt(num_workers)))
        rcs = list(range(0, H, eff)); ccs = list(range(0, W, eff))

    output_data = np.zeros_like(data_np)
    coords = [(r, c, min(r + eff, H), min(c + eff, W)) for r in rcs for c in ccs]
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_process_spatial_chunk_smooth,
                             data_chunk=data_np[:, :, r0:r1, c0:c1].copy(),
                             data_clim=data_clim[:, :, r0:r1, c0:c1],
                             data_month=data_month, window_length=smooth_w, polyorder=smooth_p): (r0, c0, r1, c1)
                   for r0, c0, r1, c1 in coords}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Smoothing", unit="chunk", disable=quiet):
            r0, c0, r1, c1 = futures[fut]
            try:
                output_data[:, :, r0:r1, c0:c1] = fut.result()
            except Exception:
                logger.exception(f"Failed chunk ({r0},{c0})")
    for idx, f in enumerate(out_files):
        with rio.open(f, "w", **profile) as dst:
            dst.write((np.clip(output_data[idx], 0, 1.0) * 10000).astype(np.uint16))
    if not quiet:
        logger.info(f"✓ Smoothed {len(all_raw_files)} images")
    return new_table