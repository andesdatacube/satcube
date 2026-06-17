from __future__ import annotations

import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import tqdm

from satcube.logging_config import setup_logger

logger = setup_logger(__name__)
_AGG = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min}


def _process_month(month_date, images, profile, output_dir, agg_method):
    """Aggregate all images of one month into a single composite (uint16)."""
    if len(images) == 0:
        data = np.zeros((profile["count"], profile["height"], profile["width"]), dtype=np.uint16)
        prof_img = profile
    else:
        if agg_method not in _AGG:
            raise ValueError(f"Invalid aggregation method: {agg_method}")
        container = []
        for image in images:
            with rio.open(image) as src:
                container.append(src.read().astype(np.float32))  # float32 avoids float64 median spike
                prof_img = src.profile
        data = _AGG[agg_method](np.stack(container, 0), axis=0)
    with rio.open(output_dir / f"{month_date}.tif", "w", **prof_img) as dst:
        dst.write(np.clip(data, 0, 65535).astype(np.uint16))
    return {"outname": f"{month_date}.tif", "date": month_date, "nodata": 0}


def _get_optimal_composite_workers():
    import os
    return min(os.cpu_count() or 4, 8)


def monthly_composites_s2(metadata=None, input_dir=None, output_dir=pathlib.Path("monthly_composites"),
                          agg_method="median", num_workers=None, cache=False, quiet=False):
    """Monthly composites (one per calendar month, centered on the 15th).

    Median is robust to residual cloud artifacts. Empty months produce a zero
    placeholder that interpolate() later fills. cache=True skips if outputs exist.
    """
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_raw_files = metadata["id"].apply(lambda s: pathlib.Path(input_dir) / f"{s}.tif").tolist()
    with rio.open(all_raw_files[0]) as src:
        profile = src.profile
    all_raw_dates = pd.to_datetime(metadata["date"])
    months = (pd.date_range(start=all_raw_dates.min().to_period("M").to_timestamp(),
                            end=all_raw_dates.max().to_period("M").to_timestamp(), freq="MS")
              + pd.DateOffset(days=14)).strftime("%Y-%m-15")

    if cache and all((output_dir / f"{d}.tif").exists() for d in months):
        if not quiet:
            logger.info("cache hit, skipping composite")
        return pd.DataFrame([{"outname": f"{d}.tif", "date": d, "nodata": 0} for d in months])

    if num_workers is None:
        num_workers = _get_optimal_composite_workers()

    month_to_images = {}
    for d in months:
        idxs = all_raw_dates.dt.strftime("%Y-%m-15") == d
        month_to_images[d] = [all_raw_files[i] for i in np.where(idxs)[0]]

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_process_month, month_date=d, images=imgs, profile=profile,
                             output_dir=output_dir, agg_method=agg_method): d
                   for d, imgs in month_to_images.items()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Compositing", unit="month", disable=quiet):
            d = futures[fut]
            try:
                results.append(fut.result())
            except Exception:
                logger.exception(f"Failed to composite {d}")
                results.append({"outname": f"{d}.tif", "date": d, "nodata": 0})
    if not quiet:
        logger.info(f"✓ Created {len(results)} monthly composites")
    return pd.DataFrame(results).sort_values("date").reset_index(drop=True)