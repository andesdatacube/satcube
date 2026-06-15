from __future__ import annotations

import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import tqdm

from satcube.logging_config import setup_logger

logger = setup_logger(__name__)


def _process_month(
    month_date: str,
    images: list[pathlib.Path],
    profile: dict,
    output_dir: pathlib.Path,
    agg_method: str,
) -> dict:
    """
    Process a single month by aggregating all images for that month.

    Args:
        month_date: Month identifier in 'YYYY-MM-15' format.
        images: List of image paths for this month.
        profile: Rasterio profile template.
        output_dir: Output directory.
        agg_method: Aggregation method ('median', 'mean', 'max', 'min').

    Returns:
        Dictionary with month metadata.
    """

    if len(images) == 0:
        data = np.zeros(
            (profile["count"], profile["height"], profile["width"]), dtype=np.uint16
        )
        nodata = 0
        profile_image = profile
    else:
        container = []
        for image in images:
            with rio.open(image) as src:
                data = src.read()
                profile_image = src.profile
            container.append(data)

        if agg_method == "mean":
            data = np.mean(container, axis=0)
        elif agg_method == "median":
            data = np.median(container, axis=0)
        elif agg_method == "max":
            data = np.max(container, axis=0)
        elif agg_method == "min":
            data = np.min(container, axis=0)
        else:
            raise ValueError(f"Invalid aggregation method: {agg_method}")

        nodata = 0

    output_path = output_dir / f"{month_date}.tif"
    with rio.open(output_path, "w", **profile_image) as dst:
        dst.write(data.astype(np.uint16))

    return {
        "outname": f"{month_date}.tif",
        "date": month_date,
        "nodata": nodata,
    }


def _get_optimal_composite_workers() -> int:
    """
    Auto-detect optimal number of workers for compositing.

    Compositing is I/O and CPU intensive (reading multiple images + aggregation),
    so we use a balanced worker count.

    Returns:
        Optimal number of ThreadPoolExecutor workers.
    """
    import os

    cpu_count = os.cpu_count() or 4
    return min(cpu_count, 8)


def monthly_composites_s2(
    metadata: pd.DataFrame | None = None,
    input_dir: pathlib.Path | None = None,
    output_dir: pathlib.Path = pathlib.Path("monthly_composites"),
    agg_method: str = "median",
    num_workers: int | None = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Generate monthly composites from time series imagery using parallel processing.

    Aggregates all images within each month into a single composite image.
    Each month is processed in parallel for optimal performance.

    Args:
        metadata: DataFrame with scene metadata (must contain 'id' and 'date' columns).
            If None, scans input_dir for .tif files.
        input_dir: Directory containing input .tif files.
        output_dir: Output directory for composite images. Default "monthly_composites".
        agg_method: Aggregation method. Options:
            - 'median': Median value across time (recommended, robust to outliers)
            - 'mean': Mean value across time
            - 'max': Maximum value across time
            - 'min': Minimum value across time
        num_workers: Number of months to process in parallel. If None, auto-detects
            optimal value (typically CPU cores, capped at 8). Default None.
        quiet: Suppress progress bars. Default False.

    Returns:
        DataFrame with composite metadata containing columns:
            - outname: Output filename
            - date: Month date in 'YYYY-MM-15' format
            - nodata: Nodata value

    Notes:
        - Each month is centered on the 15th day (YYYY-MM-15)
        - Months with no images will generate empty composites
        - Parallel processing provides ~4-8x speedup on multi-core systems
        - Median is recommended as it's robust to cloud artifacts and outliers

    Examples:
        >>> # Auto-detect workers (recommended)
        >>> composites = monthly_composites_s2(
        ...     metadata=meta,
        ...     input_dir="gapfilled"
        ... )

        >>> # Manual workers with mean aggregation
        >>> composites = monthly_composites_s2(
        ...     metadata=meta,
        ...     input_dir="gapfilled",
        ...     agg_method="mean",
        ...     num_workers=4
        ... )

        >>> # Quiet mode for automation
        >>> composites = monthly_composites_s2(
        ...     metadata=meta,
        ...     input_dir="gapfilled",
        ...     quiet=True
        ... )
    """

    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_raw_files = metadata["id"].apply(lambda s: input_dir / f"{s}.tif").tolist()

    with rio.open(all_raw_files[0]) as src:
        profile = src.profile

    all_raw_dates = pd.to_datetime(metadata["date"])
    all_raw_dates_unique = pd.date_range(
        start=all_raw_dates.min().to_period("M").to_timestamp(),
        end=all_raw_dates.max().to_period("M").to_timestamp(),
        freq="MS",
    ) + pd.DateOffset(days=14)
    all_raw_dates_unique = all_raw_dates_unique.strftime("%Y-%m-15")

    if num_workers is None:
        num_workers = _get_optimal_composite_workers()

    month_to_images = {}
    for date_str in all_raw_dates_unique:
        idxs = all_raw_dates.dt.strftime("%Y-%m-15") == date_str
        images = [all_raw_files[i] for i in np.where(idxs)[0]]
        month_to_images[date_str] = images

    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_month,
                month_date=date_str,
                images=images,
                profile=profile,
                output_dir=output_dir,
                agg_method=agg_method,
            ): date_str
            for date_str, images in month_to_images.items()
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Compositing",
            unit="month",
            disable=quiet,
        ):
            month_date = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception:
                logger.exception(f"Failed to composite {month_date}")
                results.append(
                    {"outname": f"{month_date}.tif", "date": month_date, "nodata": 0}
                )

    if not quiet:
        logger.info(f"✓ Created {len(results)} monthly composites")

    return pd.DataFrame(results)
