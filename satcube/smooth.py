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


def _process_spatial_chunk_smooth(
    data_chunk: np.ndarray,
    data_clim: np.ndarray,
    data_month: pd.Series,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    """
    Process a spatial chunk with climatology subtraction and SG smoothing.

    Args:
        data_chunk: Array of shape (Time, Bands, Chunk_Height, Chunk_Width).
        data_clim: Climatology array (12, Bands, Chunk_Height, Chunk_Width).
        data_month: Month indices for each timestep.
        window_length: SG filter window length.
        polyorder: SG filter polynomial order.

    Returns:
        Processed chunk of same shape.
    """
    T, C, H, W = data_chunk.shape

    # Subtract climatology
    for idx, month in enumerate(data_month):
        data_chunk[idx] = data_chunk[idx] - data_clim[month - 1]

    # Apply SG smoothing
    # Adjust window if needed
    if window_length % 2 == 0:
        window_length += 1
    if window_length > T:
        window_length = T if T % 2 != 0 else T - 1

    if window_length >= 3:
        data_chunk = savgol_filter(
            data_chunk,
            window_length=window_length,
            polyorder=polyorder,
            axis=0,
            mode="interp",
        )

    # Add climatology back
    for idx, month in enumerate(data_month):
        data_chunk[idx] = data_chunk[idx] + data_clim[month - 1]

    return data_chunk.astype(np.float32)


def _get_optimal_smooth_workers() -> int:
    """Auto-detect optimal number of workers for smoothing."""
    import os

    cpu_count = os.cpu_count() or 4
    return min(cpu_count // 2, 4)


def smooth_fn(
    metadata: pd.DataFrame,
    input_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path = "smoothed",
    *,
    smooth_w: int = 7,
    smooth_p: int = 2,
    num_workers: int | None = None,
    chunk_size: int = 512,
    quiet: bool = False,
) -> pd.DataFrame:
    """
        Apply Savitzky-Golay smoothing with climatology removal using parallel spatial processing.

        **⚠️ INTERPOLATE vs SMOOTH - Key Differences:**

        These functions are COMPLEMENTARY, not redundant:

        - **interpolate()**: CORRECTS data errors
          - Fills empty months (0 values) via linear interpolation
          - REMOVES extreme outliers (spikes >threshold from rolling median)
          - Example: Spike of 0.65 → removed and interpolated
          - Use: ALWAYS after compositing (fixes missing data + artifacts)

        - **smooth()**: REFINES clean data
          - Reduces gradual noise in COMPLETE time series
          - Smooths ALL values (not just outliers)
          - Example: [0.18, 0.21, 0.19] → [0.18, 0.20, 0.19] (subtle smoothing)
          - Use: OPTIONAL after interpolate (improves signal quality)

        **Pipeline order:**
    ```
        composite() → interpolate() → [smooth()] → analysis
                      ↑ Fixes errors  ↑ Optional refinement
    ```

        **Workflow:**
            1. Load all timesteps into memory
            2. Calculate monthly climatology (median for each month)
            3. Subtract climatology to get anomalies
            4. Apply Savitzky-Golay smoothing to anomalies (parallel by spatial chunks)
            5. Add climatology back
            6. Save smoothed images

        Args:
            metadata: DataFrame with scene metadata (must contain 'date' and 'outname' columns).
            input_dir: Directory containing input .tif files (typically from interpolate()).
            output_dir: Output directory for smoothed images. Default "smoothed".
            smooth_w: Savitzky-Golay filter window length (must be odd). Default 7.
            smooth_p: Savitzky-Golay filter polynomial order. Default 2.
            num_workers: Number of spatial chunks to process in parallel. If None,
                auto-detects (typically CPU_cores/2, max 4). Default None.
            chunk_size: Initial spatial chunk size in pixels. Auto-reduces if needed.
                Default 512.
            quiet: Suppress progress bars and logging. Default False.

        Returns:
            Updated metadata DataFrame with smoothed image info.

        Raises:
            FileNotFoundError: If input files not found.

        Notes:
            - Requires loading full time series into memory
            - Input should be COMPLETE series (no 0s) - run interpolate() first
            - Climatology removal helps preserve seasonal patterns while smoothing noise
            - Parallel processing by spatial chunks for efficiency
            - SG filter window auto-adjusts for short time series

        Examples:
            >>> # Standard pipeline (recommended)
            >>> composites = gapfilled.composite()
            >>> interp = composites.interpolate()  # Fixes errors
            >>> smoothed = interp.smooth()         # Optional refinement

            >>> # Skip smoothing for extreme event analysis
            >>> composites = gapfilled.composite()
            >>> interp = composites.interpolate()  # Stop here

            >>> # Custom smoothing parameters
            >>> smoothed = smooth_fn(
            ...     metadata=meta,
            ...     input_dir="interpolated",
            ...     smooth_w=9,
            ...     smooth_p=3
            ... )

            >>> # Manual workers
            >>> smoothed = smooth_fn(
            ...     metadata=meta,
            ...     input_dir="interpolated",
            ...     num_workers=4
            ... )
    """

    input_dir = pathlib.Path(input_dir).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_raw_files = [input_dir / fname for fname in metadata["outname"]]
    out_files = [output_dir / f.name for f in all_raw_files]

    if not all_raw_files[0].exists():
        raise FileNotFoundError(f"Input files not found in {input_dir}")

    profile = rio.open(all_raw_files[0]).profile

    data_np = (np.array([rio.open(f).read() for f in all_raw_files]) / 10000).astype(
        np.float32
    )

    T, C, H, W = data_np.shape

    # Calculate climatology
    data_month = pd.to_datetime(metadata["date"]).dt.month
    data_clim = []

    for month in range(1, 13):
        month_mask = data_month == month
        if np.any(month_mask):
            median_val = np.nanmedian(data_np[month_mask], axis=0)
        else:
            median_val = np.zeros(data_np.shape[1:], dtype=np.float32)
        data_clim.append(median_val)

    data_clim = np.array(data_clim)

    if num_workers is None:
        num_workers = _get_optimal_smooth_workers()

    effective_chunk_size = min(chunk_size, max(H, W))

    row_chunks = list(range(0, H, effective_chunk_size))
    col_chunks = list(range(0, W, effective_chunk_size))
    total_chunks = len(row_chunks) * len(col_chunks)

    if total_chunks < num_workers and effective_chunk_size > 128:
        target_chunks = num_workers
        optimal_size = max(128, int(min(H, W) / np.sqrt(target_chunks)))

        effective_chunk_size = optimal_size
        row_chunks = list(range(0, H, effective_chunk_size))
        col_chunks = list(range(0, W, effective_chunk_size))
        total_chunks = len(row_chunks) * len(col_chunks)

    output_data = np.zeros_like(data_np)

    chunk_coords = [
        (r, c, min(r + effective_chunk_size, H), min(c + effective_chunk_size, W))
        for r in row_chunks
        for c in col_chunks
    ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_spatial_chunk_smooth,
                data_chunk=data_np[:, :, r_start:r_end, c_start:c_end].copy(),
                data_clim=data_clim[:, :, r_start:r_end, c_start:c_end],
                data_month=data_month,
                window_length=smooth_w,
                polyorder=smooth_p,
            ): (r_start, c_start, r_end, c_end)
            for r_start, c_start, r_end, c_end in chunk_coords
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Smoothing",
            unit="chunk",
            disable=quiet,
        ):
            r_start, c_start, r_end, c_end = futures[future]
            try:
                result_chunk = future.result()
                output_data[:, :, r_start:r_end, c_start:c_end] = result_chunk
            except Exception:
                logger.exception(f"Failed to process chunk ({r_start}, {c_start})")

    for idx, file in enumerate(out_files):
        to_save = np.clip(output_data[idx], 0, 1.0)
        with rio.open(file, "w", **profile) as dst:
            dst.write((to_save * 10000).astype(np.uint16))

    if not quiet:
        logger.info(f"✓ Smoothed {len(all_raw_files)} images")

    new_table = pd.DataFrame(
        {
            "id": metadata["outname"].apply(lambda x: pathlib.Path(x).stem),
            "date": metadata["date"],
            "outname": metadata["outname"],
        }
    )

    return new_table


def sg_smooth(
    data: np.ndarray, window_length: int = 7, polyorder: int = 2
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing on the time dimension (axis 0).

    Legacy function kept for backwards compatibility.

    Args:
        data (np.ndarray): Input array of shape (Time, Bands, Height, Width).
        window_length (int): The length of the filter window (must be odd).
        polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
        np.ndarray: Smoothed data preserving the input shape and dtype.
    """
    if window_length % 2 == 0:
        window_length += 1

    time_len = data.shape[0]
    if time_len < window_length:
        window_length = time_len if time_len % 2 != 0 else time_len - 1

    if window_length < 3:
        return data

    smoothed = savgol_filter(
        data, window_length=window_length, polyorder=polyorder, axis=0, mode="interp"
    )

    return smoothed.astype(data.dtype)
