from __future__ import annotations

import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import numpy as np
import pandas as pd
import rasterio as rio
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from tqdm import tqdm

from satcube.logging_config import setup_logger

logger = setup_logger(__name__)

_GAP_METHOD = Literal["histogram_matching", "linear"]


def _fill_one(
    img_path: pathlib.Path,
    ref_paths: list[pathlib.Path],
    dates: np.ndarray,
    this_date: np.datetime64,
    *,
    method: _GAP_METHOD,
    out_dir: pathlib.Path,
    threshold: float,
    enable_inpainting: bool,
) -> tuple[str, float]:
    """
    Fill gaps in a single image using temporal neighbors with BEST match selection.

    New strategy:
        - Evaluates up to 10 temporal neighbors (not just first valid one)
        - Selects the reference with LOWEST color matching error
        - Falls back to more distant images if needed

    Binary dilation parameters:
        - target_hole (iterations=25): Expanded from 20 to cover larger affected areas
        - hole_expanded (iterations=12): Expanded from 10 for wider search
        - local_context (iterations=25): Expanded from 20 for better histogram matching
    """

    with rio.open(img_path) as src:
        data = src.read() / 1e4
        prof = src.profile

        invalid_mask = (data <= 0) | (data > 1.0)
        data[invalid_mask] = np.nan
        base_missing_mask = np.isnan(data).any(axis=0)

    final_data = data.copy()

    if base_missing_mask.sum() == 0:
        _save_image(final_data, out_dir / img_path.name, prof)
        return img_path.stem, 0.0

    labeled_array, num_features = label(base_missing_mask)
    idxs = np.argsort(np.abs(dates - this_date))

    # NEW: Process each hole independently
    for region_idx in range(1, num_features + 1):

        current_region_mask = labeled_array == region_idx
        # Increased from 20 to 25 iterations for better coverage
        target_hole = binary_dilation(current_region_mask, iterations=10)

        still_missing = np.isnan(final_data).any(axis=0)
        target_hole = target_hole & still_missing

        if target_hole.sum() == 0:
            continue

        # NEW: Collect multiple candidates instead of using first valid one
        candidates = []
        max_candidates = 10  # Evaluate up to 10 temporal neighbors

        for i in idxs:
            if len(candidates) >= max_candidates:
                break

            ref_path = ref_paths[i]
            if ref_path.name == img_path.name:
                continue

            try:
                with rio.open(ref_path) as src:
                    ref = src.read() / 1e4
                    ref_invalid = (ref <= 0) | (ref > 1.0)
                    ref[ref_invalid] = np.nan
                    ref_valid_mask = ~np.isnan(ref).any(axis=0)
            except Exception:
                continue

            # Increased from 10 to 12 iterations
            hole_expanded = binary_dilation(target_hole, iterations=5)
            fillable_mask = hole_expanded & ref_valid_mask

            intersection = fillable_mask & target_hole
            coverage = intersection.sum() / (target_hole.sum() + 1e-6)

            # Lowered threshold from 0.90 to 0.85 to accept more candidates
            if coverage < 0.85:
                continue

            # Increased from 20 to 25 iterations
            local_context = binary_dilation(fillable_mask, iterations=15)
            valid_in_target = ~np.isnan(final_data).any(axis=0)
            training_mask = valid_in_target & ref_valid_mask & local_context

            if training_mask.sum() < 20:
                continue

            filled_patch = np.zeros_like(ref)
            patch_error = 0.0
            rgb_bands = [0, 1, 2] if data.shape[0] > 2 else [0]
            band_valid = True

            for b in range(data.shape[0]):
                valid_tgt = final_data[b][training_mask]
                valid_ref = ref[b][training_mask]
                pixels_fill = ref[b][fillable_mask]

                nan_tgt_ratio = np.isnan(valid_tgt).sum() / len(valid_tgt)
                nan_ref_ratio = np.isnan(valid_ref).sum() / len(valid_ref)

                if nan_tgt_ratio > 0.5 or nan_ref_ratio > 0.5:
                    band_valid = False
                    break

                valid_tgt_clean = valid_tgt[~np.isnan(valid_tgt)]
                valid_ref_clean = valid_ref[~np.isnan(valid_ref)]

                if len(valid_tgt_clean) < 10 or len(valid_ref_clean) < 10:
                    band_valid = False
                    break

                if method == "histogram_matching":
                    hist_t, bins = np.histogram(valid_tgt_clean, 128, [0, 1.0])
                    hist_r, _ = np.histogram(valid_ref_clean, 128, [0, 1.0])
                    cdf_t = hist_t.cumsum() / (hist_t.sum() + 1e-10)
                    cdf_r = hist_r.cumsum() / (hist_r.sum() + 1e-10)
                    lut = np.interp(cdf_r, cdf_t, bins[:-1])
                    matched = np.interp(pixels_fill, bins[:-1], lut)
                    val_pixels = np.interp(valid_ref_clean, bins[:-1], lut)
                else:
                    try:
                        A = np.vstack(
                            [valid_ref_clean, np.ones(len(valid_ref_clean))]
                        ).T
                        m, c = np.linalg.lstsq(A, valid_tgt_clean, rcond=None)[0]
                        matched = pixels_fill * m + c
                        val_pixels = valid_ref_clean * m + c
                    except Exception:
                        matched = pixels_fill
                        val_pixels = valid_ref_clean

                matched = np.clip(matched, 0.0, 1.0)
                filled_patch[b][fillable_mask] = matched

                if b in rgb_bands:
                    diff = np.abs(valid_tgt_clean - val_pixels)
                    patch_error += np.mean(diff)

            if not band_valid:
                continue

            current_metric = patch_error / len(rgb_bands)

            # NEW: Store candidate with its error score
            candidates.append(
                {
                    "ref": ref,
                    "filled_patch": filled_patch,
                    "fillable_mask": fillable_mask,
                    "error": current_metric,
                    "coverage": coverage,
                }
            )

        # Select BEST candidate (lowest error)
        if len(candidates) == 0:
            continue

        # Sort by error (ascending) - best candidate first
        candidates.sort(key=lambda x: x["error"])
        best = candidates[0]

        # Only use if error is acceptable
        if best["error"] > threshold:
            continue

        # Apply the BEST candidate
        dist_map = distance_transform_edt(best["fillable_mask"])
        alpha = np.clip(dist_map / 5.0, 0, 1)

        for b in range(data.shape[0]):
            p = best["filled_patch"][b][best["fillable_mask"]]
            c = final_data[b][best["fillable_mask"]]
            a = alpha[best["fillable_mask"]]

            c_nan = np.isnan(c)
            if c_nan.any():
                c[c_nan] = p[c_nan]
                a[c_nan] = 1.0

            blended = (p * a) + (c * (1.0 - a))
            blended = np.clip(blended, 0.0, 1.0)

            final_data[b][best["fillable_mask"]] = blended

    # Rest remains the same (inpainting, etc.)
    if enable_inpainting:
        bad_mask = np.isnan(final_data) | (final_data <= 0.00005) | (final_data > 1.0)
        bad_2d = bad_mask.any(axis=0)

        if bad_2d.sum() > 0:
            valid_2d = ~bad_2d
            if valid_2d.sum() > 0:
                y_v, x_v = np.where(valid_2d)
                pts = np.column_stack((y_v, x_v))
                y_b, x_b = np.where(bad_2d)

                for b in range(data.shape[0]):
                    vals = final_data[b][valid_2d]
                    try:
                        fill = griddata(pts, vals, (y_b, x_b), method="nearest")
                        final_data[b][bad_2d] = np.clip(fill, 0.0, 1.0)
                    except Exception:
                        pass

    _save_image(final_data, out_dir / img_path.name, prof)

    final_gaps = np.isnan(final_data) | (final_data <= 0.00005)
    remaining_mask = final_gaps.any(axis=0)
    height, width = final_data.shape[1], final_data.shape[2]
    remaining_gaps_pct = (remaining_mask.sum() / (height * width)) * 100.0

    return img_path.stem, remaining_gaps_pct


def _fill_image_complete(
    img_idx: int,
    img_paths: list[pathlib.Path],
    filled_paths: list[pathlib.Path],
    dates: np.ndarray,
    rounds: list[dict],
    method: str,
    output_dir: pathlib.Path,
) -> tuple[str, float]:
    """
    Execute all 3 rounds for a single image.

    Args:
        img_idx: Index of the image to process.
        img_paths: List of all input image paths.
        filled_paths: List of all output image paths.
        dates: Array of dates for all images.
        rounds: List of round configurations.
        method: Color transfer method.
        output_dir: Output directory.

    Returns:
        Tuple of (image_id, remaining_gaps_pct_after_round1).
    """
    img_path = img_paths[img_idx]
    remaining_gaps_pct = 0.0

    try:
        for r_idx, r_data in enumerate(rounds):
            src_img = r_data["src"][img_idx]
            if not src_img.exists():
                src_img = img_paths[img_idx]

            refs = img_paths if r_idx == 0 else filled_paths

            _, gaps_pct = _fill_one(
                img_path=src_img,
                ref_paths=refs,
                dates=dates,
                this_date=dates[img_idx],
                method=method,
                out_dir=output_dir,
                threshold=r_data["thresh"],
                enable_inpainting=r_data["inpaint"],
            )

            if r_idx == 0:
                remaining_gaps_pct = gaps_pct

        return img_path.stem, remaining_gaps_pct

    except Exception:
        logger.exception(f"Failed to fill {img_path.stem}")
        return img_path.stem, 100.0


def _save_image(data: np.ndarray, path: pathlib.Path, profile: dict) -> None:
    """
    Save processed image array to GeoTIFF.

    Args:
        data: Image array in float32 [0-1] range.
        path: Output file path.
        profile: Rasterio profile for metadata.
    """
    data = np.nan_to_num(data, nan=0.0)
    data = np.clip(data, 0.0, 1.0)
    data_uint = (data * 1e4).astype(np.uint16)
    with rio.open(path, "w", **profile) as dst:
        dst.write(data_uint)


def _get_optimal_gapfill_workers() -> int:
    """Auto-detect optimal number of workers for gap filling."""
    import os

    cpu_count = os.cpu_count() or 4
    return min(cpu_count, 8)


def gapfill_fn(
    metadata: pd.DataFrame,
    input_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path = "gapfilled",
    *,
    method: _GAP_METHOD = "histogram_matching",
    num_workers: int | None = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Fill cloud/shadow gaps using multi-round temporal matching with parallel processing.

    Implements a sophisticated 3-round cascading strategy:
        1. **Strict Round** (threshold=0.15): High-quality matches only, no inpainting
        2. **Relaxed Round** (threshold=0.40): More tolerant matches for remaining gaps
        3. **Final Round** (threshold=∞): Inpainting for any remaining pixels

    Each gap is treated as an independent component, preventing artifacts from
    distant cloud regions merging together. Uses histogram matching for accurate
    color transfer and spatial blending for seamless transitions.

    **Binary dilation strategy:**
        - 20 iterations (~200m): Expand holes to cover hidden affected areas
        - 10 iterations (~100m): Search for reference pixels near holes
        - 20 iterations (~200m): Create training region for histogram matching

    Args:
        metadata: DataFrame with scene metadata (must contain 'id' and 'date' columns).
        input_dir: Directory containing input GeoTIFF files.
        output_dir: Output directory for gap-filled images. Default "gapfilled".
        method: Color transfer method. Options:
            - 'histogram_matching': Match full histogram distribution (recommended)
            - 'linear': Simple linear regression
        num_workers: Number of images to process in parallel. If None, auto-detects
            optimal value (typically equal to CPU cores, capped at 8). Default None.
        quiet: Suppress progress bars. Default False.

    Returns:
        Updated metadata DataFrame with 'remaining_gaps_pct' column showing the
        percentage of pixels that could not be filled via strict temporal matching (0-100).

    Examples:
        >>> filled = gapfill_fn(metadata=meta, input_dir="masked")

        >>> # Filter by quality
        >>> filled = filled[filled["remaining_gaps_pct"] < 10.0]
    """

    input_dir = pathlib.Path(input_dir).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = [input_dir / f"{i}.tif" for i in metadata["id"]]
    filled_paths = [output_dir / f"{i}.tif" for i in metadata["id"]]
    dates = pd.to_datetime(metadata["date"]).to_numpy()

    if num_workers is None:
        num_workers = _get_optimal_gapfill_workers()

    rounds = [
        {"name": "Strict", "thresh": 0.15, "inpaint": False, "src": img_paths},
        {"name": "Relaxed", "thresh": 0.40, "inpaint": False, "src": filled_paths},
        {"name": "Final", "thresh": 100.0, "inpaint": True, "src": filled_paths},
    ]

    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _fill_image_complete,
                img_idx=i,
                img_paths=img_paths,
                filled_paths=filled_paths,
                dates=dates,
                rounds=rounds,
                method=method,
                output_dir=output_dir,
            ): i
            for i in range(len(img_paths))
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Gap filling",
            unit="image",
            disable=quiet,
        ):
            img_idx = futures[future]
            try:
                img_id, remaining_gaps_pct = future.result()
                results.append({"id": img_id, "remaining_gaps_pct": remaining_gaps_pct})
            except Exception:
                logger.exception(f"Unexpected error for image {img_idx}")
                results.append(
                    {"id": img_paths[img_idx].stem, "remaining_gaps_pct": 100.0}
                )

    if not quiet:
        logger.info(f"✓ Gap filled {len(img_paths)} images")

    results_df = pd.DataFrame(results)

    metadata = metadata.drop(columns=["remaining_gaps_pct"], errors="ignore")
    metadata = metadata.merge(results_df, on="id", how="left")
    metadata["remaining_gaps_pct"] = metadata["remaining_gaps_pct"].fillna(100.0)

    return metadata
