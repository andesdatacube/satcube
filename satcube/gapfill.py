from __future__ import annotations

import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import rasterio as rio
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from tqdm import tqdm

from satcube.logging_config import setup_logger

logger = setup_logger(__name__)

_GAP_METHOD = Literal["histogram_matching", "linear"]


def _load_ref(path: pathlib.Path, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read a reference image once and cache it for reuse across holes.

    Returns the reflectance array (with invalid pixels as nan) and its 2D
    valid mask. Caching avoids re-reading the same neighbor from disk for
    every hole in the target image.
    """
    if path in cache:
        return cache[path]

    with rio.open(path) as src:
        ref = src.read() / 1e4
    ref[(ref <= 0) | (ref > 1.0)] = np.nan
    valid = ~np.isnan(ref).any(axis=0)
    cache[path] = (ref, valid)
    return cache[path]


def _inpaint_bands(data: np.ndarray, bad_2d: np.ndarray, radius: int = 3) -> np.ndarray:
    """Smoothly fill the remaining bad pixels of each band with cv2 Telea.

    cv2.inpaint only supports 8-bit, so we scale to uint8 and back. That is
    fine here because this is the last-resort fill on the few pixels temporal
    matching could not cover, and the monthly median composite smooths it out.
    Replaces nearest-neighbor griddata, which left visible blocky patches.
    """
    mask = bad_2d.astype(np.uint8)
    out = data.copy()
    for b in range(data.shape[0]):
        band = np.nan_to_num(data[b], nan=0.0)
        band = np.clip(band, 0.0, 1.0)
        band_u8 = (band * 255.0).astype(np.uint8)
        filled_u8 = cv2.inpaint(band_u8, mask, radius, cv2.INPAINT_TELEA)
        filled = filled_u8.astype(np.float32) / 255.0
        out[b][bad_2d] = filled[bad_2d]
    return out


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

    Strategy:
        - Treats each hole (connected component) independently.
        - Evaluates up to 10 temporal neighbors and picks the LOWEST color error.
        - Blends the chosen patch with a distance-based feather.
        - Optional final inpainting (cv2 Telea) for anything left.

    Reference images are read once and cached, so a target with many holes does
    not re-read the same neighbors from disk per hole.
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

    # Cache of reference reads, reused across all holes in this image.
    ref_cache: dict = {}

    for region_idx in range(1, num_features + 1):

        current_region_mask = labeled_array == region_idx
        target_hole = binary_dilation(current_region_mask, iterations=10)

        still_missing = np.isnan(final_data).any(axis=0)
        target_hole = target_hole & still_missing

        if target_hole.sum() == 0:
            continue

        candidates = []
        max_candidates = 10

        for i in idxs:
            if len(candidates) >= max_candidates:
                break

            ref_path = ref_paths[i]
            if ref_path.name == img_path.name:
                continue

            try:
                ref, ref_valid_mask = _load_ref(ref_path, ref_cache)
            except Exception:
                continue

            hole_expanded = binary_dilation(target_hole, iterations=5)
            fillable_mask = hole_expanded & ref_valid_mask

            intersection = fillable_mask & target_hole
            coverage = intersection.sum() / (target_hole.sum() + 1e-6)

            if coverage < 0.85:
                continue

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

            candidates.append(
                {
                    "filled_patch": filled_patch,
                    "fillable_mask": fillable_mask,
                    "error": current_metric,
                    "coverage": coverage,
                }
            )

        if len(candidates) == 0:
            continue

        candidates.sort(key=lambda x: x["error"])
        best = candidates[0]

        if best["error"] > threshold:
            continue

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

    if enable_inpainting:
        bad_mask = np.isnan(final_data) | (final_data <= 0.00005) | (final_data > 1.0)
        bad_2d = bad_mask.any(axis=0)
        if bad_2d.sum() > 0:
            final_data = _inpaint_bands(final_data, bad_2d, radius=3)

    _save_image(final_data, out_dir / img_path.name, prof)

    final_gaps = np.isnan(final_data) | (final_data <= 0.00005)
    remaining_mask = final_gaps.any(axis=0)
    height, width = final_data.shape[1], final_data.shape[2]
    remaining_gaps_pct = (remaining_mask.sum() / (height * width)) * 100.0

    return img_path.stem, remaining_gaps_pct


def _save_image(data: np.ndarray, path: pathlib.Path, profile: dict) -> None:
    """Save processed image array to GeoTIFF."""
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
    Fill cloud/shadow gaps using multi-round temporal matching.

    Three cascading rounds, run with a barrier between them so each round reads
    the PREVIOUS round's outputs and writes its OWN, never a file another worker
    is writing (this is the deterministic, race-free version):
        1. Strict  (threshold=0.15): high-quality matches only, no inpainting.
        2. Relaxed (threshold=0.40): more tolerant matches for remaining gaps.
        3. Final   (threshold=inf): cv2 Telea inpainting for anything left.

    Each gap is treated as an independent component. Histogram matching gives
    accurate color transfer; a distance feather blends the seams.

    Args:
        metadata: DataFrame with scene metadata (needs 'id' and 'date' columns).
        input_dir: Directory containing input GeoTIFF files.
        output_dir: Output directory for gap-filled images. Default "gapfilled".
        method: 'histogram_matching' (recommended) or 'linear'.
        num_workers: Images processed in parallel per round. If None, auto-detects
            (CPU cores, capped at 8).
        quiet: Suppress progress bars. Default False.

    Returns:
        Updated metadata with 'remaining_gaps_pct' (percent of pixels the STRICT
        round could not fill via temporal matching, 0-100).

    Examples:
        >>> filled = gapfill_fn(metadata=meta, input_dir="masked")
        >>> filled = filled[filled["remaining_gaps_pct"] < 1.0]
    """
    input_dir = pathlib.Path(input_dir).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = list(metadata["id"])
    dates = pd.to_datetime(metadata["date"]).to_numpy()
    n = len(ids)

    if num_workers is None:
        num_workers = _get_optimal_gapfill_workers()

    # Separate dir per round so reads (previous round) and writes (current round)
    # never touch the same file. Cleaned up at the end.
    r0_dir = output_dir / "_round0"
    r1_dir = output_dir / "_round1"
    r0_dir.mkdir(parents=True, exist_ok=True)
    r1_dir.mkdir(parents=True, exist_ok=True)

    rounds = [
        {"name": "Strict", "thresh": 0.15, "inpaint": False, "in": input_dir, "out": r0_dir},
        {"name": "Relaxed", "thresh": 0.40, "inpaint": False, "in": r0_dir, "out": r1_dir},
        {"name": "Final", "thresh": 100.0, "inpaint": True, "in": r1_dir, "out": output_dir},
    ]

    strict_gaps: dict[str, float] = {}

    for r_idx, r in enumerate(rounds):
        in_dir = r["in"]
        out_dir = r["out"]
        ref_paths = [in_dir / f"{i}.tif" for i in ids]

        src_paths = []
        for i in ids:
            p = in_dir / f"{i}.tif"
            if not p.exists():
                p = input_dir / f"{i}.tif"  # fall back to raw if a round dropped one
            src_paths.append(p)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _fill_one,
                    img_path=src_paths[k],
                    ref_paths=ref_paths,
                    dates=dates,
                    this_date=dates[k],
                    method=method,
                    out_dir=out_dir,
                    threshold=r["thresh"],
                    enable_inpainting=r["inpaint"],
                ): k
                for k in range(n)
            }

            for future in tqdm(
                as_completed(futures),
                total=n,
                desc=f"Gap filling ({r['name']})",
                unit="image",
                disable=quiet,
            ):
                k = futures[future]
                try:
                    img_id, gaps_pct = future.result()
                except Exception:
                    logger.exception(f"Failed {r['name']} round for {ids[k]}")
                    img_id, gaps_pct = ids[k], 100.0

                if r_idx == 0:
                    strict_gaps[img_id] = gaps_pct

    shutil.rmtree(r0_dir, ignore_errors=True)
    shutil.rmtree(r1_dir, ignore_errors=True)

    if not quiet:
        logger.info(f"✓ Gap filled {n} images")

    results_df = pd.DataFrame(
        [{"id": i, "remaining_gaps_pct": strict_gaps.get(i, 100.0)} for i in ids]
    )

    metadata = metadata.drop(columns=["remaining_gaps_pct"], errors="ignore")
    metadata = metadata.merge(results_df, on="id", how="left")
    metadata["remaining_gaps_pct"] = metadata["remaining_gaps_pct"].fillna(100.0)

    return metadata