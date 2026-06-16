from __future__ import annotations

import pathlib
import shutil
from collections import OrderedDict
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

# Max reference images kept in memory per target (LRU). Bounds RAM: with 13-band
# 1024x1024 float32 refs (~55 MB each) this caps per-worker cache near 440 MB.
_REF_CACHE_MAX = 8


def _load_ref(path, cache, max_cache=_REF_CACHE_MAX):
    """Read a reference once and keep it in a bounded LRU cache, in float32."""
    if path in cache:
        cache.move_to_end(path)
        return cache[path]
    with rio.open(path) as src:
        ref = src.read().astype(np.float32) / 1e4
    ref[(ref <= 0) | (ref > 1.0)] = np.nan
    valid = ~np.isnan(ref).any(axis=0)
    cache[path] = (ref, valid)
    cache.move_to_end(path)
    while len(cache) > max_cache:
        cache.popitem(last=False)
    return cache[path]


def _inpaint_bands(data, bad_2d, radius=3):
    """Smoothly fill leftover bad pixels per band with cv2 Telea inpainting.

    Runs on uint8 internally since cv2.inpaint only supports 8-bit. Fine here
    because this is the last-resort fill on the few pixels temporal matching
    could not cover, and the monthly median composite smooths it afterwards.
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


def _fill_one(img_path, ref_paths, dates, this_date, *, method, out_dir, threshold, enable_inpainting):
    """Fill one image. Each hole is matched to its best temporal neighbor by
    color error, transferred via histogram matching, and feather-blended.
    References are read once into a bounded cache, and candidates store only the
    hole pixels (not full arrays) to keep memory flat with many holes.
    """
    with rio.open(img_path) as src:
        data = src.read().astype(np.float32) / 1e4
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
    ref_cache = OrderedDict()
    n_bands = data.shape[0]

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

            n_fill = int(fillable_mask.sum())
            # keep only the hole pixels, not a full-size array
            filled_values = np.zeros((n_bands, n_fill), dtype=np.float32)
            patch_error = 0.0
            rgb_bands = [0, 1, 2] if n_bands > 2 else [0]
            band_valid = True

            for b in range(n_bands):
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
                        A = np.vstack([valid_ref_clean, np.ones(len(valid_ref_clean))]).T
                        m, c = np.linalg.lstsq(A, valid_tgt_clean, rcond=None)[0]
                        matched = pixels_fill * m + c
                        val_pixels = valid_ref_clean * m + c
                    except Exception:
                        matched = pixels_fill
                        val_pixels = valid_ref_clean

                matched = np.clip(matched, 0.0, 1.0).astype(np.float32)
                filled_values[b] = matched
                if b in rgb_bands:
                    diff = np.abs(valid_tgt_clean - val_pixels)
                    patch_error += np.mean(diff)

            if not band_valid:
                continue
            current_metric = patch_error / len(rgb_bands)
            candidates.append({"filled_values": filled_values, "fillable_mask": fillable_mask,
                               "error": current_metric, "coverage": coverage})

        if len(candidates) == 0:
            continue
        candidates.sort(key=lambda x: x["error"])
        best = candidates[0]
        if best["error"] > threshold:
            continue

        fmask = best["fillable_mask"]
        dist_map = distance_transform_edt(fmask)
        alpha = np.clip(dist_map / 5.0, 0, 1)
        a_fill = alpha[fmask]
        for b in range(n_bands):
            p = best["filled_values"][b]
            c = final_data[b][fmask]
            a = a_fill.copy()
            c_nan = np.isnan(c)
            if c_nan.any():
                c[c_nan] = p[c_nan]
                a[c_nan] = 1.0
            blended = np.clip((p * a) + (c * (1.0 - a)), 0.0, 1.0)
            final_data[b][fmask] = blended

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


def _save_image(data, path, profile):
    """Save a float [0-1] array back to a uint16 GeoTIFF."""
    data = np.nan_to_num(data, nan=0.0)
    data = np.clip(data, 0.0, 1.0)
    data_uint = (data * 1e4).astype(np.uint16)
    with rio.open(path, "w", **profile) as dst:
        dst.write(data_uint)


def _get_optimal_gapfill_workers():
    """Auto-detect a reasonable worker count, capped at 8."""
    import os
    return min(os.cpu_count() or 4, 8)


def gapfill_fn(metadata, input_dir, output_dir="gapfilled", *, method="histogram_matching", num_workers=None, quiet=False):
    """Fill cloud/shadow gaps with multi-round temporal matching.

    Three rounds run with a barrier between them, so each round reads the
    previous round's outputs and writes its own, never a file another worker is
    writing (deterministic, race-free):
        1. Strict  (0.15): high-quality matches only.
        2. Relaxed (0.40): more tolerant matches for what is left.
        3. Final   (inf):  cv2 Telea inpainting for the rest.

    Returns the metadata with a 'remaining_gaps_pct' column, the percent of
    pixels the strict round could not fill via temporal matching.

    Memory note: references use float32 and a bounded LRU cache, and candidates
    store only hole pixels, so per-worker RAM stays near 0.5 GB even on
    1024x1024 13-band scenes. On Colab num_workers=4 is a safe default.
    """
    input_dir = pathlib.Path(input_dir).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = list(metadata["id"])
    dates = pd.to_datetime(metadata["date"]).to_numpy()
    n = len(ids)
    if num_workers is None:
        num_workers = _get_optimal_gapfill_workers()

    r0_dir = output_dir / "_round0"
    r1_dir = output_dir / "_round1"
    r0_dir.mkdir(parents=True, exist_ok=True)
    r1_dir.mkdir(parents=True, exist_ok=True)

    rounds = [
        {"name": "Strict", "thresh": 0.15, "inpaint": False, "in": input_dir, "out": r0_dir},
        {"name": "Relaxed", "thresh": 0.40, "inpaint": False, "in": r0_dir, "out": r1_dir},
        {"name": "Final", "thresh": 100.0, "inpaint": True, "in": r1_dir, "out": output_dir},
    ]
    strict_gaps = {}

    for r_idx, r in enumerate(rounds):
        in_dir = r["in"]
        out_dir = r["out"]
        ref_paths = [in_dir / f"{i}.tif" for i in ids]
        src_paths = []
        for i in ids:
            p = in_dir / f"{i}.tif"
            if not p.exists():
                p = input_dir / f"{i}.tif"
            src_paths.append(p)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_fill_one, img_path=src_paths[k], ref_paths=ref_paths, dates=dates,
                                this_date=dates[k], method=method, out_dir=out_dir,
                                threshold=r["thresh"], enable_inpainting=r["inpaint"]): k
                for k in range(n)
            }
            for future in tqdm(as_completed(futures), total=n, desc=f"Gap filling ({r['name']})", unit="image", disable=quiet):
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
    results_df = pd.DataFrame([{"id": i, "remaining_gaps_pct": strict_gaps.get(i, 100.0)} for i in ids])
    metadata = metadata.drop(columns=["remaining_gaps_pct"], errors="ignore")
    metadata = metadata.merge(results_df, on="id", how="left")
    metadata["remaining_gaps_pct"] = metadata["remaining_gaps_pct"].fillna(100.0)
    return metadata