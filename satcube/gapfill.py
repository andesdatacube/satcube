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

# Max reference images kept per target (LRU, float32 ~55 MB each at 1024x1024x13).
_REF_CACHE_MAX = 16


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
    """Fill leftover bad pixels per band with cv2 Telea (smooth, no blocky patches)."""
    mask = bad_2d.astype(np.uint8)
    out = data.copy()
    for b in range(data.shape[0]):
        band = np.clip(np.nan_to_num(data[b], nan=0.0), 0.0, 1.0)
        band_u8 = (band * 255.0).astype(np.uint8)
        filled = cv2.inpaint(band_u8, mask, radius, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
        out[b][bad_2d] = filled[bad_2d]
    return out


def _match_band(tgt_band, ref_band, training_mask, fillable_mask, method):
    """Color-transfer one band. Returns (fill values at fillable_mask, match error)."""
    vt = tgt_band[training_mask]
    vr = ref_band[training_mask]
    pixels_fill = ref_band[fillable_mask]
    if len(vt) < 10 or len(vr) < 10:
        return None, None
    if method == "histogram_matching":
        hist_t, bins = np.histogram(vt, 128, [0, 1.0])
        hist_r, _ = np.histogram(vr, 128, [0, 1.0])
        cdf_t = hist_t.cumsum() / (hist_t.sum() + 1e-10)
        cdf_r = hist_r.cumsum() / (hist_r.sum() + 1e-10)
        lut = np.interp(cdf_r, cdf_t, bins[:-1])
        matched = np.interp(pixels_fill, bins[:-1], lut)
        val_pixels = np.interp(vr, bins[:-1], lut)
    else:
        try:
            A = np.vstack([vr, np.ones(len(vr))]).T
            m, c = np.linalg.lstsq(A, vt, rcond=None)[0]
            matched = pixels_fill * m + c
            val_pixels = vr * m + c
        except Exception:
            matched, val_pixels = pixels_fill, vr
    matched = np.clip(matched, 0.0, 1.0).astype(np.float32)
    err = float(np.mean(np.abs(vt - val_pixels)))
    return matched, err


def _fill_one(img_path, ref_paths, dates, this_date, *, method, out_dir, threshold, enable_inpainting):
    """Fill one image. Each hole picks its best temporal neighbor by RGB color
    error, transfers color via histogram matching, and feather-blends the seam.
    Per-hole masks are precomputed once and references are cached, so it stays
    fast and flat in memory even with many holes.
    """
    with rio.open(img_path) as src:
        data = src.read().astype(np.float32) / 1e4
        prof = src.profile
        data[(data <= 0) | (data > 1.0)] = np.nan
        base_missing_mask = np.isnan(data).any(axis=0)

    final_data = data.copy()
    if base_missing_mask.sum() == 0:
        _save_image(final_data, out_dir / img_path.name, prof)
        return img_path.stem, 0.0

    labeled_array, num_features = label(base_missing_mask)
    idxs = np.argsort(np.abs(dates - this_date))
    ref_cache = OrderedDict()
    cache_cap = min(len(ref_paths), _REF_CACHE_MAX)
    n_bands = data.shape[0]
    max_candidates = 10
    rgb_bands = [0, 1, 2] if n_bands > 2 else [0]

    for region_idx in range(1, num_features + 1):
        current_region_mask = labeled_array == region_idx
        target_hole = binary_dilation(current_region_mask, iterations=10)
        still_missing = np.isnan(final_data).any(axis=0)
        target_hole = target_hole & still_missing
        ht = target_hole.sum()
        if ht == 0:
            continue

        # precomputed once per hole (these used to run per candidate, the slow part)
        hole_expanded = binary_dilation(target_hole, iterations=5)
        context = binary_dilation(hole_expanded, iterations=15)
        valid_in_target = ~np.isnan(final_data).any(axis=0)

        candidates = []
        for i in idxs:
            if len(candidates) >= max_candidates:
                break
            ref_path = ref_paths[i]
            if ref_path.name == img_path.name:
                continue
            try:
                ref, ref_valid_mask = _load_ref(ref_path, ref_cache, cache_cap)
            except Exception:
                continue
            fillable_mask = hole_expanded & ref_valid_mask
            coverage = (fillable_mask & target_hole).sum() / (ht + 1e-6)
            if coverage < 0.85:
                continue
            training_mask = valid_in_target & ref_valid_mask & context
            if training_mask.sum() < 20:
                continue
            # score with rgb bands only; full 13-band fill happens only for the winner
            errs, ok = [], True
            for b in rgb_bands:
                _, e = _match_band(final_data[b], ref[b], training_mask, fillable_mask, method)
                if e is None:
                    ok = False
                    break
                errs.append(e)
            if not ok:
                continue
            candidates.append({"i": i, "error": sum(errs) / len(errs)})

        if not candidates:
            continue
        candidates.sort(key=lambda x: x["error"])
        best = candidates[0]
        if best["error"] > threshold:
            continue

        ref, ref_valid_mask = _load_ref(ref_paths[best["i"]], ref_cache, cache_cap)
        fmask = hole_expanded & ref_valid_mask
        tmask = valid_in_target & ref_valid_mask & context
        n_fill = int(fmask.sum())
        filled_values = np.zeros((n_bands, n_fill), dtype=np.float32)
        for b in range(n_bands):
            matched, _ = _match_band(final_data[b], ref[b], tmask, fmask, method)
            if matched is None:
                matched = np.clip(ref[b][fmask], 0.0, 1.0).astype(np.float32)
            filled_values[b] = matched

        dist_map = distance_transform_edt(fmask)
        a_fill = np.clip(dist_map / 5.0, 0, 1)[fmask]
        for b in range(n_bands):
            p = filled_values[b]
            c = final_data[b][fmask]
            a = a_fill.copy()
            c_nan = np.isnan(c)
            if c_nan.any():
                c[c_nan] = p[c_nan]
                a[c_nan] = 1.0
            final_data[b][fmask] = np.clip((p * a) + (c * (1.0 - a)), 0.0, 1.0)

    if enable_inpainting:
        bad = np.isnan(final_data) | (final_data <= 0.00005) | (final_data > 1.0)
        bad_2d = bad.any(axis=0)
        if bad_2d.sum() > 0:
            final_data = _inpaint_bands(final_data, bad_2d, radius=3)

    _save_image(final_data, out_dir / img_path.name, prof)
    final_gaps = (np.isnan(final_data) | (final_data <= 0.00005)).any(axis=0)
    h, w = final_data.shape[1], final_data.shape[2]
    return img_path.stem, (final_gaps.sum() / (h * w)) * 100.0


def _save_image(data, path, profile):
    """Save a float [0-1] array back to a uint16 GeoTIFF."""
    data = np.clip(np.nan_to_num(data, nan=0.0), 0.0, 1.0)
    with rio.open(path, "w", **profile) as dst:
        dst.write((data * 1e4).astype(np.uint16))


def _get_optimal_gapfill_workers():
    import os
    return min(os.cpu_count() or 4, 8)


def gapfill_fn(metadata, input_dir, output_dir="gapfilled", *, method="histogram_matching", num_workers=None, quiet=False):
    """Fill cloud/shadow gaps with multi-round temporal matching.

    Three rounds run with a barrier between them, so each round reads the
    previous round's outputs and writes its own (deterministic, race-free):
    Strict (0.15), Relaxed (0.40), then cv2 Telea inpainting for the rest.
    Returns metadata with a 'remaining_gaps_pct' column (percent the strict
    round could not fill via temporal matching).

    On Colab num_workers=4 is a safe default for RAM.
    """
    input_dir = pathlib.Path(input_dir).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = list(metadata["id"])
    dates = pd.to_datetime(metadata["date"]).to_numpy()
    n = len(ids)
    if num_workers is None:
        num_workers = _get_optimal_gapfill_workers()

    r0_dir, r1_dir = output_dir / "_round0", output_dir / "_round1"
    r0_dir.mkdir(parents=True, exist_ok=True)
    r1_dir.mkdir(parents=True, exist_ok=True)
    rounds = [
        {"name": "Strict", "thresh": 0.15, "inpaint": False, "in": input_dir, "out": r0_dir},
        {"name": "Relaxed", "thresh": 0.40, "inpaint": False, "in": r0_dir, "out": r1_dir},
        {"name": "Final", "thresh": 100.0, "inpaint": True, "in": r1_dir, "out": output_dir},
    ]
    strict_gaps = {}

    for r_idx, r in enumerate(rounds):
        in_dir, out_dir = r["in"], r["out"]
        ref_paths = [in_dir / f"{i}.tif" for i in ids]
        src_paths = [(in_dir / f"{i}.tif") if (in_dir / f"{i}.tif").exists()
                     else (input_dir / f"{i}.tif") for i in ids]
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