from __future__ import annotations

import os
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlstac
import numpy as np
import pandas as pd
import rasterio as rio
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from satcube.config import (
    CLOUDSEN12_URL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_WORKERS,
)
from satcube.exceptions import CloudMaskingError, ModelLoadError
from satcube.logging_config import setup_logger
from satcube.utils import define_iteration

logger = setup_logger(__name__)


def get_spline_window(window_size: int, power: int = 2) -> np.ndarray:
    """
    Generate 2D Hann window for smooth blending at tile edges.

    Args:
        window_size: Size of the window (typically chunk_size).
        power: Exponent for sharper blending. Higher values = sharper transitions.

    Returns:
        2D array with smooth weights for blending overlapping patches.
    """
    intersection = np.hanning(window_size)
    window_2d = np.outer(intersection, intersection)
    return (window_2d**power).astype(np.float32)


def infer_cloudmask_fast(
    input_path: str | pathlib.Path,
    cloud_model: torch.nn.Module,
    device: str = DEFAULT_DEVICE,
) -> tuple[dict, np.ndarray, dict, np.ndarray]:
    """
    Fast cloud mask inference for small images (≤512x512).

    Optimized for small images by avoiding batching overhead. Uses simple
    window-based processing with adaptive chunk sizing.

    Args:
        input_path: Path to input GeoTIFF file.
        cloud_model: Loaded CloudSEN12 model.
        device: Device for inference ('cpu' or 'cuda').

    Returns:
        Tuple containing:
            - percentages: Dict with cloud coverage statistics.
            - masked: Masked image array (clouds set to 0).
            - profile: Rasterio profile for saving.
            - full_mask: Full cloud mask array.
    """
    input_path = pathlib.Path(input_path).expanduser().resolve()

    with rio.open(input_path) as src:
        profile = src.profile
        height, width = profile["height"], profile["width"]

        chunk_size = max(height, width)
        if chunk_size > 512:
            chunk_size = 512
        overlap = 32 if chunk_size == 512 else 0

        full_mask = np.zeros((height, width), dtype=np.float32)
        coords = define_iteration((height, width), chunk_size, overlap)

        for row_off, col_off in coords:
            window = Window(col_off, row_off, chunk_size, chunk_size)
            patch = src.read(window=window) / 1e4

            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)

            result = cloud_model(patch_tensor).cpu().numpy()

            if result.ndim == 3:
                result = result.squeeze(0)
            result = result.astype(np.uint8)

            if col_off == 0:
                offset_x = 0
            else:
                offset_x = col_off + overlap // 2
            if row_off == 0:
                offset_y = 0
            else:
                offset_y = row_off + overlap // 2
            if (offset_x + chunk_size) >= width:
                length_x = width - offset_x
                sub_x_start = 0
            else:
                length_x = chunk_size - (overlap // 2)
                sub_x_start = overlap // 2 if col_off != 0 else 0

            if (offset_y + chunk_size) >= height:
                length_y = height - offset_y
                sub_y_start = 0
            else:
                length_y = chunk_size - (overlap // 2)
                sub_y_start = overlap // 2 if row_off != 0 else 0

            full_mask[
                offset_y : offset_y + length_y, offset_x : offset_x + length_x
            ] = result[
                sub_y_start : sub_y_start + length_y,
                sub_x_start : sub_x_start + length_x,
            ]

        data = src.read()
        masked = data.copy()
        masked[:, full_mask.astype(bool)] = 0

    flat = full_mask.astype(np.uint8).ravel()
    counts = np.bincount(flat, minlength=4)
    total = flat.size
    percentages = {
        "id": input_path.stem,
        "clear_pct": counts[0] / total * 100.0,
        "thin_cloud_pct": counts[1] / total * 100.0,
        "cloud_shadow_pct": counts[2] / total * 100.0,
        "thick_cloud_pct": counts[3] / total * 100.0,
    }

    return percentages, masked, profile, full_mask


class CloudMaskDataset(Dataset):
    """
    PyTorch Dataset for batched cloud mask inference on large images.

    Extracts patches on-the-fly with padding for edge cases.

    Args:
        image: Normalized image array (C, H, W) in [0, 1] range.
        coords: List of (row, col) top-left coordinates for patches.
        chunk_size: Size of each patch (typically 512).
    """

    def __init__(self, image: np.ndarray, coords: list, chunk_size: int):
        self.image = image
        self.coords = coords
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row_off, col_off = self.coords[idx]

        patch = self.image[
            :, row_off : row_off + self.chunk_size, col_off : col_off + self.chunk_size
        ]

        c, h, w = patch.shape
        patch_tensor = torch.from_numpy(patch).float()

        pad_h = self.chunk_size - h
        pad_w = self.chunk_size - w
        if pad_h > 0 or pad_w > 0:
            patch_tensor = F.pad(
                patch_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

        return patch_tensor, row_off, col_off, h, w


def infer_cloudmask_batched(
    input_path: str | pathlib.Path,
    cloud_model: torch.nn.Module,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_WORKERS,
    device: str = DEFAULT_DEVICE,
) -> tuple[dict, np.ndarray, dict, np.ndarray]:
    """
    Batched cloud mask inference for large images (>512x512).

    Uses GPU-optimized batching with Gaussian blending for smooth transitions
    at patch boundaries. Ideal for high-resolution satellite imagery.

    Args:
        input_path: Path to input GeoTIFF file.
        cloud_model: Loaded CloudSEN12 model.
        batch_size: Number of patches to process simultaneously.
        num_workers: DataLoader workers for parallel patch loading.
        device: Device for inference ('cpu' or 'cuda').

    Returns:
        Tuple containing:
            - percentages: Dict with cloud coverage statistics.
            - masked: Masked image array (clouds set to 0).
            - profile: Rasterio profile for saving.
            - full_mask: Full cloud mask array.
    """
    input_path = pathlib.Path(input_path).expanduser().resolve()

    with rio.open(input_path) as src:
        image = src.read()
        profile = src.profile
        height, width = profile["height"], profile["width"]

    image_norm = image.astype(np.float32) / 10000.0

    chunk_size = 512
    overlap = 32

    full_mask = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)

    window_spline = get_spline_window(chunk_size, power=2)
    window_tensor = torch.from_numpy(window_spline).to(device)

    coords = define_iteration((height, width), chunk_size, overlap)
    dataset = CloudMaskDataset(image_norm, coords, chunk_size)

    workers = num_workers if device != "cpu" else 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=2 if workers > 0 else None,
    )

    cloud_model.eval()
    with torch.no_grad():
        for batch in loader:
            patches, r_offs, c_offs, h_actuals, w_actuals = batch

            patches = patches.to(device, non_blocking=True)
            preds = cloud_model(patches)

            if preds.ndim == 2:
                preds = preds.unsqueeze(0)
            elif preds.ndim == 4 and preds.shape[1] == 1:
                preds = preds.squeeze(1)

            B = preds.size(0)
            batch_weights = window_tensor.unsqueeze(0).repeat(B, 1, 1)
            preds_weighted = preds.float() * batch_weights

            preds_cpu = preds_weighted.cpu().numpy()
            weights_cpu = batch_weights.cpu().numpy()

            for i in range(B):
                r = r_offs[i].item()
                c = c_offs[i].item()
                h = h_actuals[i].item()
                w = w_actuals[i].item()

                valid_pred = preds_cpu[i, :h, :w]
                valid_weight = weights_cpu[i, :h, :w]

                full_mask[r : r + h, c : c + w] += valid_pred
                count_map[r : r + h, c : c + w] += valid_weight

    mask_zero = count_map == 0
    count_map[mask_zero] = 1.0
    full_mask /= count_map
    full_mask[mask_zero] = 0.0

    full_mask = np.round(full_mask).astype(np.uint8)

    masked = image.copy()
    masked[:, full_mask > 0] = 0

    flat = full_mask.ravel()
    counts = np.bincount(flat, minlength=4)
    total = flat.size

    percentages = {
        "id": input_path.stem,
        "clear_pct": counts[0] / total * 100.0,
        "thin_cloud_pct": counts[1] / total * 100.0,
        "cloud_shadow_pct": counts[2] / total * 100.0,
        "thick_cloud_pct": counts[3] / total * 100.0,
    }

    return percentages, masked, profile, full_mask


def infer_cloudmask(
    input_path: str | pathlib.Path,
    cloud_model: torch.nn.Module,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_WORKERS,
    device: str = DEFAULT_DEVICE,
) -> tuple[dict, np.ndarray, dict, np.ndarray]:
    """
    Smart dispatcher that selects optimal inference strategy based on image size.

    Automatically chooses between:
        - Fast path for small images (≤512x512): No batching overhead
        - Batched path for large images (>512x512): GPU-optimized with blending

    Args:
        input_path: Path to input GeoTIFF file.
        cloud_model: Loaded CloudSEN12 model.
        batch_size: Batch size for large images (GPU inference).
        num_workers: DataLoader workers for large images.
        device: Device for inference ('cpu' or 'cuda').

    Returns:
        Tuple containing:
            - percentages: Dict with cloud coverage statistics.
            - masked: Masked image array (clouds set to 0).
            - profile: Rasterio profile for saving.
            - full_mask: Full cloud mask array.
    """
    input_path = pathlib.Path(input_path).expanduser().resolve()

    with rio.open(input_path) as src:
        height, width = src.profile["height"], src.profile["width"]

    if max(height, width) <= 512:
        return infer_cloudmask_fast(input_path, cloud_model, device)
    else:
        return infer_cloudmask_batched(
            input_path,
            cloud_model,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )


def _load_model(model_path: pathlib.Path, device: str) -> torch.nn.Module:
    """
    Load CloudSEN12 model with error handling.

    Args:
        model_path: Path to model directory.
        device: Device to load model on ('cpu' or 'cuda').

    Returns:
        Loaded and eval-mode PyTorch model.

    Raises:
        ModelLoadError: If model loading fails.
    """
    try:
        model = mlstac.load(str(model_path))
        cloud_model = model.compiled_model(device=device)
        cloud_model = cloud_model.to(device)
        cloud_model.eval()
        return cloud_model
    except Exception as e:
        raise ModelLoadError(f"Failed to load cloud model: {e}") from e


def _get_optimal_nworks(device: str) -> int:
    """
    Auto-detect optimal number of parallel workers based on CPU cores.

    Empirical formula to avoid thread contention:
        - CPU: cores × 0.375 (e.g., 8 cores → 3 workers)
        - GPU: Fixed 2 workers (GPU does heavy lifting)

    Args:
        device: Device type ('cpu' or 'cuda').

    Returns:
        Optimal number of ThreadPoolExecutor workers.
    """
    cpu_count = os.cpu_count() or 4

    if device == "cpu":
        return max(2, int(cpu_count * 0.375))
    else:
        return 2


def cloud_fn(
    metadata: pd.DataFrame | None = None,
    input_dir: str | pathlib.Path | None = None,
    output_dir: str | pathlib.Path = "masked",
    model_path: str | pathlib.Path = "SEN2CloudEnsemble",
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int | None = None,
    save_mask: bool = True,
    cache: bool = False,
) -> pd.DataFrame | None:
    """
    Apply cloud masking to Sentinel-2 scenes with parallel processing.

    Automatically optimizes processing strategy based on image size:
        - Small images (≤512): Fast path with parallel image processing
        - Large images (>512): Batched GPU path with Gaussian blending

    Args:
        metadata: DataFrame with scene IDs. If None, scans input_dir.
        input_dir: Directory containing input .tif files.
        output_dir: Output directory for masked images. Default "masked".
        model_path: Path to CloudSEN12 model. Default "SEN2CloudEnsemble".
        device: Device for inference ('cpu' or 'cuda'). Default "cpu".
        batch_size: Batch size for large images. Default 4.
        num_workers: Number of images to process in parallel. If None, auto-detects
            optimal value (~cores × 0.375 for CPU, 2 for GPU).
            Examples: 8 cores → ~3 workers, 16 cores → ~6 workers.
        save_mask: Save cloud mask as separate file. Default True.
        cache: Skip already processed files. Default False.

    Returns:
        Updated metadata DataFrame with cloud statistics (clear_pct, etc.)
        and renamed IDs (original_id_XXX where XXX is clear percentage).

    Raises:
        CloudMaskingError: If required parameters missing or invalid.
        ModelLoadError: If model loading fails.

    Examples:
        >>> # Auto-detect (recommended) - 8 cores → ~3 workers
        >>> cloudmeta = cloud_fn(metadata=df, input_dir="raw")

        >>> # Specify available cores
        >>> cloudmeta = cloud_fn(
        ...     metadata=df,
        ...     input_dir="raw",
        ...     num_workers=8  # Auto-scales to ~3 workers
        ... )

        >>> # GPU inference
        >>> cloudmeta = cloud_fn(
        ...     metadata=df,
        ...     input_dir="raw",
        ...     device="cuda",
        ...     num_workers=16,  # Auto-scales to 2 workers for GPU
        ...     batch_size=16
        ... )
    """
    if input_dir is None:
        raise CloudMaskingError("input_dir must be specified")

    input_dir = pathlib.Path(input_dir).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        if input_dir.is_dir():
            tif_paths = list(input_dir.rglob("*.tif"))
            df = pd.DataFrame(
                {"id": [p.stem for p in tif_paths], "path": [str(p) for p in tif_paths]}
            )
        elif input_dir.is_file() and input_dir.suffix.lower() == ".tif":
            df = pd.DataFrame({"id": [input_dir.stem], "path": [str(input_dir)]})
        else:
            raise CloudMaskingError(
                f"Input must be .tif or directory, got: {input_dir}"
            )
    else:
        df = metadata["id"].to_frame()
        df["path"] = df["id"].apply(lambda x: str(input_dir / f"{x}.tif"))

    if cache:
        exist_files = {f.stem for f in output_dir.glob("*.tif")}
        df = df[~df["id"].isin(exist_files)]
        if df.empty:
            logger.info("All images already masked (cache hit)")
            return metadata

    model_path = pathlib.Path(model_path).expanduser().resolve()
    if not model_path.exists():
        logger.info(f"Downloading cloud model to {model_path}")
        mlstac.download(file=CLOUDSEN12_URL, output_dir=model_path)

    # Auto-detect or apply internal scaling
    if num_workers is None:
        num_workers = _get_optimal_nworks(device)
    else:
        # User specified cores -> apply internal scaling factor
        if device == "cpu":
            num_workers = max(2, int(num_workers * 0.375))
        else:
            num_workers = 2  # GPU always uses 2 workers

    logger.info(f"Loading cloud model on {device} (workers={num_workers})")
    cloud_model = _load_model(model_path, device)

    # DataLoader workers (internal, not exposed to user)
    dataloader_workers = 4 if device == "cuda" else 0

    results_cloud = []
    failed = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                infer_cloudmask,
                input_path=row["path"],
                cloud_model=cloud_model,
                batch_size=batch_size,
                num_workers=dataloader_workers,
                device=device,
            ): row
            for _, row in df.iterrows()
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Cloud-masking",
            unit="image",
        ):
            row = futures[future]
            try:
                percentages, masked, profile, full_mask = future.result()

                clear_pct_int = int(percentages["clear_pct"])
                new_stem = f"{row['id']}_{clear_pct_int:03d}"
                output_file_path = output_dir / f"{new_stem}.tif"

                percentages["new_id"] = new_stem

                img_prof = profile.copy()
                img_prof.update(dtype="uint16", nodata=0)

                with rio.open(output_file_path, "w", **img_prof) as dst:
                    dst.write(masked)

                if save_mask:
                    out_meta = profile.copy()
                    out_meta.update(count=1, dtype="uint8", nodata=255)
                    output_mask = output_dir / f"{new_stem}_mask.tif"
                    with rio.open(output_mask, "w", **out_meta) as dst:
                        dst.write(full_mask, 1)

                results_cloud.append(percentages)

            except Exception:
                logger.exception(f"Failed to mask {row['id']}")
                failed.append(row["id"])

    if failed:
        logger.warning(f"{len(failed)}/{len(df)} masking failed")
    else:
        logger.info(f"✓ Masked {len(results_cloud)} images")

    cloud_df = pd.DataFrame(results_cloud)

    if cloud_df.empty:
        return metadata

    if metadata is not None:
        metadata = metadata.drop(
            columns=[
                "clear_pct",
                "thin_cloud_pct",
                "cloud_shadow_pct",
                "thick_cloud_pct",
            ],
            errors="ignore",
        )
        metadata = metadata.merge(cloud_df, on="id", how="left")

        if "new_id" in metadata.columns:
            metadata["id"] = metadata["new_id"].fillna(metadata["id"])
            metadata = metadata.drop(columns=["new_id"])
    else:
        metadata = cloud_df
        if "new_id" in metadata.columns:
            metadata["id"] = metadata["new_id"]
            metadata = metadata.drop(columns=["new_id"])

    return metadata
