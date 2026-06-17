from __future__ import annotations

import itertools
import pathlib
from collections.abc import Sequence

import mlstac
import numpy as np
import pandas as pd
import rasterio as rio
import torch
import torch.nn.functional as F
from rasterio.transform import Affine
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from satcube.logging_config import setup_logger

logger = setup_logger(__name__)

# --- Constants ---------------------------------------------------------------

INPUT_CHUNK_SIZE = 128
INPUT_OVERLAP = 32

_MODEL_SPECS: dict[str, dict[str, object]] = {
    "SEN2SRLite": {
        "hf_url": "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json",
        "default_bands": [2, 3, 4, 5, 6, 7, 8, 9, 12, 13],
        "scale": 4,
    },
    "SEN2SR": {
        "hf_url": "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SR/main/mlm.json",
        "default_bands": [2, 3, 4, 5, 6, 7, 8, 9, 12, 13],
        "scale": 4,
    },
}

# --- 1. Coordinate Logic (Fix Last Chunk Strategy) ---------------------------


def fix_lastchunk(iterchunks, s2dim, chunk_size):
    """
    Adjusts the last chunks to fit within image boundaries by shifting back.
    This prevents the need for padding, avoiding edge artifacts.
    """
    itercontainer = []
    for index_i, index_j in iterchunks:
        # Check bounds for Row (Y)
        if index_i + chunk_size > s2dim[0]:
            index_i = max(s2dim[0] - chunk_size, 0)

        # Check bounds for Col (X)
        if index_j + chunk_size > s2dim[1]:
            index_j = max(s2dim[1] - chunk_size, 0)

        itercontainer.append((index_i, index_j))

    return sorted(set(itercontainer))


def define_iteration(dimension: tuple, chunk_size: int, overlap: int):
    """
    Define the iteration strategy to walk through the image with an overlap.
    """
    dimy, dimx = dimension  # height, width

    if chunk_size > max(dimx, dimy):
        return [(0, 0)]

    step = chunk_size - overlap

    # Generate initial chunk positions
    iterchunks = list(itertools.product(range(0, dimy, step), range(0, dimx, step)))

    # Fix chunks at the edges
    return fix_lastchunk(iterchunks, dimension, chunk_size)


# --- 2. Dataset for Batching -------------------------------------------------


class SuperResDataset(Dataset):
    """
    PyTorch Dataset for batched super-resolution inference.

    Yields 128x128 patches with coordinates for stitching.
    Handles edge cases with reflection padding.
    """

    def __init__(self, image: np.ndarray, coords: list[tuple[int, int]]):
        self.image = image
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # Coordinates are (row, col) -> (y, x)
        r, c = self.coords[idx]

        # Slicing (Guaranteed to be valid thanks to fix_lastchunk logic)
        # Handle edge case where image < 128 pixels total
        if (
            self.image.shape[1] < INPUT_CHUNK_SIZE
            or self.image.shape[2] < INPUT_CHUNK_SIZE
        ):
            patch = self.image[:, r : r + INPUT_CHUNK_SIZE, c : c + INPUT_CHUNK_SIZE]
            patch_t = torch.from_numpy(patch).float()
            _, h, w = patch_t.shape
            pad_h = INPUT_CHUNK_SIZE - h
            pad_w = INPUT_CHUNK_SIZE - w
            patch_t = F.pad(patch_t, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            patch = self.image[:, r : r + INPUT_CHUNK_SIZE, c : c + INPUT_CHUNK_SIZE]
            patch_t = torch.from_numpy(patch).float()

        return patch_t, r, c


# --- 3. Optimized Inference Engine -------------------------------------------


def predict_optimized(
    image: np.ndarray,
    model: torch.nn.Module,
    scale: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> np.ndarray:
    """
    Run super-resolution inference with optimized batching and stitching.

    Uses smart overlap cropping to avoid edge artifacts:
        - Strips half of the overlap (16px) from each prediction edge
        - Keeps only the high-quality center part of each patch
        - Results in seamless stitching without visible seams

    Args:
        image: Input image (C, H, W) in [0, 1] range.
        model: Loaded super-resolution model.
        scale: Upscaling factor (typically 4).
        batch_size: Number of patches to process simultaneously.
        num_workers: DataLoader workers for parallel patch loading.
        device: Device for inference ('cpu' or 'cuda').

    Returns:
        Super-resolved image (C, H*scale, W*scale) in [0, 1] range.
    """

    bands, height, width = image.shape
    out_h, out_w = height * scale, width * scale

    # Pre-allocate output buffer
    output = torch.zeros((bands, out_h, out_w), dtype=torch.float32)

    # Generate valid coordinates
    coords = define_iteration((height, width), INPUT_CHUNK_SIZE, INPUT_OVERLAP)
    dataset = SuperResDataset(image, coords)

    workers = num_workers if device != "cpu" else 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device == "cuda"),
    )

    # --- STITCHING LOGIC CONSTANTS ---
    # We strip exactly half of the overlap from each side of the prediction.
    # This leaves the center part which is the highest quality.

    res_n = scale

    # The margin to discard (e.g., 32px overlap -> discard 16px from edges)
    margin_in = INPUT_OVERLAP // 2
    margin_out = margin_in * res_n

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, r_starts, c_starts = batch
            x = x.to(device, non_blocking=True)

            # Predict Batch
            result_batch = model(x)
            result_batch = result_batch.cpu()

            for i in range(len(result_batch)):
                patch_out = result_batch[i]
                r_in, c_in = r_starts[i].item(), c_starts[i].item()

                # --- CANVAS COORDINATES ---
                # Where does this patch technically belong?
                r_canvas = r_in * scale
                c_canvas = c_in * scale

                # --- CROPPING LOGIC ---
                # Determine cropping based on borders

                # Top
                if r_in == 0:
                    crop_top = 0
                    paste_y = r_canvas
                else:
                    crop_top = margin_out
                    paste_y = r_canvas + margin_out

                # Left
                if c_in == 0:
                    crop_left = 0
                    paste_x = c_canvas
                else:
                    crop_left = margin_out
                    paste_x = c_canvas + margin_out

                # Bottom
                # If we are at the bottom edge (shifted back or exact fit)
                if (r_in + INPUT_CHUNK_SIZE) >= height:
                    crop_bottom = 0
                else:
                    crop_bottom = margin_out

                # Right
                if (c_in + INPUT_CHUNK_SIZE) >= width:
                    crop_right = 0
                else:
                    crop_right = margin_out

                # Apply Crops
                # Note: patch_out size is (C, 512, 512)
                h_p, w_p = patch_out.shape[1], patch_out.shape[2]
                valid_patch = patch_out[
                    :, crop_top : h_p - crop_bottom, crop_left : w_p - crop_right
                ]

                # Write to Output
                h_v, w_v = valid_patch.shape[1], valid_patch.shape[2]

                # Safety Clip (in case output calculation is off by 1 px due to rounding)
                if (paste_y + h_v) > out_h:
                    h_v = out_h - paste_y
                if (paste_x + w_v) > out_w:
                    w_v = out_w - paste_x

                output[:, paste_y : paste_y + h_v, paste_x : paste_x + w_v] = (
                    valid_patch[:, :h_v, :w_v]
                )

    return output.numpy()


# --- 4. Utilities ------------------------------------------------------------


def _ensure_model(
    variant: str, model_dir: str | pathlib.Path, device: str
) -> torch.nn.Module:
    """Load or download super-resolution model."""
    spec = _MODEL_SPECS.get(variant)
    if spec is None:
        raise ValueError(f"Unknown variant '{variant}'")

    model_dir = pathlib.Path(model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    if not any(model_dir.iterdir()):
        logger.info(f"Downloading {variant} model...")
        mlstac.download(file=spec["hf_url"], output_dir=model_dir)

    logger.info(f"Loading {variant} on {device}...")
    model = mlstac.load(str(model_dir)).compiled_model(device=device)
    model = model.to(device)
    return model


def superresolve_single(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    model: torch.nn.Module,
    *,
    variant: str = "SEN2SRLite",
    device: str = "cpu",
    batch_size: int = 4,
    num_workers: int = 4,
    band_indices: Sequence[int] | None = None,
) -> pathlib.Path:
    """
    Super-resolve a single Sentinel-2 image from 10m to 2.5m resolution.

    Upscales selected bands by 4x using deep learning model with seamless
    stitching and proper NoData preservation.

    Args:
        input_path: Path to input 10m GeoTIFF.
        output_path: Path for output 2.5m GeoTIFF.
        model: Loaded super-resolution model.
        variant: Model variant name.
        device: Device for inference ('cpu' or 'cuda').
        batch_size: Number of patches to process simultaneously.
        num_workers: DataLoader workers for parallel patch loading.
        band_indices: Bands to upscale (1-indexed). If None, uses model defaults.

    Returns:
        Path to output file.
    """

    input_path = pathlib.Path(input_path).expanduser().resolve()
    output_path = pathlib.Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = _MODEL_SPECS[variant]
    scale_factor = spec["scale"]
    bands_to_read = band_indices if band_indices is not None else spec["default_bands"]

    with rio.open(input_path) as src:
        if max(bands_to_read) > src.count:
            raise ValueError(
                f"Requested band index {max(bands_to_read)} exceeds image count {src.count}"
            )

        data = src.read(bands_to_read)
        profile = src.profile.copy()
        transform_src = src.transform

        # --- NODATA HANDLING ---
        # Detect true NoData (0) in input
        nodata_mask = (data == 0).all(axis=0)

    # Normalize 0-10000 -> 0-1
    X = data.astype(np.float32) / 10000.0

    # Run Optimized Inference
    superX = predict_optimized(
        image=X,
        model=model,
        scale=scale_factor,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    # --- RESTORE NODATA ---
    # Upsample the mask to match 2.5m resolution
    if nodata_mask.any():
        # (H, W) -> (1, 1, H, W) for interpolation
        mask_t = torch.from_numpy(nodata_mask).float().unsqueeze(0).unsqueeze(0)
        mask_out = F.interpolate(mask_t, scale_factor=scale_factor, mode="nearest")
        mask_out = mask_out.squeeze().numpy() > 0.5

        # Set output to 0 where input was 0
        superX[:, mask_out] = 0.0

    # Post-process
    # Clip to avoid artifacts (0 to 15000 is safe range for reflectances)
    superX = np.clip(superX * 10000.0, 0, 65535).astype(np.uint16)

    # Update transform
    new_transform = Affine(
        transform_src.a / scale_factor,
        transform_src.b,
        transform_src.c,
        transform_src.d,
        transform_src.e / scale_factor,
        transform_src.f,
    )

    profile.update(
        count=superX.shape[0],
        height=superX.shape[1],
        width=superX.shape[2],
        dtype="uint16",
        transform=new_transform,
        nodata=0,
    )

    with rio.open(output_path, "w", **profile) as dst:
        dst.write(superX)

    return output_path


# --- 5. Main API -------------------------------------------------------------


def superresolve_fn(
    metadata: pd.DataFrame,
    input_dir: str | pathlib.Path | None = None,
    output_dir: str | pathlib.Path = "superresolved",
    *,
    variant: str = "SEN2SRLite",
    model_dir: str | pathlib.Path | None = None,
    device: str = "cpu",
    batch_size: int = 4,
    nworks: int = 4,
    cache: bool = False,
    band_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Super-resolve Sentinel-2 images from 10m to 2.5m resolution using deep learning.

    Upscales imagery by 4x using SEN2SR models with seamless patch stitching
    and proper NoData preservation. Processes images sequentially with batched
    inference per image.

    **Model Variants:**

    - **SEN2SRLite** (recommended): Faster, smaller model (~200MB)
      - Good quality for most applications
      - 2-3x faster than full model
      - Default bands: [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]

    - **SEN2SR**: Full model (~600MB)
      - Highest quality, more details
      - Slower inference
      - Same default bands

    **Hardware Recommendations:**

    - CPU: batch_size=4, nworks=4 (typical: ~30s per 256×256 image)
    - GPU (8GB): batch_size=16, nworks=8 (typical: ~5s per 256×256 image)
    - GPU (16GB+): batch_size=32, nworks=12

    Args:
        metadata: DataFrame with scene IDs (must contain 'id' column).
        input_dir: Directory containing input 10m .tif files.
        output_dir: Output directory for 2.5m images. Default "superresolved".
        variant: Model variant ("SEN2SRLite" or "SEN2SR"). Default "SEN2SRLite".
        model_dir: Custom model directory. If None, uses output_dir/models/{variant}.
        device: Device for inference ('cpu' or 'cuda'). Default "cpu".
        batch_size: Number of 128×128 patches to process simultaneously.
            Higher = faster but more memory. Default 4.
        nworks: DataLoader workers for patch loading. Higher = better CPU utilization
            during GPU inference. Default 4.
        cache: Skip already processed files. Default False.
        band_indices: Bands to upscale (1-indexed). If None, uses model defaults
            [2,3,4,5,6,7,8,9,12,13]. Example: [4,3,2] for RGB only.

    Returns:
        Updated metadata DataFrame with 'superresolved' column.

    Raises:
        ValueError: If input_dir is None or variant is unknown.
        ImportError: If sen2sr package is not installed.

    Notes:
        - Models auto-download from HuggingFace on first use (~200-600MB)
        - Output resolution: 2.5m (4× upscaling from 10m)
        - NoData (0) pixels are preserved and not upscaled
        - Transform is automatically updated for GIS compatibility
        - Processing is sequential per image (no parallel image processing)

    Examples:
        >>> # CPU inference (recommended for small datasets)
        >>> sr = meta.superresolve(device="cpu", batch_size=4)

        >>> # GPU inference (recommended for large datasets)
        >>> sr = meta.superresolve(
        ...     device="cuda",
        ...     batch_size=16,
        ...     nworks=8
        ... )

        >>> # Full model for highest quality
        >>> sr = meta.superresolve(
        ...     variant="SEN2SR",
        ...     device="cuda",
        ...     batch_size=8
        ... )

        >>> # Upscale RGB bands only
        >>> sr = meta.superresolve(
        ...     band_indices=[4, 3, 2],
        ...     device="cuda"
        ... )

        >>> # Cache mode (skip existing files)
        >>> sr = meta.superresolve(cache=True)
    """

    if input_dir is None:
        raise ValueError("input_dir required")

    input_dir = pathlib.Path(input_dir).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_dir is None:
        model_dir = output_dir / "models" / variant
    else:
        model_dir = pathlib.Path(model_dir).expanduser().resolve()

    df = metadata["id"].to_frame()
    df["in_path"] = df["id"].apply(lambda x: str(input_dir / f"{x}.tif"))
    df["out_path"] = df["id"].apply(lambda x: str(output_dir / f"{x}.tif"))

    if cache:
        existing = {p.stem for p in output_dir.glob("*.tif")}
        df = df[~df["id"].isin(existing)]
        if df.empty:
            logger.info("All images already super-resolved (cache hit)")
            return metadata

    model = _ensure_model(variant=variant, model_dir=model_dir, device=device)

    results: list[dict] = []
    failed = []

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"SR ({variant})", unit="img"
    ):
        try:
            superresolve_single(
                input_path=row["in_path"],
                output_path=row["out_path"],
                model=model,
                variant=variant,
                device=device,
                batch_size=batch_size,
                num_workers=nworks,
                band_indices=band_indices,
            )
            results.append({"id": row["id"], "superresolved": True})

        except Exception:
            logger.exception(f"Failed to super-resolve {row['id']}")
            failed.append(row["id"])

    if failed:
        logger.warning(f"{len(failed)}/{len(df)} super-resolutions failed")
    else:
        logger.info(f"✓ Super-resolved {len(results)} images to 2.5m")

    if results:
        log = pd.DataFrame(results)
        metadata = metadata.drop(columns=["superresolved"], errors="ignore")
        metadata = metadata.merge(log, on="id", how="left")

    return metadata
