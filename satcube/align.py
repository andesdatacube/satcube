from __future__ import annotations

import pathlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import rasterio as rio
import satalign
from tqdm import tqdm

from satcube.config import ALIGN_OVERLAP_RATIO, ALIGN_UPSAMPLE_FACTOR
from satcube.exceptions import AlignmentError
from satcube.logging_config import setup_logger

logger = setup_logger(__name__)


def _process_image(
    image: np.ndarray,
    reference: np.ndarray,
    profile: dict,
    output_path: pathlib.Path,
) -> tuple[float, float]:
    """
    Align image to reference and save the result.

    Args:
        image: 3D array (bands × height × width) in uint16 scale.
        reference: Reference scene in float32 [0-1] range.
        profile: Rasterio profile copied from image.
        output_path: Destination GeoTIFF path.

    Returns:
        Tuple of (dx_px, dy_px) sub-pixel translation in pixel units.
    """

    if np.all(image == 0):
        logger.warning(f"Skipping empty image: {output_path.name}")
        # Just copy the empty image without aligning
        with rio.open(output_path, "w", **profile) as dst:
            dst.write(image)
        return 0.0, 0.0

    image_float = image.astype(np.float32) / 10_000
    image_float = image_float[np.newaxis, ...]

    aligned_cube, M = satalign.PCC(
        datacube=image_float,
        reference=reference,
        num_threads=4,
        channel="gradients",
        disambiguate=True,
        overlap_ratio=ALIGN_OVERLAP_RATIO,
        upsample_factor=ALIGN_UPSAMPLE_FACTOR,
        space="real",
        interpolation=cv2.INTER_NEAREST,
    ).run_multicore()

    aligned_image = (aligned_cube * 10_000).astype(np.uint16).squeeze()

    with rio.open(output_path, "w", **profile) as dst:
        dst.write(aligned_image)

    return M[0][0, 2], M[0][1, 2]


def _process_row(
    row: pd.Series,
    reference: np.ndarray,
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> tuple[str, float, float]:
    """
    Worker helper executed in a ThreadPoolExecutor.

    Args:
        row: Single row from metadata (must contain an id column).
        reference: Reference scene in float32 [0-1] range.
        input_dir: Directory holding the raw scenes.
        output_dir: Directory where aligned scenes are written.

    Returns:
        Tuple of (id, dx_px, dy_px) suitable for merging back into metadata.
    """
    row_path = input_dir / f"{row['id']}.tif"
    output_path = output_dir / f"{row['id']}.tif"

    with rio.open(row_path) as src:
        image = src.read()
        profile = src.profile
    with warnings.catch_warnings():

        warnings.filterwarnings("ignore", message=r".*translation.*too large.*")
        warnings.filterwarnings("ignore", message=r".*No matching points.*")

        dx_px, dy_px = _process_image(
            image=image,
            reference=reference,
            profile=profile,
            output_path=output_path,
        )

    return row["id"], dx_px, dy_px


def align_fn(
    metadata: pd.DataFrame | None = None,
    input_dir: str | pathlib.Path = "raw",
    output_dir: str | pathlib.Path = "aligned",
    num_workers: int = 8,
    cache: bool = False,
) -> pd.DataFrame | None:
    """
    Align a collection of scenes to a common reference.

    Uses Local Gradient Matching (LGM) via satalign to co-register all scenes
    to the clearest reference image. The reference is automatically selected
    based on the highest clear_pct or cs_cdf value.

    Args:
        metadata: Table from download step. Must contain id and either cs_cdf or clear_pct columns.
        input_dir: Directory containing the raw .tif scenes. Default "raw".
        output_dir: Directory where aligned images will be stored. Default "aligned".
        num_workers: Number of worker threads. Default 4.
        cache: Skip alignment for scenes that already exist in output_dir. Default False.

    Returns:
        Metadata augmented with dx_px and dy_px (sub-pixel shifts).
        If cache caused every scene to be skipped, returns original metadata unchanged.

    Raises:
        AlignmentError: If metadata is None or does not contain required columns.

    Examples:
        >>> from satcube import align_fn
        >>> aligned_meta = align_fn(
        ...     metadata=meta,
        ...     input_dir="raw",
        ...     output_dir="aligned",
        ...     num_workers=8,
        ...     cache=True
        ... )
        >>> print(aligned_meta[['id', 'dx_px', 'dy_px']].head())
    """
    input_dir = pathlib.Path(input_dir).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        raise AlignmentError("metadata is required. Run the download step first.")

    # Select reference image (clearest scene)
    if "clear_pct" in metadata.columns:
        id_reference = metadata.sort_values("clear_pct", ascending=False).iloc[0]["id"]
    elif "cs_cdf" in metadata.columns:
        id_reference = metadata.sort_values("cs_cdf", ascending=False).iloc[0]["id"]
    else:
        raise AlignmentError("Metadata must contain 'clear_pct' or 'cs_cdf' columns")

    df = metadata.copy()

    if cache:
        existing = {p.stem for p in output_dir.glob("*.tif")}
        df = df[~df["id"].isin(existing)]
        if df.empty:
            logger.info("All images already aligned (cache hit)")
            return metadata

    logger.info(f"Using reference image: {id_reference}")

    reference_path = input_dir / f"{id_reference}.tif"
    with rio.open(reference_path) as ref_src:
        reference = ref_src.read()

    reference_float = reference.astype(np.float32) / 10_000

    results: list[dict[str, float]] = []
    failed = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_row,
                row=row,
                reference=reference_float,
                input_dir=input_dir,
                output_dir=output_dir,
            ): row["id"]
            for _, row in df.iterrows()
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Aligning",
            unit="image",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.5,
        ):
            img_id = futures[future]
            try:
                img_id, dx_px, dy_px = future.result()
                results.append({"id": img_id, "dx_px": dx_px, "dy_px": dy_px})
            except Exception:
                logger.exception(f"Failed to align {img_id}")
                failed.append(img_id)

    if failed:
        logger.warning(f"{len(failed)}/{len(df)} alignments failed")
    else:
        logger.info(f"✓ Aligned {len(results)} images")

    shift_df = pd.DataFrame(results)

    metadata = metadata.drop(columns=["dx_px", "dy_px"], errors="ignore")
    metadata = metadata.merge(shift_df, on="id", how="left")

    return metadata
