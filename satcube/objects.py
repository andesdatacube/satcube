from __future__ import annotations

import pathlib
from dataclasses import dataclass

import cubexpress
import pandas as pd

from satcube.align import align_fn
from satcube.cloud import cloud_fn
from satcube.composite import monthly_composites_s2
from satcube.gapfill import gapfill_fn
from satcube.interpolate import interpolate_fn
from satcube.smooth import smooth_fn
from satcube.super import superresolve_fn


@dataclass
class SatCubeMetadata:
    df: pd.DataFrame
    raw_dir: pathlib.Path = pathlib.Path("raw")
    _current_dir: pathlib.Path | None = None
    _table: object = None

    def _dir(self) -> pathlib.Path:
        return self._current_dir or self.raw_dir

    def _spawn(self, *, df: pd.DataFrame, current_dir: pathlib.Path) -> SatCubeMetadata:
        """Create new SatCubeMetadata instance and save metadata to Parquet."""
        new_instance = SatCubeMetadata(
            df=df, raw_dir=self.raw_dir, _current_dir=current_dir
        )
        new_instance._save_metadata()
        return new_instance

    def _save_metadata(self) -> None:
        """Save current metadata DataFrame to Parquet in current directory."""
        current = self._dir()
        current.mkdir(parents=True, exist_ok=True)
        metadata_path = current / "metadata.parquet"
        self.df.to_parquet(metadata_path, index=False)

    @classmethod
    def from_directory(
        cls, directory: str | pathlib.Path, raw_dir: str | pathlib.Path | None = None
    ) -> SatCubeMetadata:
        """
        Load SatCubeMetadata from a directory containing metadata.parquet.

        This allows resuming the pipeline from any stage without re-running
        previous steps.

        Args:
            directory: Directory containing metadata.parquet and processed images.
            raw_dir: Optional path to raw directory. If None, uses directory.

        Returns:
            SatCubeMetadata instance ready to continue processing.

        Raises:
            FileNotFoundError: If metadata.parquet doesn't exist in directory.

        Examples:
            >>> # Resume from cloud masking stage
            >>> meta = SatCubeMetadata.from_directory("masked")
            >>> filled = meta.gapfill()

            >>> # Resume from aligned stage
            >>> meta = SatCubeMetadata.from_directory("aligned", raw_dir="raw")
            >>> cloudmeta = meta.cloud_masking()
        """
        directory = pathlib.Path(directory).expanduser().resolve()
        metadata_path = directory / "metadata.parquet"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No metadata.parquet found in {directory}. "
                f"This directory may not contain processed satcube data."
            )

        df = pd.read_parquet(metadata_path)

        if raw_dir is None:
            raw_dir = directory
        else:
            raw_dir = pathlib.Path(raw_dir).expanduser().resolve()

        return cls(df=df, raw_dir=raw_dir, _current_dir=directory)


    def select_bands(self, *bands: str) -> SatCubeMetadata:
        """Keep only the given bands (in order) before downloading.

        Delegates to cubexpress's RequestTable.select_bands. The output band
        order matches the order you pass here. Immutable: returns a new
        SatCubeMetadata; the original is unchanged.

        Args:
            *bands: Band names to keep, in the desired output order
                (e.g. "B4", "B3", "B2" for RGB).

        Returns:
            New SatCubeMetadata carrying only the selected bands.

        Raises:
            ValueError: if there is no RequestTable (metadata came from disk),
                or a requested band is not available.

        Examples:
            >>> meta_rgb = meta.select_bands("B4", "B3", "B2")   # RGB only
            >>> cube = meta_rgb.download(output_dir="rgb")
        """
        if self._table is None:
            raise ValueError(
                "no RequestTable to select bands from; build it via satcube.metadata()."
            )

        new_table = self._table.select_bands(*bands)
        instance = SatCubeMetadata(
            df=self.df.copy().reset_index(drop=True),
            raw_dir=self.raw_dir,
            _current_dir=self._current_dir,
        )
        instance._table = new_table
        return instance


    def download(
        self,
        output_dir: str = "raw",
        num_workers: int = 8,
        cache: bool = False,
    ) -> SatCubeMetadata:
        """Download the discovered scenes to disk as GeoTIFFs.

        With cache=True, skip the download when every scene already exists
        on disk, so re-running a cell does not hit Earth Engine again.
        """
        output_dir = pathlib.Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._table is None:
            raise ValueError(
                "no RequestTable to download; build it via satcube.metadata()."
            )

        new_df = self.df.copy().reset_index(drop=True)

        expected = {f"{row['id']}.tif" for _, row in new_df.iterrows()}
        existing = {p.name for p in output_dir.glob("*.tif")}

        if cache and expected and expected.issubset(existing):
            instance = SatCubeMetadata(
                df=new_df, raw_dir=output_dir, _current_dir=output_dir
            )
            instance._save_metadata()
            return instance

        cubexpress.express(self._table, output_dir, nworkers=num_workers)

        instance = SatCubeMetadata(
            df=new_df, raw_dir=output_dir, _current_dir=output_dir
        )
        instance._save_metadata()
        return instance


    def align(
        self,
        input_dir: str | pathlib.Path | None = None,
        output_dir: str | pathlib.Path = "aligned",
        num_workers: int = 8,
        cache: bool = False,
    ) -> SatCubeMetadata:
        self.aligned_dir = pathlib.Path(output_dir).resolve()

        input_dir = (
            pathlib.Path(input_dir).expanduser().resolve() if input_dir else self._dir()
        )
        output_dir = pathlib.Path(output_dir).resolve()

        new_df = align_fn(
            metadata=self.df,
            input_dir=input_dir,
            output_dir=self.aligned_dir,
            num_workers=num_workers,
            cache=cache,
        )

        return self._spawn(df=new_df, current_dir=output_dir)

    def cloud_masking(
        self,
        input_dir: str | pathlib.Path | None = None,
        output_dir: str | pathlib.Path = "masked",
        device: str = "cpu",
        num_workers: int | None = None,
        batch_size: int = 1,
        save_mask: bool = False,
        cache: bool = False,
    ) -> SatCubeMetadata:
        """
        Apply cloud masking to Sentinel-2 scenes with automatic optimization.

        Uses CloudSEN12 ensemble model with smart strategy selection:
            - Small images (≤512x512): Fast parallel processing
            - Large images (>512x512): GPU-optimized batching with Gaussian blending

        Args:
            input_dir: Directory with input images. Default uses current pipeline dir.
            output_dir: Output directory for masked images. Default "masked".
            device: Device for inference ('cpu' or 'cuda'). Default "cpu".
            num_workers: Number of images to process in parallel. If None, auto-detects
                (~cores × 0.375 for CPU, 2 for GPU).
                Examples: 8 cores → ~3 workers, 16 cores → ~6 workers.
            batch_size: Batch size for large images (GPU). Default 1.
            save_mask: Save cloud mask as separate file. Default False.
            cache: Skip already processed files. Default False.

        Returns:
            New SatCubeMetadata with updated cloud statistics and renamed IDs.

        Examples:
            >>> # Auto-detect (recommended)
            >>> cloudmeta = meta.cloud_masking()

            >>> # Specify cores (auto-scales: 8 → ~3 workers)
            >>> cloudmeta = meta.cloud_masking(num_workers=8)

            >>> # GPU inference
            >>> cloudmeta = meta.cloud_masking(
            ...     device="cuda",
            ...     num_workers=16,
            ...     batch_size=16
            ... )
        """
        input_dir = (
            pathlib.Path(input_dir).expanduser().resolve() if input_dir else self._dir()
        )
        output_dir = pathlib.Path(output_dir).resolve()

        new_df = cloud_fn(
            metadata=self.df,
            input_dir=input_dir,
            output_dir=output_dir,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            save_mask=save_mask,
            cache=cache,
        )

        return self._spawn(df=new_df, current_dir=output_dir)

    def gapfill(
        self,
        *,
        input_dir: str | pathlib.Path | None = None,
        output_dir: str | pathlib.Path = "gapfilled",
        method: str = "histogram_matching",
        num_workers: int | None = None,
        quiet: bool = False,
    ) -> SatCubeMetadata:
        """
        Fill cloud/shadow gaps using multi-round temporal matching.

        Implements a sophisticated 3-round cascading strategy for optimal quality:
            1. **Strict Round**: High-quality matches only (threshold=0.15)
            2. **Relaxed Round**: More tolerant matches (threshold=0.40)
            3. **Final Round**: Inpainting for any remaining pixels

        Each cloud gap is treated independently to prevent artifacts. Uses
        histogram matching for accurate color transfer and spatial blending
        for seamless transitions.

        Args:
            input_dir: Directory with input images. Default uses current pipeline dir.
            output_dir: Output directory for filled images. Default "gapfilled".
            method: Color transfer method. Options:
                - 'histogram_matching': Match full histogram (recommended)
                - 'linear': Simple linear regression
            num_workers: Number of images to process in parallel. If None, auto-detects
                optimal value (typically CPU cores, capped at 8). Default None.
            quiet: Suppress progress bars. Default False.

        Returns:
            New SatCubeMetadata with gap-filled images.

        Notes:
            The 3-round strategy is essential for quality:
                - Round 1 ensures best matches for easy gaps
                - Round 2 fills more difficult gaps with relaxed constraints
                - Round 3 uses inpainting only as last resort

        Examples:
            >>> # Auto-detect workers (recommended)
            >>> filled = meta.gapfill()

            >>> # Manual workers (for debugging or specific hardware)
            >>> filled = meta.gapfill(num_workers=4, method="histogram_matching")

            >>> # Quiet mode for automation
            >>> filled = meta.gapfill(quiet=True)
        """
        in_dir = (
            pathlib.Path(input_dir).expanduser().resolve() if input_dir else self._dir()
        )
        out_dir = pathlib.Path(output_dir).resolve()

        new_df = gapfill_fn(
            metadata=self.df,
            input_dir=in_dir,
            output_dir=out_dir,
            method=method,
            num_workers=num_workers,
            quiet=quiet,
        )

        return self._spawn(df=new_df, current_dir=out_dir)

    def composite(
        self,
        input_dir: str | pathlib.Path | None = None,
        output_dir: str | pathlib.Path = "monthly_composites",
        agg_method: str = "median",
        num_workers: int | None = None,
        quiet: bool = False,
    ) -> SatCubeMetadata:
        """
        Generate monthly composites from time series imagery.

        Aggregates all images within each month into a single composite image
        using parallel processing for optimal performance.

        Args:
            input_dir: Directory with input images. Default uses current pipeline dir.
            output_dir: Output directory for composites. Default "monthly_composites".
            agg_method: Aggregation method. Options:
                - 'median': Median value (recommended, robust to outliers)
                - 'mean': Mean value
                - 'max': Maximum value
                - 'min': Minimum value
            num_workers: Number of months to process in parallel. If None, auto-detects
                optimal value (typically CPU cores, capped at 8). Default None.
            quiet: Suppress progress bars. Default False.

        Returns:
            New SatCubeMetadata with monthly composite metadata.

        Notes:
            - Each month is centered on the 15th day (YYYY-MM-15)
            - Median aggregation is recommended for robustness to cloud artifacts
            - Parallel processing provides ~4-8x speedup on multi-core systems

        Examples:
            >>> # Auto-detect workers (recommended)
            >>> composites = meta.composite()

            >>> # Mean aggregation with manual workers
            >>> composites = meta.composite(agg_method="mean", num_workers=4)

            >>> # Quiet mode
            >>> composites = meta.composite(quiet=True)
        """

        input_dir = (
            pathlib.Path(input_dir).expanduser().resolve() if input_dir else self._dir()
        )
        output_dir = pathlib.Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        out_table = monthly_composites_s2(
            metadata=self.df,
            input_dir=input_dir,
            output_dir=output_dir,
            agg_method=agg_method,
            num_workers=num_workers,
            quiet=quiet,
        )

        if "outname" in out_table.columns and "id" not in out_table.columns:
            out_table["id"] = out_table["outname"].apply(lambda x: pathlib.Path(x).stem)

        return self._spawn(df=out_table, current_dir=output_dir)

    def interpolate(
        self,
        input_dir: str | pathlib.Path | None = None,
        output_dir: str | pathlib.Path = "interpolated",
        despike_threshold: float = 0.15,
        num_workers: int | None = None,
        chunk_size: int = 512,
        quiet: bool = False,
    ) -> SatCubeMetadata:
        """
        Interpolate missing values with temporal despiking.

        Removes temporal outliers (spikes) caused by undetected clouds or sensor artifacts,
        then fills gaps via linear interpolation. Processing is parallelized across spatial
        chunks with automatic size adjustment for optimal worker utilization.

        **⚠️ Warning:** Despiking may remove real extreme events (fires, floods, droughts)
        if they cause abrupt changes exceeding the threshold. For extreme event monitoring,
        increase despike_threshold to 0.25-0.35 or disable entirely (set to 999).

        Args:
            input_dir: Directory with input images. Default uses current pipeline dir.
            output_dir: Output directory for interpolated images. Default "interpolated".
            despike_threshold: Spike detection threshold (0-1 scale). Values jumping more
                than this from their 3-month rolling median are removed. Default 0.15.

                - 0.10-0.12: Aggressive artifact removal (may affect real rapid changes)
                - 0.15: Balanced (recommended) ⭐
                - 0.25-0.35: Conservative (preserves extreme events)

            num_workers: Number of spatial chunks to process in parallel. If None,
                auto-detects (typically CPU_cores/2). Chunk size auto-adjusts to
                utilize all workers. Default None.
            chunk_size: Initial spatial chunk size in pixels. Auto-reduces if needed.
                Default 512.
            quiet: Suppress progress bars. Default False.

        Returns:
            New SatCubeMetadata with interpolated images.

        Notes:
            - Requires loading full time series into memory
            - Parallel processing by spatial chunks (preserves temporal continuity)
            - Auto-adjusts chunk_size to maximize worker utilization
            - Default threshold (0.15) validated for post-composite time series

        Examples:
            >>> # Standard usage
            >>> interp = meta.interpolate()

            >>> # Conservative for extreme event monitoring
            >>> interp = meta.interpolate(despike_threshold=0.30)

            >>> # Aggressive for clean, stable regions
            >>> interp = meta.interpolate(despike_threshold=0.10)
        """
        input_dir = (
            pathlib.Path(input_dir).expanduser().resolve() if input_dir else self._dir()
        )
        output_dir = pathlib.Path(output_dir).resolve()

        new_df = interpolate_fn(
            metadata=self.df,
            input_dir=input_dir,
            output_dir=output_dir,
            despike_threshold=despike_threshold,
            num_workers=num_workers,
            chunk_size=chunk_size,
            quiet=quiet,
        )

        return self._spawn(df=new_df, current_dir=output_dir)

    def smooth(
        self,
        input_dir: str | pathlib.Path | None = None,
        output_dir: str | pathlib.Path = "smoothed",
        smooth_w: int = 7,
        smooth_p: int = 2,
        num_workers: int | None = None,
        chunk_size: int = 512,
        quiet: bool = False,
    ) -> SatCubeMetadata:
        """
        Apply Savitzky-Golay smoothing with climatology removal.

        **⚠️ Note:** This is an OPTIONAL refinement step AFTER interpolate().

        - interpolate() = FIXES errors (fills 0s, removes spikes)
        - smooth() = REFINES signal (reduces gradual noise)

        Use smooth() when:
            - You need very clean time series for ML models
            - You want to reduce small-scale temporal noise
            - You're calculating vegetation indices for agriculture

        Skip smooth() when:
            - You need to preserve all rapid changes (extreme events)
            - You're analyzing phenology with high temporal variability
            - You want maximum preservation of original signal

        Removes seasonal climatology, smooths anomalies, then adds climatology back.
        Processing is parallelized across spatial chunks for efficiency.

        Args:
            input_dir: Directory with input images. Default uses current pipeline dir.
            output_dir: Output directory for smoothed images. Default "smoothed".
            smooth_w: Savitzky-Golay window length (must be odd). Default 7.
            smooth_p: Savitzky-Golay polynomial order. Default 2.
            num_workers: Number of spatial chunks to process in parallel. If None,
                auto-detects (typically CPU_cores/2). Default None.
            chunk_size: Initial spatial chunk size in pixels. Auto-reduces if needed.
                Default 512.
            quiet: Suppress progress bars. Default False.

        Returns:
            New SatCubeMetadata with smoothed images.

        Notes:
            - Requires loading full time series into memory
            - Input should be complete series (run .interpolate() first)
            - Climatology removal preserves seasonal patterns
            - Parallel processing by spatial chunks

        Examples:
            >>> # Standard usage (after interpolate)
            >>> smoothed = meta.interpolate().smooth()

            >>> # Custom parameters
            >>> smoothed = meta.smooth(smooth_w=9, smooth_p=3)

            >>> # Skip if preserving rapid changes is critical
            >>> interp = meta.interpolate()  # Stop here, no smoothing
        """

        input_dir = (
            pathlib.Path(input_dir).expanduser().resolve() if input_dir else self._dir()
        )
        output_dir = pathlib.Path(output_dir).resolve()

        new_table = smooth_fn(
            metadata=self.df,
            input_dir=input_dir,
            output_dir=output_dir,
            smooth_w=smooth_w,
            smooth_p=smooth_p,
            num_workers=num_workers,
            chunk_size=chunk_size,
            quiet=quiet,
        )

        return self._spawn(df=new_table, current_dir=output_dir)

    def superresolve(
        self,
        input_dir: str | pathlib.Path | None = None,
        output_dir: str | pathlib.Path = "superresolved",
        *,
        variant: str = "SEN2SRLite",
        model_dir: str | pathlib.Path | None = None,
        device: str = "cpu",
        batch_size: int = 4,
        nworks: int = 4,
        cache: bool = False,
        band_indices: list[int] | None = None,
    ) -> SatCubeMetadata:
        """
        Super-resolve Sentinel-2 images from 10m to 2.5m resolution.

        Upscales imagery by 4× using SEN2SR deep learning models with seamless
        stitching and proper NoData preservation. Ideal for applications requiring
        higher spatial detail.

        **When to use super-resolution:**
            ✓ Urban mapping and building detection
            ✓ Precision agriculture (field boundaries, crop health)
            ✓ Infrastructure monitoring (roads, power lines)
            ✓ Small-scale land cover classification
            ✓ Visual enhancement for presentations

        **Model comparison:**
            - SEN2SRLite: Fast, good quality (~200MB, recommended)
            - SEN2SR: Slower, best quality (~600MB)

        Args:
            input_dir: Directory with input images. Default uses current pipeline dir.
            output_dir: Output directory for 2.5m images. Default "superresolved".
            variant: Model variant ("SEN2SRLite" or "SEN2SR"). Default "SEN2SRLite".
            model_dir: Custom model directory. If None, auto-manages. Default None.
            device: Device for inference ('cpu' or 'cuda'). Default "cpu".
            batch_size: Patches to process simultaneously (higher = faster + more memory).
                CPU: 4, GPU (8GB): 16, GPU (16GB+): 32. Default 4.
            nworks: DataLoader workers for patch loading. Higher improves GPU utilization.
                Default 4.
            cache: Skip already processed files. Default False.
            band_indices: Bands to upscale (1-indexed). If None, uses model defaults
                [2,3,4,5,6,7,8,9,12,13]. Example: [4,3,2] for RGB only. Default None.

        Returns:
            New SatCubeMetadata with 2.5m super-resolved images.

        Notes:
            - Models auto-download on first use (~200-600MB)
            - Processing time: CPU ~30s, GPU ~5s per 256×256 image
            - NoData pixels preserved
            - Transform updated automatically for GIS

        Examples:
            >>> # CPU inference (default)
            >>> sr = meta.superresolve()

            >>> # GPU inference (recommended for speed)
            >>> sr = meta.superresolve(
            ...     device="cuda",
            ...     batch_size=16,
            ...     nworks=8
            ... )

            >>> # Full model for best quality
            >>> sr = meta.superresolve(variant="SEN2SR", device="cuda")

            >>> # RGB bands only
            >>> sr = meta.superresolve(band_indices=[4, 3, 2])
        """
        in_dir = (
            pathlib.Path(input_dir).expanduser().resolve() if input_dir else self._dir()
        )
        out_dir = pathlib.Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        new_df = superresolve_fn(
            metadata=self.df,
            input_dir=in_dir,
            output_dir=out_dir,
            variant=variant,
            model_dir=model_dir,
            device=device,
            batch_size=batch_size,
            nworks=nworks,
            cache=cache,
            band_indices=band_indices,
        )

        return self._spawn(df=new_df, current_dir=out_dir)

    def process_all(
        self,
        output_dir: str | pathlib.Path = "final",
        *,
        min_clear_pct: float = 15.0,
        max_remaining_gaps: float = 1.0,
        num_workers: int | None = None,
        device: str = "cpu",
        batch_size: int = 4,
        sr_variant: str = "SEN2SRLite",
        despike_threshold: float = 0.15,
        quiet: bool = False,
    ) -> SatCubeMetadata:
        """
        Execute complete processing pipeline from raw images to super-resolved output.

        **Pipeline stages:**
        1. Alignment → co-register all images
        2. Cloud masking → detect and mask clouds
        3. Quality filtering → keep images with clear_pct >= min_clear_pct
        4. Re-alignment → improve alignment after cloud removal
        5. Gap filling → fill cloud gaps temporally
        6. Quality check → keep images with remaining_gaps_pct <= max_remaining_gaps
        7. Monthly composites → aggregate to monthly time series
        8. Interpolation → despike and interpolate
        9. Smoothing → apply Savitzky-Golay filter
        10. Super-resolution → upscale to 2.5m

        Args:
            output_dir: Final output directory. Default "final".
            min_clear_pct: Minimum clear percentage to keep images (0-100). Default 15.
            max_remaining_gaps: Maximum remaining gaps percentage after gapfill (0-100). Default 1.
            num_workers: Number of parallel workers. If None, auto-detects. Default None.
            device: Device for ML models ('cpu' or 'cuda'). Default "cpu".
            batch_size: Batch size for cloud masking. Default 4.
            sr_variant: Super-resolution model ("SEN2SRLite" or "SEN2SR"). Default "SEN2SRLite".
            despike_threshold: Interpolation spike threshold. Default 0.15.
            quiet: Suppress all progress bars and logs. Default False.

        Returns:
            Final SatCubeMetadata with super-resolved images.

        Examples:
            >>> # Complete pipeline with defaults
            >>> final = meta.process_all()

            >>> # GPU acceleration
            >>> final = meta.process_all(
            ...     device="cuda",
            ...     num_workers=16,
            ...     batch_size=16
            ... )

            >>> # Strict quality filters
            >>> final = meta.process_all(
            ...     min_clear_pct=25,
            ...     max_remaining_gaps=0.5
            ... )
        """

        import time
        from datetime import timedelta

        output_dir = pathlib.Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if num_workers is None:
            import os

            num_workers = min(os.cpu_count() or 4, 8)

        stages = [
            "Alignment",
            "Cloud masking",
            "Quality filter",
            "Re-alignment",
            "Gap filling",
            "Gap quality check",
            "Monthly composites",
            "Interpolation",
            "Smoothing",
            "Super-resolution",
        ]

        total_stages = len(stages)
        start_time = time.time()

        def log_stage(stage_num, stage_name, desc=""):
            if not quiet:
                elapsed = time.time() - start_time
                pct = (stage_num / total_stages) * 100
                eta = (
                    (elapsed / stage_num) * (total_stages - stage_num)
                    if stage_num > 0
                    else 0
                )
                eta_str = str(timedelta(seconds=int(eta)))
                print(f"\n{'='*70}")
                print(f"[{pct:5.1f}%] Stage {stage_num}/{total_stages}: {stage_name}")
                if desc:
                    print(f"         {desc}")
                print(
                    f"         Elapsed: {timedelta(seconds=int(elapsed))} | ETA: {eta_str}"
                )
                print(f"{'='*70}\n")

        current = self

        # Stage 1: Alignment
        log_stage(1, stages[0], "Co-registering all scenes to reference")
        aligned = current.align(
            output_dir=output_dir / "01_aligned", num_workers=num_workers, cache=False
        )

        # Stage 2: Cloud masking
        log_stage(2, stages[1], f"Detecting clouds (device={device})")
        masked = aligned.cloud_masking(
            output_dir=output_dir / "02_masked",
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            save_mask=True,
            cache=False,
        )

        # Stage 3: Quality filtering
        log_stage(3, stages[2], f"Keeping images with clear_pct >= {min_clear_pct}%")
        initial_count = len(masked)
        masked_filtered = masked[masked["clear_pct"] >= min_clear_pct]
        filtered_count = len(masked_filtered)
        if not quiet:
            print(f"         Filtered: {initial_count} → {filtered_count} images")

        if filtered_count == 0:
            raise ValueError(
                f"No images remain after clear_pct >= {min_clear_pct} filter"
            )

        # Stage 4: Re-alignment
        log_stage(4, stages[3], "Re-aligning after cloud removal")
        realigned = masked_filtered.align(
            output_dir=output_dir / "03_aligned2", num_workers=num_workers, cache=False
        )

        # Stage 5: Gap filling
        log_stage(5, stages[4], "Filling cloud gaps using temporal matching")
        filled = realigned.gapfill(
            output_dir=output_dir / "04_gapfilled", num_workers=num_workers, quiet=quiet
        )

        # Stage 6: Gap quality check
        log_stage(
            6, stages[5], f"Keeping images with remaining_gaps <= {max_remaining_gaps}%"
        )
        initial_gap_count = len(filled)
        filled_clean = filled[filled["remaining_gaps_pct"] <= max_remaining_gaps]
        clean_count = len(filled_clean)
        if not quiet:
            print(f"         Filtered: {initial_gap_count} → {clean_count} images")

        if clean_count == 0:
            raise ValueError(
                f"No images remain after remaining_gaps_pct <= {max_remaining_gaps} filter"
            )

        # Stage 7: Monthly composites
        log_stage(7, stages[6], "Aggregating to monthly time series (median)")
        composites = filled_clean.composite(
            output_dir=output_dir / "05_monthly",
            agg_method="median",
            num_workers=num_workers,
            quiet=quiet,
        )

        # Stage 8: Interpolation
        log_stage(
            8, stages[7], f"Despiking (threshold={despike_threshold}) + interpolation"
        )
        interpolated = composites.interpolate(
            output_dir=output_dir / "06_interpolated",
            despike_threshold=despike_threshold,
            num_workers=num_workers,
            quiet=quiet,
        )

        # Stage 9: Smoothing
        log_stage(9, stages[8], "Applying Savitzky-Golay smoothing")
        smoothed = interpolated.smooth(
            output_dir=output_dir / "07_smoothed",
            smooth_w=7,
            smooth_p=2,
            num_workers=num_workers,
            quiet=quiet,
        )

        # Stage 10: Super-resolution
        log_stage(10, stages[9], f"Upscaling to 2.5m ({sr_variant})")
        final = smoothed.superresolve(
            output_dir=output_dir / "08_superresolved",
            variant=sr_variant,
            device=device,
            batch_size=batch_size if device == "cuda" else 4,
            nworks=num_workers if device == "cuda" else 4,
            cache=False,
        )

        # Summary
        if not quiet:
            total_time = time.time() - start_time
            print(f"\n{'='*70}")
            print("✓ PIPELINE COMPLETE")
            print(f"{'='*70}")
            print(f"Total time: {timedelta(seconds=int(total_time))}")
            print(f"Output directory: {output_dir / '08_superresolved'}")
            print(f"Final images: {len(final)}")
            print(f"{'='*70}\n")

        return final

    def filter_metadata(self, condition) -> SatCubeMetadata:
        filtered_df = self.df[condition(self.df)]
        return self._spawn(df=filtered_df, current_dir=self._current_dir)

    def __repr__(self) -> str:
        return self.df.__repr__()

    __str__ = __repr__

    def _repr_html_(self) -> str:
        if self._table is not None:
            table_html = getattr(self._table, "_repr_html_", None)
            if callable(table_html):
                return table_html()
        html = getattr(self.df, "_repr_html_", None)
        return html() if callable(html) else self.df.__repr__()

    def __getattr__(self, item):
        return getattr(self.df, item)

    def __getitem__(self, key):
        if isinstance(key, pd.Series) or isinstance(key, pd.DataFrame):
            filtered_df = self.df[key]
            return self._spawn(df=filtered_df, current_dir=self._current_dir)
        return self.df[key]

    def __len__(self):
        return len(self.df)

    def update_metadata(self, new_df: pd.DataFrame) -> None:
        """Update metadata and save to Parquet."""
        self.df = new_df
        self._save_metadata()
