from __future__ import annotations

from datetime import date, timedelta

import ee
import cubexpress

from satcube.objects import SatCubeMetadata


def _cloud_score(image, geometry, source_ids=None):
    """Cloud percentage over the ROI (0-100, higher = cloudier).

    Uses CloudScore+ (cs_cdf): a pixel with cs_cdf >= 0.65 counts as clear.
    Returns the percentage of CLOUDY pixels, so:
        0   -> fully clear scene (ideal)
        100 -> fully cloudy scene (unusable)
    """
    csplus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    if source_ids is not None:
        cs = (
            csplus.filter(ee.Filter.inList("system:index", ee.List(source_ids)))
            .select("cs_cdf")
            .mosaic()
        )
    else:
        cs = csplus.filter(
            ee.Filter.eq("system:index", image.get("system:index"))
        ).first()
        cs = ee.Image(
            ee.Algorithms.If(cs, cs, ee.Image.constant(0).rename("cs_cdf"))
        ).select("cs_cdf")
    frac_clear = cs.gte(0.65).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=10, maxPixels=int(1e9)
    ).get("cs_cdf")
    frac_clear = ee.Number(ee.Algorithms.If(frac_clear, frac_clear, 0))
    return ee.Number(1).subtract(frac_clear).multiply(100)   # % CLOUD


def metadata(
    lon: float,
    lat: float,
    width: int,
    height: int,
    *,
    scale: float = 10,
    start: str = "2015-01-01",
    end: str | None = None,
    max_cloud: float = 100.0,
    mosaic: bool = True,
    score_nworkers: int = 8,
    score_batch: int = 25,
) -> SatCubeMetadata:
    """Discover Sentinel-2 imagery over a patch, scored by cloud percentage.

    Args:
        lon: Patch center longitude (WGS-84 degrees).
        lat: Patch center latitude (WGS-84 degrees).
        width: Patch width in pixels.
        height: Patch height in pixels.
        scale: Meters per pixel (Sentinel-2 native = 10).
        start: Start date 'YYYY-MM-DD'. Default '2015-01-01'.
        end: End date 'YYYY-MM-DD'. If None, defaults to yesterday.
        max_cloud: Keep scenes whose cloud percentage (0-100) is <= this.
            0 keeps only perfectly clear scenes; 100 keeps everything. Default 100.
        mosaic: If True, fuse same-date scenes before scoring. Default True.

    Returns:
        SatCubeMetadata with id, image, date, coverage_pct, score (score = % cloud).
    """
    if end is None:
        end = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    rt = cubexpress.point_to_rt(lon=lon, lat=lat, width=width, height=height, scale=scale)
    table = cubexpress.discover_images("COPERNICUS/S2_HARMONIZED", rt, start, end)

    if mosaic:
        table = table.mosaic(by="date")

    scored = cubexpress.add_metrics(
        table,
        score_fn=_cloud_score,
        nworkers=score_nworkers,
        batch_size=score_batch,
    )

    kept = scored[scored.df["score"] <= max_cloud]

    sat = SatCubeMetadata(df=kept.df.copy().reset_index(drop=True))
    sat._table = kept
    return sat


def metadata_polygon(
    geometry,
    *,
    scale: float = 10,
    start: str = "2015-01-01",
    end: str | None = None,
    max_cloud: float = 100.0,
    mosaic: bool = True,
) -> SatCubeMetadata:
    """Discover Sentinel-2 imagery over a single polygon, scored by cloud percentage.

    Accepts one polygon as a shapely Polygon, a WKT string, or a GeoJSON dict
    (geometry, Feature, or single-feature FeatureCollection). MultiPolygons are
    accepted only when they have a single part; pass one polygon at a time if
    you have several.

    Args:
        geometry: A single shapely Polygon, WKT string, or GeoJSON dict.
        scale: Meters per pixel (Sentinel-2 native = 10).
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD'. If None, defaults to yesterday.
        max_cloud: Keep scenes whose cloud percentage (0-100) is <= this.
            0 keeps only perfectly clear scenes; 100 keeps everything. Default 100.
        mosaic: If True, one image per date before scoring. Default True.

    Returns:
        SatCubeMetadata with id, image, date, coverage_pct, score (score = % cloud).

    Raises:
        ValueError: if the geometry is a MultiPolygon with several parts.
    """
    import shapely

    if end is None:
        end = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    poly = cubexpress.to_polygon(geometry)

    if isinstance(poly, shapely.MultiPolygon):
        parts = list(poly.geoms)
        if len(parts) == 1:
            poly = parts[0]
        else:
            raise ValueError(
                f"satcube.metadata_polygon accepts a single polygon, but got a "
                f"MultiPolygon with {len(parts)} parts. Pass one polygon at a time "
                f"(e.g. loop over geometry.geoms)."
            )

    rt = cubexpress.polygon_to_rt(poly, scale=scale)
    table = cubexpress.discover_images("COPERNICUS/S2_HARMONIZED", rt, start, end)

    if mosaic:
        table = table.mosaic(by="date")

    scored = cubexpress.add_metrics(table, score_fn=_cloud_score)
    kept = scored[scored.df["score"] <= max_cloud]

    sat = SatCubeMetadata(df=kept.df.copy().reset_index(drop=True))
    sat._table = kept
    return sat