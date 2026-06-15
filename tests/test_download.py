"""Tests for the download module of satcube (metadata, metadata_polygon, download)."""

from __future__ import annotations

import pandas as pd
import pytest

from satcube.objects import SatCubeMetadata


class _FakeTable:
    """Minimal stand-in for a cubexpress RequestTable."""

    def __init__(self, df):
        self.df = df

    def mosaic(self, by):
        return self


def _make_meta(rows=3):
    df = pd.DataFrame(
        {
            "id": [f"S2_2023010{i}_-77_-9" for i in range(rows)],
            "date": [f"2023-01-0{i}" for i in range(rows)],
            "score": [10.0 * i for i in range(rows)],
            "coverage_pct": [100.0] * rows,
        }
    )
    meta = SatCubeMetadata(df=df)
    meta._table = _FakeTable(df)
    return meta


# --- download() method (mocked express, no real GEE) ---

def test_download_calls_express(tmp_path, monkeypatch):
    """download() calls cubexpress.express with the table and output folder."""
    import satcube.objects as objmod

    calls = {}

    def fake_express(table, outfolder, nworkers=8):
        calls["outfolder"] = outfolder
        calls["nworkers"] = nworkers

    monkeypatch.setattr(objmod.cubexpress, "express", fake_express)

    meta = _make_meta()
    meta.download(output_dir=str(tmp_path), num_workers=4)

    assert calls["nworkers"] == 4
    assert str(tmp_path) in str(calls["outfolder"])


def test_download_returns_metadata_with_score(tmp_path, monkeypatch):
    """The returned df carries id/date/score/coverage_pct through."""
    import satcube.objects as objmod

    monkeypatch.setattr(objmod.cubexpress, "express", lambda *a, **k: None)

    meta = _make_meta()
    out = meta.download(output_dir=str(tmp_path))

    assert isinstance(out, SatCubeMetadata)
    for col in ("id", "date", "score", "coverage_pct"):
        assert col in out.df.columns


def test_download_without_table_raises(tmp_path):
    """download() needs a RequestTable (built by metadata)."""
    meta = SatCubeMetadata(df=pd.DataFrame({"id": ["x"]}))
    with pytest.raises(ValueError, match="no RequestTable"):
        meta.download(output_dir=str(tmp_path))


# --- metadata / metadata_polygon (real GEE, skipped without it) ---

@pytest.mark.needs_ee
def test_metadata_mosaic_one_row_per_date():
    """metadata(mosaic=True) gives one row per date, with a score column."""
    import satcube

    meta = satcube.metadata(
        lon=-77.06, lat=-9.54, width=64, height=64,
        start="2023-01-01", end="2023-02-01", mosaic=True,
    )
    dates = meta.df["date"].tolist()
    assert len(dates) == len(set(dates))
    assert "score" in meta.df.columns


@pytest.mark.needs_ee
def test_metadata_polygon_single_part_works():
    """A single-part polygon (FeatureCollection) discovers scenes with a score."""
    import satcube

    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-77.10, -9.65],
                            [-77.07, -9.65],
                            [-77.07, -9.62],
                            [-77.10, -9.62],
                            [-77.10, -9.65],
                        ]
                    ],
                },
            }
        ],
    }
    meta = satcube.metadata_polygon(gj, start="2023-01-01", end="2023-02-01")
    assert "score" in meta.df.columns


def test_metadata_polygon_multipart_raises():
    """A MultiPolygon with several parts is rejected with a clear message."""
    import satcube
    import shapely

    p1 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    p2 = shapely.Polygon([(5, 5), (6, 5), (6, 6), (5, 6), (5, 5)])
    mp = shapely.MultiPolygon([p1, p2])

    with pytest.raises(ValueError, match="single polygon"):
        satcube.metadata_polygon(mp, start="2023-01-01", end="2023-02-01")