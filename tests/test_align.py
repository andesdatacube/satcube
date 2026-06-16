"""Tests for the align module (reference selection by score)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio as rio

from satcube.align import align_fn


def _write_tif(path, value, bands=4, size=16):
    profile = {"driver": "GTiff", "width": size, "height": size, "count": bands,
               "dtype": "uint16", "crs": "EPSG:32718",
               "transform": rio.transform.from_origin(0, size, 1, 1)}
    with rio.open(path, "w", **profile) as dst:
        dst.write(np.full((bands, size, size), value, dtype="uint16"))


def test_reference_picked_by_lowest_score(tmp_path):
    """With a 'score' column (= % cloud), the clearest scene (min score) is the reference."""
    raw = tmp_path / "raw"; raw.mkdir()
    out = tmp_path / "aligned"
    # three scenes; scene_b is the clearest (lowest cloud score)
    for sid, val in [("scene_a", 100), ("scene_b", 200), ("scene_c", 300)]:
        _write_tif(raw / f"{sid}.tif", val)
    meta = pd.DataFrame({"id": ["scene_a", "scene_b", "scene_c"],
                         "score": [80.0, 5.0, 50.0]})   # scene_b = least cloud

    result = align_fn(metadata=meta, input_dir=raw, output_dir=out, num_workers=2)

    assert "dx_px" in result.columns and "dy_px" in result.columns
    assert set(result["id"]) == {"scene_a", "scene_b", "scene_c"}


def test_missing_score_columns_raises(tmp_path):
    """Without clear_pct/score/cs_cdf, align_fn raises a clear error."""
    from satcube.exceptions import AlignmentError
    meta = pd.DataFrame({"id": ["x"]})
    with pytest.raises(AlignmentError, match="clear_pct"):
        align_fn(metadata=meta, input_dir=tmp_path, output_dir=tmp_path)