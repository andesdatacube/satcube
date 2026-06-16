import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.transform import from_origin

from satcube.composite import monthly_composites_s2
from satcube.interpolate import interpolate_fn, _interp_axis0, _rolling_median3
from satcube.smooth import smooth_fn


def _write(path, arr_u16):
    b, h, w = arr_u16.shape
    prof = {"driver": "GTiff", "height": h, "width": w, "count": b, "dtype": "uint16",
            "crs": "EPSG:32718", "transform": from_origin(0, h, 1, 1), "nodata": 0}
    with rio.open(path, "w", **prof) as dst:
        dst.write(arr_u16)


def _const(val, b=4, h=16, w=16):
    return np.full((b, h, w), val, dtype="uint16")


def test_composite_monthly_median(tmp_path):
    indir = tmp_path / "in"; indir.mkdir()
    _write(indir / "a.tif", _const(2000)); _write(indir / "b.tif", _const(4000)); _write(indir / "c.tif", _const(1000))
    meta = pd.DataFrame({"id": ["a", "b", "c"], "date": ["2022-01-05", "2022-01-20", "2022-02-10"]})
    out = tmp_path / "out"
    df = monthly_composites_s2(meta, input_dir=indir, output_dir=out, quiet=True)
    assert set(df["date"]) == {"2022-01-15", "2022-02-15"}
    with rio.open(out / "2022-01-15.tif") as src:
        assert np.allclose(src.read(), 3000, atol=1)


def test_composite_cache(tmp_path):
    indir = tmp_path / "in"; indir.mkdir(); _write(indir / "a.tif", _const(2000))
    meta = pd.DataFrame({"id": ["a"], "date": ["2022-01-05"]}); out = tmp_path / "out"
    monthly_composites_s2(meta, input_dir=indir, output_dir=out, quiet=True)
    mt = (out / "2022-01-15.tif").stat().st_mtime
    monthly_composites_s2(meta, input_dir=indir, output_dir=out, cache=True, quiet=True)
    assert (out / "2022-01-15.tif").stat().st_mtime == mt


def test_interp_axis0_known():
    a = np.array([0.2, np.nan, np.nan, 0.5], dtype="float32").reshape(4, 1)
    assert np.allclose(_interp_axis0(a).ravel(), [0.2, 0.3, 0.4, 0.5], atol=1e-6)


def test_interp_axis0_edges_fill():
    a = np.array([np.nan, 0.3, 0.3, np.nan], dtype="float32").reshape(4, 1)
    assert np.allclose(_interp_axis0(a).ravel(), [0.3, 0.3, 0.3, 0.3])


def test_rolling_median3_matches():
    from scipy.ndimage import median_filter
    a = np.random.rand(7, 2, 8, 8).astype("float32")
    assert np.allclose(_rolling_median3(a), median_filter(a, size=(3, 1, 1, 1), mode="nearest"))


def test_interpolate_removes_spike(tmp_path):
    indir = tmp_path / "in"; indir.mkdir()
    for i, v in enumerate([2000, 2000, 7000, 2000, 2000]):
        _write(indir / f"2022-{i+1:02d}-15.tif", _const(v))
    meta = pd.DataFrame({"date": [f"2022-{i+1:02d}-15" for i in range(5)]})
    out = tmp_path / "out"
    interpolate_fn(meta, input_dir=indir, output_dir=out, despike_threshold=0.15, num_workers=2, quiet=True)
    with rio.open(out / "2022-03-15.tif") as src:
        assert np.all(src.read() < 3000)


def test_smooth_runs_and_ranges(tmp_path):
    indir = tmp_path / "in"; indir.mkdir()
    rng = np.random.default_rng(0)
    names = [f"2022-{i+1:02d}-15.tif" for i in range(6)]
    for n in names:
        _write(indir / n, (2000 + rng.integers(-100, 100, size=(4, 16, 16))).astype("uint16"))
    meta = pd.DataFrame({"outname": names, "date": [f"2022-{i+1:02d}-15" for i in range(6)]})
    out = tmp_path / "out"
    df = smooth_fn(meta, input_dir=indir, output_dir=out, smooth_w=5, num_workers=2, quiet=True)
    assert len(df) == 6 and "id" in df.columns
    for n in names:
        with rio.open(out / n) as src:
            assert src.read().max() <= 10000


def test_smooth_cache(tmp_path):
    indir = tmp_path / "in"; indir.mkdir()
    names = [f"2022-{i+1:02d}-15.tif" for i in range(4)]
    for n in names:
        _write(indir / n, _const(2000))
    meta = pd.DataFrame({"outname": names, "date": [f"2022-{i+1:02d}-15" for i in range(4)]})
    out = tmp_path / "out"
    smooth_fn(meta, input_dir=indir, output_dir=out, num_workers=2, quiet=True)
    mt = (out / names[0]).stat().st_mtime
    smooth_fn(meta, input_dir=indir, output_dir=out, cache=True, num_workers=2, quiet=True)
    assert (out / names[0]).stat().st_mtime == mt