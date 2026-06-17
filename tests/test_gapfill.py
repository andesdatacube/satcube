from collections import OrderedDict
import tracemalloc

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.transform import from_origin

from satcube.gapfill import _inpaint_bands, _load_ref, _fill_one, gapfill_fn


def _write_tif(path, arr_u16):
    bands, h, w = arr_u16.shape
    profile = {"driver": "GTiff", "height": h, "width": w, "count": bands,
               "dtype": "uint16", "crs": "EPSG:32718",
               "transform": from_origin(0, h, 1, 1), "nodata": 0}
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr_u16)


def _scene(h=32, w=32, bands=4, value=2000, seed=0):
    rng = np.random.default_rng(seed)
    return (value + rng.integers(-100, 100, size=(bands, h, w))).astype("uint16")


def test_inpaint_bands_fills_without_nan():
    data = (np.random.rand(4, 16, 16) * 0.4).astype("float32")
    bad = np.zeros((16, 16), dtype=bool); bad[5:11, 5:11] = True
    data[:, bad] = np.nan
    out = _inpaint_bands(data, bad, radius=3)
    assert not np.isnan(out).any()
    assert (out >= 0).all() and (out <= 1).all()


def test_load_ref_caches(tmp_path):
    p = tmp_path / "ref.tif"; _write_tif(p, _scene())
    cache = OrderedDict()
    ref1, _ = _load_ref(p, cache)
    ref2, _ = _load_ref(p, cache)
    assert ref1 is ref2
    assert ref1.dtype == np.float32


def test_load_ref_lru_bounded(tmp_path):
    cache = OrderedDict()
    for i in range(20):
        p = tmp_path / f"r{i}.tif"; _write_tif(p, _scene(seed=i))
        _load_ref(p, cache, max_cache=8)
    assert len(cache) <= 8


def test_fill_one_no_gaps_returns_zero(tmp_path):
    p = tmp_path / "img.tif"; out_dir = tmp_path / "out"; out_dir.mkdir()
    _write_tif(p, _scene(seed=1))
    dates = np.array(["2022-01-01"], dtype="datetime64[ns]")
    stem, gaps = _fill_one(img_path=p, ref_paths=[p], dates=dates, this_date=dates[0],
                           method="histogram_matching", out_dir=out_dir, threshold=0.15,
                           enable_inpainting=False)
    assert gaps == 0.0
    assert (out_dir / "img.tif").exists()


def test_fill_one_fills_hole(tmp_path):
    out_dir = tmp_path / "out"; out_dir.mkdir()
    tgt = _scene(seed=1); tgt[:, 10:20, 10:20] = 0
    ptgt = tmp_path / "t.tif"; _write_tif(ptgt, tgt)
    pref = tmp_path / "r.tif"; _write_tif(pref, _scene(seed=2))
    dates = np.array(["2022-01-01", "2022-01-06"], dtype="datetime64[ns]")
    stem, gaps = _fill_one(img_path=ptgt, ref_paths=[ptgt, pref], dates=dates, this_date=dates[0],
                           method="histogram_matching", out_dir=out_dir, threshold=1.0,
                           enable_inpainting=True)
    assert gaps < 5.0
    with rio.open(out_dir / "t.tif") as src:
        filled = src.read()
    assert (filled[:, 10:20, 10:20] > 0).any()


def test_gapfill_fn_end_to_end(tmp_path):
    in_dir = tmp_path / "masked"; out_dir = tmp_path / "gapfilled"; in_dir.mkdir()
    ids = ["s0", "s1", "s2"]; dates = ["2022-01-01", "2022-01-06", "2022-01-11"]
    a0 = _scene(seed=1); a0[:, 8:16, 8:16] = 0
    _write_tif(in_dir / "s0.tif", a0)
    _write_tif(in_dir / "s1.tif", _scene(seed=2))
    _write_tif(in_dir / "s2.tif", _scene(seed=3))
    meta = pd.DataFrame({"id": ids, "date": dates})
    out = gapfill_fn(meta, input_dir=in_dir, output_dir=out_dir, num_workers=2, quiet=True)
    assert "remaining_gaps_pct" in out.columns and len(out) == 3
    for i in ids:
        assert (out_dir / f"{i}.tif").exists()
    assert not (out_dir / "_round0").exists() and not (out_dir / "_round1").exists()


def test_memory_bounded_on_larger_images(tmp_path):
    in_dir = tmp_path / "m"; out_dir = tmp_path / "g"; in_dir.mkdir()
    ids = [f"s{i}" for i in range(6)]
    dates = [f"2022-01-{2*i+1:02d}" for i in range(6)]
    for i, sid in enumerate(ids):
        arr = _scene(h=256, w=256, bands=13, seed=i)
        if i == 0:
            arr[:, 80:160, 80:160] = 0
        _write_tif(in_dir / f"{sid}.tif", arr)
    meta = pd.DataFrame({"id": ids, "date": dates})
    tracemalloc.start()
    gapfill_fn(meta, input_dir=in_dir, output_dir=out_dir, num_workers=2, quiet=True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / 1e6 < 400