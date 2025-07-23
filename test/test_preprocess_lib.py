import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
from src.preprocess_lib import (
    find_fid_column,
    extract_date_from_name,
    buffer_geom,
    select_best_scenes,
    get_activation_date,
    rasterize_mask,
    process_scene,
    PERCENTILE_SAMPLE_SIZE,
    TILE_SIZE,
    ACTIVATION_DATES,
    logger
)
import logging
from datetime import datetime, timezone

def test_find_fid_column():
    gdf = gpd.GeoDataFrame({'fid_foo': [1, 2], 'geometry': [Polygon([(0,0),(1,0),(1,1),(0,1)])]*2})
    assert find_fid_column(gdf) == 'fid_foo'
    gdf = gpd.GeoDataFrame({'id': [1, 2], 'geometry': [Polygon([(0,0),(1,0),(1,1),(0,1)])]*2})
    assert find_fid_column(gdf) == 'id'
    gdf = gpd.GeoDataFrame({'foo': [1, 2], 'geometry': [Polygon([(0,0),(1,0),(1,1),(0,1)])]*2})
    assert find_fid_column(gdf) is None

def test_extract_date_from_name():
    assert extract_date_from_name('foo20230101bar') == datetime(2023, 1, 1, tzinfo=timezone.utc)
    assert extract_date_from_name('no_date_here') is None

def test_buffer_geom():
    poly = Polygon([(0,0),(1,0),(1,1),(0,1)])
    crs = gpd.GeoSeries([poly], crs=4326).crs
    buf = buffer_geom(poly, crs)
    assert buf.area > poly.area

def test_select_best_scenes():
    dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
    scenes = [
        {'date': dt, 'bands': {'red': 'dummy'}, 'geometry': Polygon([(0,0),(1,0),(1,1),(0,1)])},
        {'date': dt.replace(year=2024), 'bands': {'red': 'dummy'}, 'geometry': Polygon([(0,0),(1,0),(1,1),(0,1)])}
    ]
    pre, post = select_best_scenes(scenes, dt.replace(month=6))
    assert pre is not None and post is not None

def test_get_activation_date(tmp_path):
    # Test with hardcoded dict
    call_dir = tmp_path / 'Call_919_Turkey'
    call_dir.mkdir()
    assert get_activation_date(call_dir) == ACTIVATION_DATES['Call_919_Turkey']
    # Test with collection.json
    call_dir2 = tmp_path / 'Call_999_Test'
    call_dir2.mkdir()
    (call_dir2 / 'collection.json').write_text('Date of Charter Activation: 2022-12-31')
    dt = get_activation_date(call_dir2)
    assert dt.year == 2022 and dt.month == 12 and dt.day == 31

def test_rasterize_mask():
    poly = Polygon([(0,0),(1,0),(1,1),(0,1)])
    out_shape = (10, 10)
    from affine import Affine
    mask = rasterize_mask(poly, None, Affine(1,0,0,0,-1,10), out_shape)
    assert mask.shape == out_shape
    assert np.any(mask == 1)

def test_process_scene_index_validation():
    # Test index validation in process_scene
    scene = {
        'bands': {'red': 'dummy', 'green': 'dummy', 'blue': 'dummy'},
        'gsd': 1.0,
        'geometry': Polygon([(0,0),(1,0),(1,1),(0,1)]),
        'scene_id': 'dummy',
    }
    bbox = (0,0,1,1)
    class DummyCRS:
        def to_epsg(self): return 4326
    # Should return None because band paths are dummy
    arr = process_scene(scene, bbox, DummyCRS(), True, scene['geometry'], percentiles=None)
    assert arr is None

def test_logger_level():
    logger.setLevel(logging.DEBUG)
    assert logger.level == logging.DEBUG 