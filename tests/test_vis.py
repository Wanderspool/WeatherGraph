import pytest
import numpy as np
import xarray as xr
import os

from weathergraph.vis import create_interactive_map, create_animation, VIS_AVAILABLE

pytestmark = pytest.mark.skipif(not VIS_AVAILABLE, reason="Visualization dependencies not installed")

@pytest.fixture
def dummy_dataset():
    # Create a small dummy dataset with time, lat, lon dimensions
    time = np.arange(3)
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)
    data = np.random.rand(3, 10, 20)
    
    ds = xr.Dataset(
        {'t': (['time', 'lat', 'lon'], data)},
        coords={'time': time, 'lat': lat, 'lon': lon}
    )
    return ds

@pytest.fixture
def dummy_dataset_with_level():
    # Create a dummy dataset with time, level, lat, lon dimensions
    time = np.arange(2)
    level = np.array([500, 850, 1000])
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 180, 10)
    data = np.random.rand(2, 3, 5, 10)
    
    ds = xr.Dataset(
        {'z': (['time', 'level', 'lat', 'lon'], data)},
        coords={'time': time, 'level': level, 'lat': lat, 'lon': lon}
    )
    return ds


def test_create_interactive_map(dummy_dataset):
    import folium
    m = create_interactive_map(dummy_dataset, variable='t', time_index=0)
    assert isinstance(m, folium.Map)
    
    # Check that bounds and image overlay were added (folium internal structure)
    children = list(m._children.values())
    assert any(isinstance(child, folium.raster_layers.ImageOverlay) for child in children)
    assert any("Colormap" in type(child).__name__ or "ColorMap" in type(child).__name__ for child in children)


def test_create_interactive_map_with_level(dummy_dataset_with_level):
    import folium
    m = create_interactive_map(dummy_dataset_with_level, variable='z', time_index=0)
    assert isinstance(m, folium.Map)


def test_create_interactive_map_missing_variable(dummy_dataset):
    with pytest.raises(ValueError, match="Variable 'missing' not found in dataset"):
        create_interactive_map(dummy_dataset, variable='missing')


def test_create_animation_mp4(dummy_dataset, tmp_path):
    output_path = tmp_path / "test.mp4"
    create_animation(dummy_dataset, variable='t', output_path=str(output_path), format='mp4', fps=2)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_create_animation_gif(dummy_dataset_with_level, tmp_path):
    output_path = tmp_path / "test.gif"
    create_animation(dummy_dataset_with_level, variable='z', output_path=str(output_path), format='gif', fps=2)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_create_animation_unsupported_format(dummy_dataset, tmp_path):
    output_path = tmp_path / "test.avi"
    with pytest.raises(ValueError, match="Unsupported format 'avi'"):
        create_animation(dummy_dataset, variable='t', output_path=str(output_path), format='avi')


def test_create_animation_no_time_dim():
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 180, 10)
    data = np.random.rand(5, 10)
    
    ds = xr.Dataset(
        {'t': (['lat', 'lon'], data)},
        coords={'lat': lat, 'lon': lon}
    )
    with pytest.raises(ValueError, match="Dataset does not have a 'time' dimension"):
        create_animation(ds, variable='t', output_path="dummy.mp4")
