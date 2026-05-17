import os
import pytest
import numpy as np
import xarray as xr
from weathergraph import WeatherGraphModel
from weathergraph.tile_bundle import grid_shape_from_resolution

@pytest.fixture
def resolution():
    res = os.environ.get("WEATHERGRAPH_TEST_RESOLUTION")
    if res is None:
        return 1.0 # Default
    return float(res)

def test_resolution_execution(resolution):
    base_dir = os.environ.get("TEST_ARTIFACTS_DIR", "test_artifacts")
    res_dir = os.path.join(base_dir, f"res_{resolution}")
    
    if not os.path.exists(res_dir):
        pytest.skip(f"Test artifacts for resolution {resolution} not found in {res_dir}")
        
    weights_dir = os.path.join(res_dir, "weights")
    
    if resolution <= 0.1:
        # Tiled execution
        bundle_path = os.path.join(res_dir, "bundle")
        model = WeatherGraphModel(
            model_path=None, # Not used when bundle is provided
            weights_dir=weights_dir,
            tile_bundle_path=bundle_path,
            spatial_tiling=True
        )
    else:
        # Single model execution
        model_path = os.path.join(res_dir, "models", "model.onnx")
        model = WeatherGraphModel(
            model_path=model_path,
            weights_dir=weights_dir,
            reference_grid_resolution_degrees=resolution
        )
        
    # Generate a single time slice of dummy data for this resolution
    grid_shape = grid_shape_from_resolution(resolution)
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    vars = ['z', 'q', 't', 'u', 'v', 'w']
    
    lat = np.linspace(90, -90, grid_shape[0])
    lon = np.linspace(0, 360, grid_shape[1], endpoint=False)
    
    ds = xr.Dataset(
        {var: (["level", "lat", "lon"], np.random.rand(len(levels), len(lat), len(lon)).astype(np.float32)) for var in vars},
        coords={"level": levels, "lat": lat, "lon": lon}
    )
    
    # Run prediction
    prediction = model.predict_one_step(ds)
    
    # Validation
    expected_nodes = grid_shape[0] * grid_shape[1]
    assert prediction.shape == (1, expected_nodes, 78)
    assert np.isfinite(prediction).all()
    print(f"Resolution {resolution} passed! Output shape: {prediction.shape}")
