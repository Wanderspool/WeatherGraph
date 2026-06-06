"""
Examples of configuring Spatial Tiling for high-resolution inference.
"""
import pytest
from unittest.mock import patch, MagicMock

@patch('weathergraph.model.WeatherGraphModel')
def test_spatial_tiling(MockModel):
    # Setup mock
    model_instance = MockModel.return_value

    # --8<-- [start:tiling_config]
    from weathergraph.model import WeatherGraphModel
    
    # 1. Initialize the model with tiling enabled
    # This configuration is required when running 0.1 degree target resolution
    # on hardware that cannot fit the entire global graph in memory.
    model = WeatherGraphModel(
        model_path="models/weather_gnn.onnx",
        weights_dir="data",
        spatial_tiling=True,
        tile_bundle_path="tile_bundle/manifest.json",
        reference_grid_shape=(1801, 3600), # Exact 0.1 deg shape
        reference_grid_resolution_degrees=0.1,
        tile_state_backend="memmap", # Offload state to disk via memory mapping
        tile_state_dir="/tmp/weathergraph_tiles"
    )
    # --8<-- [end:tiling_config]
