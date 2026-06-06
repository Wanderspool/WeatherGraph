"""
Examples of using Data Adapters in WeatherGraph.
"""
import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch, MagicMock

# --8<-- [start:imports]
from weathergraph.data_sources import load_source
from weathergraph.model import WeatherGraphModel
# --8<-- [end:imports]

@patch('weathergraph.data_sources.ERA5NetCDFAdapter.load')
def test_era5_adapter(mock_load):
    # Setup mock to return a mock dataset
    mock_load.return_value = xr.Dataset()

    # --8<-- [start:era5_adapter]
    # 1. Initialize the adapter
    era5_source = load_source('era5_netcdf', path="data/era5_archives/init_state.nc")

    # 2. Fetch or load the dataset
    dataset = era5_source.load()

    # 3. Create the model and run the forecast
    # model = WeatherGraphModel(model_path="models/weather_gnn.onnx", weights_dir="data")
    # forecast_ds = model.forecast(dataset, steps=10)
    # --8<-- [end:era5_adapter]

@patch('weathergraph.data_sources.CustomAdapter.load')
def test_custom_adapter(mock_load):
    # Setup mock to return a mock dataset
    mock_load.return_value = xr.Dataset()

    # --8<-- [start:custom_adapter]
    # To map custom variable names, use the 'custom' adapter
    custom_source = load_source(
        'custom',
        path="data/my_custom_run.nc",
        variable_map={
            'z': 'geopotential',
            'q': 'specific_humidity',
            't': 'temperature',
            'u': 'u_component_of_wind',
            'v': 'v_component_of_wind',
            'w': 'vertical_velocity'
        }
    )
    
    dataset = custom_source.load()
    # --8<-- [end:custom_adapter]


