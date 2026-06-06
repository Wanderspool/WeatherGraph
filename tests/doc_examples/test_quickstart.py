"""
Tested example for the Python API Quickstart.
"""
import pytest
import xarray as xr
from unittest.mock import patch, MagicMock

# --8<-- [start:imports]
import weathergraph as wg
import xarray as xr
# --8<-- [end:imports]

@patch('weathergraph.data_sources.ERA5NetCDFAdapter.load')
@patch('weathergraph.WeatherGraphModel')
def test_quickstart_flow(MockModel, mock_load):
    # Setup mocks
    mock_load.return_value = xr.Dataset()
    model_instance = MockModel.return_value
    model_instance.forecast.return_value = xr.Dataset()

    # --8<-- [start:quickstart]
    # 1. Load the initial state from an ERA5 NetCDF archive
    adapter = wg.load_source("era5_netcdf", path="data/era5_archives/init.nc")
    initial_ds = adapter.load()

    # 2. Instantiate the model on CPU (can also be 'cuda' for GPUs)
    model = wg.WeatherGraphModel(
        model_path="models/weather_gnn.onnx",
        weights_dir="data",
        execution_provider="cpu"
    )

    # 3. Run a 10-step (60-hour) autoregressive rollout forecast
    forecast_ds = model.forecast(initial_ds, steps=10, as_dataset=True)
    # --8<-- [end:quickstart]
