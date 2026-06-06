"""
Examples of streaming forecasts to disk to conserve memory.
"""
import pytest
import xarray as xr
from unittest.mock import patch, MagicMock

@patch('weathergraph.model.WeatherGraphModel')
def test_forecast_export(MockModel):
    model_instance = MockModel.return_value
    dataset = xr.Dataset() # Mocked input
    
    # --8<-- [start:forecast_export]
    from weathergraph.model import WeatherGraphModel
    
    model = WeatherGraphModel(model_path="models/weather_gnn.onnx", weights_dir="data")
    
    # Stream a 14-day forecast directly to a NetCDF file.
    # The `forecast_export` method prevents out-of-memory errors on long rollouts
    # by flushing each step to disk rather than keeping the entire trajectory in RAM.
    model.forecast_export(
        dataset,
        steps=56, # 56 steps * 6 hours = 14 days
        output_format="netcdf4",
        output_path="long_range_forecast.nc"
    )
    # --8<-- [end:forecast_export]

@patch('weathergraph.model.WeatherGraphModel')
def test_iter_forecast(MockModel):
    model_instance = MockModel.return_value
    model_instance.iter_forecast.return_value = [xr.Dataset() for _ in range(5)]
    dataset = xr.Dataset()
    
    # --8<-- [start:iter_forecast]
    from weathergraph.model import WeatherGraphModel
    
    # For custom post-processing pipelines, `iter_forecast` yields steps one by one.
    model = WeatherGraphModel(model_path="models/weather_gnn.onnx", weights_dir="data")
    
    for step_idx, step_ds in enumerate(model.iter_forecast(dataset, steps=20)):
        # Calculate derived metrics (e.g., wind speed)
        # or upload step_ds directly to cloud storage
        pass
    # --8<-- [end:iter_forecast]
