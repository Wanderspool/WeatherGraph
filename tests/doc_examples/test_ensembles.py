"""
Examples of generating Probabilistic Ensembles in WeatherGraph.
"""
import pytest
import xarray as xr
import numpy as np
from unittest.mock import patch, MagicMock

@patch('weathergraph.model.WeatherGraphModel')
def test_probabilistic_ensemble(MockModel):
    # Setup mock
    model_instance = MockModel.return_value
    # Mocking predict_ensemble to return a dataset
    model_instance.predict_ensemble.return_value = xr.Dataset()

    # --8<-- [start:ensemble_prediction]
    from weathergraph.model import WeatherGraphModel
    
    # 1. Initialize the model
    model = WeatherGraphModel(
        model_path="models/weather_gnn.onnx",
        weights_dir="data",
        execution_provider="cuda" # Ensembles benefit greatly from GPU
    )

    # Assume `dataset` is our initial atmospheric state
    dataset = xr.Dataset() # (Mocked for this example)

    # 2. Run ensemble prediction
    # This runs 50 members to 10 steps. Welford's algorithm aggregates the
    # mean and variance without blowing up memory.
    ensemble_stats_ds = model.predict_ensemble(
        dataset,
        members=50,
        steps=10,
        noise_scale=0.05, # Per-channel Gaussian noise scale
        aggregate_steps=[5, 10] # Only return statistics at these specific time steps
    )
    # --8<-- [end:ensemble_prediction]
