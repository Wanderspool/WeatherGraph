import os
import sys
import numpy as np
import pytest
from weathergraph import WeatherGraphModel

EXTENDED_VALIDATION_ENV = "WEATHERGRAPH_ENABLE_EXTENDED_VALIDATION"

# 10 Real Historical Meteorological Events for Hindcast Validation
EVENTS = [
    {"id": 1, "name": "Hurricane Katrina", "date": "20050823", "target_var": "sp", "desc": "Extreme low pressure tracking"},
    {"id": 2, "name": "European Heatwave", "date": "20030801", "target_var": "t2m", "desc": "Persistent blocking high and heat accumulation"},
    {"id": 3, "name": "Texas Winter Storm", "date": "20210210", "target_var": "t2m", "desc": "Arctic boundary layer plunge"},
    {"id": 4, "name": "Typhoon Haiyan", "date": "20131103", "target_var": "u10", "desc": "Category 5 wind speeds at surface"},
    {"id": 5, "name": "Russian Heatwave", "date": "20100720", "target_var": "t2m", "desc": "Mega-heatwave and atmospheric blocking"},
    {"id": 6, "name": "Storm Eunice", "date": "20220214", "target_var": "u10", "desc": "Explosive cyclogenesis (Sting jet)"},
    {"id": 7, "name": "Bushfire Drought", "date": "20191201", "target_var": "q", "desc": "Extreme specific humidity deficit at 850hPa"},
    {"id": 8, "name": "Hurricane Sandy", "date": "20121022", "target_var": "z", "desc": "Anomalous left-hook path due to blocking"},
    {"id": 9, "name": "Stratospheric Warming", "date": "20180210", "target_var": "t", "desc": "Temperature spike at 50hPa causing polar vortex split"},
    {"id": 10, "name": "Atmospheric River", "date": "20211110", "target_var": "q", "desc": "Intense moisture transport tracking"}
]

@pytest.fixture(scope="module")
def real_environment():
    """
    Ensures that the real ONNX model and real ERA5 data directories exist.
    If they don't, the entire test suite is skipped with a clear explanation.
    """
    model_path = os.getenv("WEATHERGRAPH_ONNX_MODEL", "models/weather_gnn.onnx")
    weights_dir = os.getenv("WEATHERGRAPH_WEIGHTS_DIR", os.getenv("KEISLER_WEIGHTS_DIR", "exporter"))
    data_dir = os.getenv("ERA5_DATA_DIR", "data/era5_archives")

    if not os.path.exists(model_path):
        pytest.skip(f"Real ONNX model missing at {model_path}. Must be exported on an AVX-capable machine.")
        
    if not os.path.exists(data_dir):
        pytest.skip(f"ERA5 data directory missing at {data_dir}. Real NetCDF files required for hindcast validation.")

    return model_path, weights_dir, data_dir

def calculate_rmse(prediction, ground_truth):
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))

def calculate_spectral_energy(field):
    spectrum = np.fft.rfftn(field.astype(np.float32))
    return np.abs(spectrum) ** 2

def q_channel_indices(model):
    q_offset = model.level_vars.index("q")
    channels_per_level = len(model.level_vars)
    return [level_index * channels_per_level + q_offset for level_index in range(len(model.levels))]

def rollout_event(model, data_dir, event, steps):
    import xarray as xr

    init_file = os.path.join(data_dir, f"{event['date']}_init.nc")
    truth_file = os.path.join(data_dir, f"{event['date']}_t10.nc")

    if not os.path.exists(init_file) or not os.path.exists(truth_file):
        pytest.skip(f"Missing real data for {event['name']}. Expected {init_file} and {truth_file}.")

    ds_init = xr.open_dataset(init_file)
    current_state = model._prepare_input(ds_init)

    for _ in range(steps):
        current_state = model.engine.predict(current_state)

    ds_truth = xr.open_dataset(truth_file)
    ground_truth = model._prepare_input(ds_truth)
    return current_state, ground_truth

def test_hindcast_validation_smoke_suite(real_environment):
    """
    Smoke test the 40-step autoregressive hindcast path against real data.

    This suite keeps the required checks cheap and deterministic: shape, finite
    values, and non-degenerate RMSE against ground truth.
    """
    model_path, weights_dir, data_dir = real_environment
    model = WeatherGraphModel(model_path=model_path, weights_dir=weights_dir)
    
    STEPS_10_DAYS = 40 
    
    for event in EVENTS:
        init_file = os.path.join(data_dir, f"{event['date']}_init.nc")
        truth_file = os.path.join(data_dir, f"{event['date']}_t10.nc")

        if not os.path.exists(init_file) or not os.path.exists(truth_file):
            import warnings
            warnings.warn(f"Missing real data for {event['name']}. Expected {init_file} and {truth_file}. Skipping event.")
            continue

        current_state, ground_truth = rollout_event(model, data_dir, event, STEPS_10_DAYS)
        
        rmse = calculate_rmse(current_state, ground_truth)
        
        assert current_state.shape == ground_truth.shape, "Output shape mismatch."
        assert np.isfinite(current_state).all(), "NaN or Inf values detected in output."
        assert rmse > 0.0, "RMSE calculation failed or tensors are identically zero."

@pytest.mark.skipif(
    os.getenv(EXTENDED_VALIDATION_ENV, "0") != "1",
    reason="Set WEATHERGRAPH_ENABLE_EXTENDED_VALIDATION=1 to run expensive spectral and physical checks.",
)
def test_hindcast_extended_validation_suite(real_environment):
    """
    Optional high-cost scientific checks for long autoregressive runs.

    These checks target two known failure modes of weather GNN rollouts:
    spectral smoothing and non-physical negative humidity.
    """
    model_path, weights_dir, data_dir = real_environment
    model = WeatherGraphModel(model_path=model_path, weights_dir=weights_dir)

    current_state, ground_truth = rollout_event(model, data_dir, EVENTS[0], steps=40)

    q_indices = q_channel_indices(model)
    predicted_q = current_state[..., q_indices]
    truth_q = ground_truth[..., q_indices]

    predicted_energy = calculate_spectral_energy(predicted_q)
    truth_energy = calculate_spectral_energy(truth_q)
    energy_ratio = predicted_energy.sum() / np.maximum(truth_energy.sum(), 1e-12)

    assert predicted_q.min() >= -1e-6, "Specific humidity became negative."
    assert 0.1 <= energy_ratio <= 10.0, "Spectral energy drift exceeded the allowed envelope."
