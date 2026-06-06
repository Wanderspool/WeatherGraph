# Quickstart (Python API)

For scientific researchers and developers, the Python API provides a flexible, programmatically rich interface to integrate WeatherGraph's neural forecasting engine directly into your data pipelines.

This quickstart guides you through running a basic multi-step autoregressive rollout forecast in Python.

---

## The Complete Quickstart Script

Below is the standard workflow to fetch an initial state, instantiate the model, and run a forecast. This script is fully verified in the test suite:

```python
--8<-- "tests/doc_examples/test_quickstart.py:imports"

--8<-- "tests/doc_examples/test_quickstart.py:quickstart"
```

---

## Step-by-Step Breakdown

Let's look at each step of the quickstart process in detail.

### 1. Ingesting Initial Conditions
WeatherGraph expects initial conditions containing variables in a specific layout. The `load_source()` factory is the easiest way to retrieve and format datasets correctly:

```python
adapter = wg.load_source("era5_netcdf", path="data/era5_archives/init.nc")
initial_ds = adapter.load()
```
The resulting `initial_ds` is a standard `xarray.Dataset` containing the expected channels: geopotential (`z`), specific humidity (`q`), temperature (`t`), $u$ and $v$ components of wind (`u`, `v`), and vertical velocity (`w`) mapped across 13 pressure levels (from 50 hPa to 1000 hPa).

### 2. Loading the Model Session
Instantiate the `WeatherGraphModel` by pointing it to the ONNX graph file and the directory containing normalization weights:

```python
model = wg.WeatherGraphModel(
    model_path="models/weather_gnn.onnx",
    weights_dir="data",
    execution_provider="cpu"
)
```
- `model_path`: Path to the exported GNN ONNX graph.
- `weights_dir`: Path to the directory containing mean/std arrays (`means.npy` and `stds.npy`) used to normalize inputs and denormalize outputs.
- `execution_provider`: Choose where to place computation graph nodes (`"cpu"`, `"cuda"`, `"tensorrt"`, `"rocm"`, or `"openvino"`).

### 3. Running Autoregressive Rollout
WeatherGraph is an autoregressive model. Each step predicts the state 6 hours into the future, which is then fed back into the model as input for the next step.

The `forecast()` method manages this loop for you:

```python
forecast_ds = model.forecast(initial_ds, steps=10, as_dataset=True)
```
- `steps=10`: Run 10 autoregressive steps (10 × 6 hours = 60 hours forecast horizon).
- `as_dataset=True`: Automatically shape and denormalize the raw model output tensor (`[1, nodes, 78]`) into a CF-compliant `xarray.Dataset` with coordinate dimensions `(time, level, lat, lon)`.

---

## Inspecting the Forecast Dataset

Because the output is a standard `xarray.Dataset`, you can inspect and manipulate it using standard scientific Python tools:

```python
# 1. Print a summary of the dataset
print(forecast_ds)

# 2. Extract a specific variable at a specific level and time
temp_850hpa = forecast_ds["t"].sel(level=850, time=forecast_ds.time[4])
print("Mean temperature at 850 hPa (24h forecast):", float(temp_850hpa.mean()))

# 3. Export the entire dataset to a local NetCDF file
forecast_ds.to_netcdf("my_forecast.nc")
```

---

## Next Steps

To explore more advanced features, check out:
- **[Your First Forecast](../tutorials/first-forecast.md)**: A detailed tutorial extending this quickstart.
- **[Working with Data Sources](../tutorials/data-sources.md)**: How to fetch real-time forecasts from ECMWF and GFS APIs.
- **[Output Formats & Exporting](../guide/exporting.md)**: Rollout long forecasts without running out of RAM.