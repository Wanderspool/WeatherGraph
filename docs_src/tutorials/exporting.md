# Exporting to NetCDF/Zarr

WeatherGraph forecasts are structured, multi-dimensional scientific datasets. This tutorial covers how to export forecasts into standard scientific formats, detailing both in-memory serialization and disk-streaming methods for handling large-scale forecasts.

---

## 1. Materializing in Memory (xarray.Dataset)

For small-scale grids (e.g. 1° resolution) or short forecast horizons (e.g. 24–48 hours), you can load the entire forecast trajectory into RAM as an `xarray.Dataset` and export it using xarray's built-in writers.

### NetCDF Export
```python
import weathergraph as wg
import xarray as xr

# Load data and model
initial_ds = wg.load_source("era5_netcdf", path="data/era5_archives/init.nc").load()
model = wg.WeatherGraphModel(model_path="models/weather_gnn.onnx", weights_dir="data")

# Run forecast and return dataset
forecast_ds = model.forecast(initial_ds, steps=8, as_dataset=True)

# Write to a single NetCDF file
forecast_ds.to_netcdf("output/complete_trajectory.nc")
```

### Zarr Export
Zarr is highly recommended for cloud-native workflows or datasets requiring chunked, compressed storage:

```python
# Save dataset as a local Zarr store
forecast_ds.to_zarr("output/complete_trajectory.zarr", mode="w")
```

---

## 2. Streaming to Disk (Memory-Efficient)

> [!WARNING]
> High-resolution forecasts (e.g., 0.1° resolution grids with over 6 million grid cells per layer) can quickly exhaust system memory if the entire trajectory is materialized in RAM. For a 14-day rollout, the array size will exceed dozens of gigabytes.

To prevent Out-Of-Memory (OOM) failures, WeatherGraph supports streaming forecast steps directly to disk as they are produced, keeping only the active step in RAM.

### Direct Streaming Export (`forecast_export`)
The `forecast_export()` method runs the autoregressive loop in a streaming fashion. It splits the output by variable and vertical level, writing each time slice step-by-step:

```python
--8<-- "tests/doc_examples/test_export.py:forecast_export"
```

#### Output Structure
When running `forecast_export` with format `"netcdf4"`, the engine creates a directory structure containing one file per variable-level combination:

```text
long_range_forecast.nc/
  ├── z_50hPa.nc
  ├── z_100hPa.nc
  ├── ...
  ├── t_50hPa.nc
  ├── t_100hPa.nc
  └── ...
```
Each file has coordinates `(time, lat, lon)` and contains only the timeseries data for that specific level, facilitating rapid subsetting during downstream analysis.

---

## 3. Custom Iterative Processing (`iter_forecast`)

If you want to perform post-processing on each forecast step (e.g., computing derived variables like wind speed) or push steps directly to a remote cloud bucket (e.g., AWS S3 or Google Cloud Storage) without writing local files first, use `iter_forecast()`.

`iter_forecast()` is a Python generator that yields each step's raw output tensor as it is computed:

```python
--8<-- "tests/doc_examples/test_export.py:iter_forecast"
```

Using this approach, only the current step's memory is allocated, making it ideal for running pipeline rollouts of arbitrary lengths.