# WeatherGraph

[![Website](https://img.shields.io/badge/website-wanderspool.github.io%2FWeatherGraph-blue?logo=github)](https://wanderspool.github.io/WeatherGraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-red.svg)](https://en.cppreference.com/w/cpp/20)

**WeatherGraph** is an open-source engine for global numerical weather prediction based on Graph Neural Networks (GNN). It combines a high-performance C++ inference core with a flexible Python scientific API, making modern AI-driven weather forecasting accessible to researchers and operational meteorologists on commonly available hardware — from laptops to cloud workstations.

Accurate weather forecasting remains one of the most impactful applications of computational science. Traditional numerical weather prediction (NWP) systems require supercomputer-scale resources, creating a barrier for regional meteorological services, academic research groups, and developing nations. WeatherGraph addresses this gap by packaging a production-ready GNN inference engine that runs on hardware already available in research labs and field offices, while delivering forecast quality competitive with far more expensive systems.

---

## 🌍 Motivation & Impact

- **Democratizing forecasting.** Graph-neural-network weather models can match or exceed conventional NWP accuracy at a fraction of the computational cost. WeatherGraph brings this capability to any researcher with a modern laptop or workstation.
- **Multi-source data ingestion.** Built-in adapters connect to ERA5, ECMWF Open Data, Copernicus CDS, NOAA GFS, and Open-Meteo, so initial conditions can come from the data source most appropriate for the region and use case.
- **Scalable resolution.** A single codebase supports coarse 1° exploratory runs, operational 0.25° global forecasts, and experimental 0.1° high-resolution tiled inference — all through runtime configuration, not code changes.
- **Reproducible science.** Deterministic inference with mathematical parity verification (atol = 10⁻⁵ against reference implementations) ensures that results are scientifically reproducible across platforms.

---

## 🚀 Key Features

- **Zero-Copy Inference.** Direct memory mapping between Python (`xarray`/`numpy`) and the C++ ONNX core via the `pybind11` Buffer Protocol — no redundant data copies on the critical path.
- **Memory-Aware Runtime.** Optional low-memory ONNX Runtime modes (`disable_cpu_mem_arena`, `disable_mem_pattern`) allow inference on RAM-constrained devices without code changes.
- **Multi-Accelerator Support.** `execution_provider` supports `cuda`, `tensorrt`, `rocm`, and `openvino` when matching ONNX Runtime execution-provider libraries are available.
- **In-Graph Normalization.** Z-score pre-processing is baked into the ONNX graph, eliminating a separate normalization step and reducing end-to-end latency.
- **Out-of-Core Processing.** Native Dask integration for processing multi-gigabyte ERA5 archives on limited hardware with controlled memory footprint.
- **Exact Spatial Tiling.** Graph-aware tiling via tile bundles with explicit partition metadata enables high-resolution inference (0.1°) on hardware that cannot fit the full global graph in memory.
- **Streaming Export.** `iter_forecast()` and `forecast_export()` stream results step-by-step to NetCDF4, Zarr, or NPZ, keeping working-set memory low during long multi-day rollouts.
- **Multi-Model Architecture.** Schema-driven design supports the reference Keisler 2022 model, GraphCast derivatives, and other GNN architectures via standard ONNX.
- **Data Sanitization & Hard Constraints.** Real-time NaN/Inf sanitization protects graph integrity, and a secondary zero-copy ONNX pipeline allows applying explicit physical constraints (e.g., non-negative humidity).
- **Halo Exchange for Tiled Inference.** Smooth edge artifacts during high-resolution tiled inference using weighted spatial aggregation (Halo Exchange).
- **Probabilistic Core (Ensembles).** Built-in O(1)-memory ensemble inference. Run dozens of parallel scenarios with per-channel noise, calculating variance and threshold probabilities on the fly in C++ without exploding RAM requirements.

---

## 🏗️ Architecture

The engine follows a two-layer design that separates performance-critical inference from scientific orchestration:

```
┌──────────────────────────────────────────────────────────┐
│  Control Plane (Python)                                  │
│  xarray · data adapters · rollout logic · export         │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Data Plane (C++)                                  │  │
│  │  ONNX Runtime · zero-copy tensors · RAII memory    │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

1. **Data Plane (C++)** — Manages the ONNX Runtime session, executes inference with RAII-based deterministic memory management, validates input buffer contiguity, and handles execution-provider configuration.
2. **Control Plane (Python)** — Provides the scientific API (`WeatherGraphModel`), manages data ingestion through pluggable adapters, orchestrates autoregressive rollout, and handles structured export to standard climate data formats.

---

## 📈 Performance Estimates

The tables below present estimated inference times for a 10-step (60-hour) autoregressive forecast at three target resolutions. These figures are based on preliminary internal benchmarks; work is currently underway to improve engine stability and reproducibility at the 0.1° resolution tier across a broader range of consumer and workstation hardware.

### GPU Inference (Discrete Laptop GPUs)

| Resolution | Grid Size | NVIDIA RTX 3060 (6 GB) | NVIDIA RTX 4060 (8 GB) | NVIDIA RTX 4070 (8 GB) |
| :--- | :--- | :--- | :--- | :--- |
| 1.0° | 181 × 360 | ~4 s | ~2.5 s | ~2 s |
| 0.25° | 721 × 1440 | ~45 s | ~28 s | ~22 s |
| 0.1° | 1801 × 3600 | ~18 min *(tiled)* | ~12 min *(tiled)* | ~9 min *(tiled)* |

### CPU Inference (Laptop/Workstation Processors)

| Resolution | Grid Size | Intel Core i7-12700H (14C) | AMD Ryzen 7 7840HS (8C) | Apple M2 Pro (12C) |
| :--- | :--- | :--- | :--- | :--- |
| 1.0° | 181 × 360 | ~18 s | ~22 s | ~15 s |
| 0.25° | 721 × 1440 | ~8 min | ~10 min | ~6 min |
| 0.1° | 1801 × 3600 | ~55 min *(tiled)* | ~70 min *(tiled)* | ~45 min *(tiled)* |

> **Note on estimates.** These values were obtained during development testing and should be treated as indicative rather than certified benchmarks. Actual performance depends on model variant, thermal headroom, background system load, and ONNX Runtime version. Tiled inference (`spatial_tiling=True`) is required at 0.1° resolution and incurs additional overhead from tile stitching.

### Current Development Focus

The primary engineering effort is currently directed at **improving inference stability and memory efficiency at 0.1° (≈11 km) resolution** to make this tier reliable on a wide range of consumer and workstation hardware released within the last five years. The goal is to make high-resolution neural weather prediction practical without specialized HPC infrastructure.

---

## 💻 Installation

### From Source

The project uses `scikit-build-core` for seamless Python/C++ integration.

```bash
git clone https://github.com/Wanderspool/WeatherGraph
cd WeatherGraph
pip install .
```

For accelerator-backed inference, provide an ONNX Runtime SDK whose `onnxruntime-sdk/lib/` contains `libonnxruntime.so` plus the matching execution-provider libraries for your GPU.

### Pre-built Wheels

Check the [Releases](https://github.com/Wanderspool/WeatherGraph/releases) page for pre-compiled wheels for Linux, macOS, and Windows.

---

## 📊 Quick Start

### Python API

```python
import xarray as xr
from weathergraph import WeatherGraphModel

# Initialize the engine
model = WeatherGraphModel(
    model_path="models/weather_gnn.onnx",
    weights_dir="data",
    intra_op_threads=4,
)

# Load ERA5 initial conditions
ds = xr.open_dataset("initial_state.nc")

# Single 6-hour forecast step
prediction = model.predict_one_step(ds)

# 10-day (40 step) autoregressive rollout
forecast = model.forecast(ds, steps=40)
```

### GPU Acceleration

```python
model = WeatherGraphModel(
    model_path="models/weather_gnn.onnx",
    weights_dir="data",
    execution_provider="cuda",
    execution_device_id=0,
    disable_cpu_ep_fallback=True,
)
```

### Ensemble Inference (Probabilistic Core)

```python
stats = model.predict_ensemble(
    initial_ds=ds,
    steps=40,
    members=50,
    perturbation_scale={"t": 0.5, "u": 0.3, "v": 0.3},
    thresholds={"frost": "t@850 < 273.15"},
    aggregate_steps=[9, 19, 29, 39], # Only return output for these steps to save memory
    seed=42,
)

# Returns O(1)-memory aggregated statistics
stats.mean                          # xr.Dataset
stats.std_dev                       # xr.Dataset
stats.probabilities["frost"]        # xr.DataArray: P(T < 273.15)
```

### High-Resolution Tiled Inference (0.1°)

```python
model = WeatherGraphModel(
    model_path="models/weather_gnn.onnx",
    weights_dir="data",
    spatial_tiling=True,
    tile_bundle_path="tile_bundle/manifest.json",
    reference_grid_resolution_degrees=0.1,
    tile_state_backend="memmap",
    tile_state_dir="/fast-scratch/weathergraph",
)
```

### Command-Line Interface

```bash
# List available data sources
weathergraph list-sources

# Inspect model metadata
weathergraph inspect \
    --model-path models/weather_gnn.onnx \
    --weights-dir data

# Run a deterministic forecast
weathergraph forecast \
    --model-path models/weather_gnn.onnx \
    --weights-dir data \
    --data-source era5_netcdf \
    --input-path initial_state.nc \
    --steps 40 \
    --output-format zarr \
    --output-path forecast_output

# Run an ensemble forecast
weathergraph ensemble \
    --model-path models/weather_gnn.onnx \
    --weights-dir data \
    --data-source era5_netcdf \
    --input-path initial_state.nc \
    --steps 40 \
    --members 50 \
    --perturbation-scale '{"t": 0.5, "q": 0.001}' \
    --threshold "frost=t@850<273.15" \
    --output-format netcdf4 \
    --output-path ensemble_output

# Visualize results
weathergraph visualize \
    --input forecast_out.nc \
    --variable t \
    --format html \
    --output interactive_map.html
```

---

## 🌐 Data Sources

WeatherGraph includes adapters for the most widely used atmospheric data providers, enabling researchers to run forecasts from whichever data source best fits their region and use case:

| Source | Description | Authentication |
| :--- | :--- | :--- |
| `era5_netcdf` | Local ERA5 reanalysis (NetCDF) | None |
| `ecmwf_open` | ECMWF Open Data — real-time global forecast | None |
| `cds_era5` | Copernicus CDS — ERA5 reanalysis via API | Free registration |
| `gfs` | NOAA GFS — global forecast via AWS Open Data | None |
| `open_meteo` | Open-Meteo — multi-model NWP aggregator | None |
| `zarr` | Zarr store — local or cloud (GCS/S3/Azure) | None |
| `custom` | Custom files with configurable variable mapping | None |

---

## 🔬 Climate Integration

WeatherGraph embeds seamlessly into the scientific Python ecosystem through a custom Xarray accessor, CF-1.11 convention compliance, and built-in interoperability with MetPy and xCDAT.

### Xarray Accessor

After `import weathergraph`, every `xr.Dataset` gains a `.weathergraph` namespace:

```python
import xarray as xr
import weathergraph  # registers the accessor

# Load initial conditions from a cloud Zarr store
ds = xr.open_zarr("gs://weatherbench2/datasets/era5/2024-01-01.zarr")

# Run a 10-day forecast — all C++ inference and tiling is hidden
ds_forecast = ds.weathergraph.predict(steps=40)

# Save CF-compliant results to Zarr
ds_forecast.to_zarr("s3://my-bucket/forecast.zarr")
```

### CF-1.11 Conventions

All forecast output carries standard CF metadata (`standard_name`, `units`, `Conventions`) enabling automatic interoperability with downstream libraries:

```python
# Every variable has CF attributes
ds_forecast["t"].attrs
# {'standard_name': 'air_temperature', 'units': 'K', 'long_name': 'Air Temperature'}

ds_forecast.attrs["Conventions"]
# 'CF-1.11'
```

### MetPy Integration

Prepare forecast data for operational meteorology analysis:

```python
from weathergraph.integrations import prepare_for_metpy
import metpy.calc as mpcalc

ds_metpy = prepare_for_metpy(ds_forecast)  # sorts pressure levels, adds CRS
u_geo, v_geo = mpcalc.geostrophic_wind(ds_metpy["z"].sel(level=500).metpy.quantify())
```

### xCDAT Integration

Compute area-weighted spatial averages and climate anomalies:

```python
from weathergraph.integrations import prepare_for_xcdat

ds_xcdat = prepare_for_xcdat(ds_forecast)  # adds spatial/temporal bounds
global_mean_t = ds_xcdat.spatial.average("t")
anomalies = ds_xcdat.temporal.departures("t", freq="month")
```

### Derived Diagnostics

```python
from weathergraph.integrations import compute_derived_diagnostics

ds_diag = compute_derived_diagnostics(ds_forecast)
# Adds: wind_speed (from u, v), geopotential_height (from z)
```

### Install Optional Dependencies

```bash
pip install 'weathergraph[metpy]'     # MetPy only
pip install 'weathergraph[xcdat]'     # xCDAT only
pip install 'weathergraph[climate]'   # MetPy + xCDAT + xESMF
pip install 'weathergraph[cloud]'     # S3/GCS Zarr access
```

---

## 🖥️ Deployment Profiles

| Profile | Resolution | Minimum RAM | Use Case |
| :--- | :--- | :--- | :--- |
| **Exploratory** | 1.0° (181 × 360) | 2 GB | Model validation, rapid prototyping, educational use |
| **Operational** | 0.25° (721 × 1440) | 16–64 GB | Scientifically useful global forecasts, operational meteorology |
| **High-Resolution** | 0.1° (1801 × 3600) | 8+ GB *(tiled)* | Regional high-detail forecasting, severe weather analysis |

For long autoregressive rollouts (40+ steps), prefer `iter_forecast()` or `forecast_export()` which stream results step-by-step to disk instead of materializing the full trajectory in memory.

---

## 🧩 Exact Spatial Tiling

WeatherGraph implements an exact graph-aware tiling system for high-resolution inference. Unlike naive spatial slicing, the tiling contract uses explicit partition metadata with properly handled halo nodes and ownership boundaries, ensuring mathematically exact results.

```bash
# Build a tile bundle for 0.1° resolution
weathergraph build-tile-bundle \
    --output-dir tile_bundle \
    --senders-path data/graph_data/senders_receivers_encoder/senders.npy \
    --receivers-path data/graph_data/senders_receivers_encoder/receivers.npy \
    --tile-model-dir tile_models \
    --reference-grid-resolution-degrees 0.1 \
    --tile-grid-shape 150x150 \
    --halo-hops 1
```

Tiled inference can optionally use memory-mapped state buffers (`tile_state_backend="memmap"`) to further reduce RAM requirements, making 0.1° global runs feasible on machines with as little as 8 GB of RAM.

---

## 🧪 Validation & Testing

The project employs a multi-vector validation strategy to ensure both scientific accuracy and operational reliability:

1. **Mathematical Parity.** Verified against reference JAX/PyTorch implementations with `atol=1e-5`, ensuring zero degradation from the original research models.
2. **Runtime Stability.** Memory behavior is monitored across extended autoregressive runs (10,000+ steps) to verify absence of unbounded growth, with checks tailored to each deployment profile.
3. **Physical Robustness.** Deterministic handling of NaN/Inf values, scenario-driven hindcast validation against historical extreme events (e.g., Hurricane Katrina 2005), and impulse-response verification of graph topology.

```bash
pytest tests/
```

---

## ✅ Implementation Status

### Completed

- C++ ONNX Runtime backend with zero-copy Python bindings
- Multi-provider execution (CPU, CUDA, TensorRT, ROCm, OpenVINO)
- Optional low-memory ONNX Runtime controls
- Exact graph-aware spatial tiling with tile-bundle packaging
- Streaming autoregressive rollout and export (NetCDF4, Zarr, NPZ)
- Configurable reference-grid export metadata
- Seven built-in data-source adapters (ERA5, ECMWF, CDS, GFS, Open-Meteo, Zarr, Custom)
- Researcher-facing CLI with forecast, inspect, visualize, and bundle-build commands
- Pipeline integrations for Ansible, Terraform, GCP Batch, and AWS Batch
- CF-1.11 convention compliance on all forecast output
- Custom Xarray accessor (`ds.weathergraph.predict()`) for climatologists
- MetPy and xCDAT integration utilities with graceful fallback
- Derived atmospheric diagnostics (wind speed, geopotential height)
- Cloud-native Zarr store adapter (GCS, S3, Azure)
- **(NEW)** Hard Constraints via secondary zero-copy ONNX graphs
- **(NEW)** Real-time Data Sanitization neutralizing NaN/Inf corruption
- **(NEW)** Safe C++ exception translation to Python to prevent SIGSEGV crashes
- **(NEW)** `mimalloc` support for mitigating memory fragmentation
- **(NEW)** Halo Exchange for artifact-free spatial tiled inference
- **(NEW)** Probabilistic Core for O(1)-memory ensemble inference and variance calculation

### In Progress

- Stabilizing 0.1° resolution inference across a wider range of consumer hardware
- Published benchmark suite with reproducible performance profiles
- Automatic tile-bundle generation from arbitrary global ONNX artifacts

---

## 🛠️ Project Structure

```text
├── src/cpp/          # C++ ONNX Runtime backend (pybind11 bindings)
├── weathergraph/     # Python package: model API, CLI, data adapters, visualization
├── exporter/         # ONNX graph construction and tile-bundle tools
├── models/           # Reference ONNX model artifacts
├── data/             # Normalization statistics and graph topology data
├── tests/            # Validation suite (accuracy, stability, memory, historical)
├── examples/         # Notebooks, simulation scripts, and deployment playbooks
├── Docs/             # Architecture and feature documentation
├── CMakeLists.txt    # C++ build configuration
└── pyproject.toml    # Python packaging (scikit-build-core)
```

---

## 📚 Documentation

Detailed technical documentation is available in the `Docs/` directory:

- [**Project Architecture**](Docs/project-architecture.md) — Data plane / control plane design, build system, extension points
- [**Feature Guide**](Docs/feature-guide.md) — Runtime feature reference with usage guidance and trade-offs
- [**Pipelines Guide**](Docs/examples-pipelines-guide.md) — Deployment playbooks and integration examples

---

## 🤝 Contributing

Contributions are welcome. Please ensure that:

- All C++ code passes AddressSanitizer and MemorySanitizer checks.
- All optimizations maintain mathematical parity within 10⁻⁵ tolerance.
- New features include corresponding test coverage in `tests/`.
- Python code is type-hinted; C++ code is documented with Doxygen-style comments.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
