# WeatherGraph

[![CI](https://github.com/Wanderspool/WeatherGraph/actions/workflows/ci.yml/badge.svg)](https://github.com/Wanderspool/WeatherGraph/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-red.svg)](https://en.cppreference.com/w/cpp/20)

A high-performance C++ engine for global weather prediction using Graph Neural Networks (GNNs). The current reference artifact is tuned for memory-conscious CPU inference, but realistic high-resolution deployments should be sized as workstation-class jobs rather than 1 GB micro-instance workloads.

## 🚀 Key Features

-   **Zero-Copy Inference:** Direct memory mapping between Python (`xarray`/`numpy`) and the C++ ONNX core using the `pybind11` Buffer Protocol.
-   **Memory-Aware Runtime:** Designed to minimize copies across Python/C++ boundaries and keep CPU inference practical on constrained machines for coarse reference models.
-   **Optional Low-Memory ORT Mode:** `disable_cpu_mem_arena` and `disable_mem_pattern` can be enabled in any supported pipeline when lower reserved RSS matters more than peak throughput.
-   **In-Graph Normalization:** Pre-processing (Z-score) is baked into the ONNX graph for maximum performance.
-   **Out-of-Core Processing:** Native integration with `Dask` for processing multi-gigabyte ERA5 archives on limited hardware.
-   **Exact Spatial Tiling Contract:** `spatial_tiling` is available as an exact graph-aware mode via tile bundles with explicit partition metadata and per-tile ONNX artifacts.
-   **Multi-Model Support:** Schema-driven architecture capable of running the reference 2022 model, GraphCast, and other GNN architectures via ONNX.

---

## 🏗️ Architecture

The engine is split into two distinct layers:

1.  **Data Plane (C++)**: Handles performance-critical operations, memory management (RAII), and ONNX Runtime execution.
2.  **Control Plane (Python)**: Provides a scientific API, handles `xarray` data orchestration, and manages autoregressive rollout logic.

---

## 💻 Installation

### From Source
The project uses `scikit-build-core` for seamless Python/C++ integration.

```bash
git clone https://github.com/Wanderspool/WeatherGraph
cd WeatherGraph
pip install .
```

### Download Pre-built Binaries
Check the [Releases](https://github.com/Wanderspool/WeatherGraph/releases) page for pre-compiled wheels for Linux, macOS, and Windows.

---

## 📊 Quick Start

## ✅ Implementation Status

Implemented in the current codebase:

- Optional ONNX Runtime low-memory mode in the C++ backend and Python wrapper.
- Runtime propagation of `intra_op_threads`, `disable_cpu_mem_arena`, and `disable_mem_pattern` through supported pipelines.
- Exact graph-aware spatial tiling contract via `spatial_tiling=True` and `tile_bundle_path=...`.
- Streaming rollout/export paths through `iter_forecast()` and `forecast_export()` for lower-memory multi-step runs.
- Pipeline wiring for notebook examples, reusable GitHub Actions, Ansible, Terraform/cloud-init, GCP Batch, AWS Batch, and the simulation runner.

Still intentionally out of scope in the current tree:

- Automatic generation of tile bundles from an arbitrary global ONNX artifact.
- Approximate node slicing. Tiling remains exact-only and fails fast without bundle metadata.

### Basic Prediction
```python
import xarray as xr
from weathergraph import WeatherGraphModel

# Initialize engine
model = WeatherGraphModel(
    model_path="models/weather_gnn.onnx",
    weights_dir="data",
    intra_op_threads=2,
    disable_cpu_mem_arena=False,
    disable_mem_pattern=False,
)

# Load ERA5 state
ds = xr.open_dataset("initial_state.nc")

# 6-hour forecast
prediction = model.predict_one_step(ds)
```

Optional low-memory runtime knobs:

- `disable_cpu_mem_arena=True` disables ONNX Runtime's CPU arena allocator and reduces reserved RSS.
- `disable_mem_pattern=True` disables memory-pattern reuse and can reduce static reservation further.
- Both are `False` by default because they trade memory headroom for slower inference.

Trade-off summary:

- Lower reserved RSS and fewer false OOMs on memory-constrained machines.
- Higher allocation overhead and typically lower throughput.
- Possible fragmentation risk on very long autoregressive runs when allocator reuse is disabled.

## 🖥️ Deployment Profiles

WeatherGraph currently has two practical operating profiles:

-   **Experimental 1° reference profile:** Useful for exporter validation, backend smoke tests, and coarse global rollouts. This is the only profile where sub-2 GB memory targets are even remotely plausible, and even then only for single-step or very short rollouts.
-   **Recommended 0.25° workstation profile:** Treat this as the default target for scientifically useful global inference. A single `float32[1, 1038240, 78]` state is already about 309 MiB, so CPU inference should be planned around at least **16 GB RAM for single-step / short-rollout work** and **32-64 GB RAM for 40-step rollouts or full-trajectory export**, depending on ONNX Runtime workspace overhead and how aggressively outputs are buffered.

`forecast()` still materializes the full trajectory in memory. For long high-resolution runs, prefer `iter_forecast()` or `forecast_export()` because they stream step-by-step and keep the working set lower than the list-based rollout path.
Allocator-disabled mode and exact tiling are both optional. Use them when RAM is the limiting resource and extra latency is acceptable.

## 🧩 Exact Tiling

WeatherGraph also supports optional exact graph-aware spatial tiling through tile bundles:

```python
model = WeatherGraphModel(
    model_path="models/weather_gnn.onnx",
    weights_dir="data",
    disable_cpu_mem_arena=True,
    disable_mem_pattern=True,
    spatial_tiling=True,
    tile_bundle_path="tile_bundle/manifest.json",
)
```

This is not naive node slicing. A tile bundle must include explicit partition metadata and per-tile ONNX artifacts so halo nodes and ownership boundaries are handled exactly. If `spatial_tiling=True` is requested without a tile bundle, the model fails fast by design.

Current status:

- The runtime and pipelines already accept exact tiling parameters.
- Bundle generation is not automated yet; users must provide a prepared tile bundle.

### Multi-Day Rollout
```python
# 10-day (40 step) auto-regressive forecast
forecast_steps = model.forecast(ds, steps=40)
```

---

## 🧪 Testing & Reliability

We employ a triple-vector validation suite:

1.  **Mathematical Parity:** Verified against original JAX/PyTorch implementations (`atol=1e-5`).
2.  **Runtime Stability:** No unbounded memory growth during repeated inference, with resource checks reported per deployment profile rather than a single fixed RAM promise.
3.  **Physical Robustness:** Deterministic handling of `NaN`/`Inf`, plus scenario-driven hindcast checks for long autoregressive runs.

Run tests locally:
```bash
pytest tests/
```

---

## 🛠️ Project Structure

```text
├── src/cpp/          # C++ Core (ONNX Runtime, pybind11)
├── weathergraph/     # Python package & xarray wrapper
├── exporter/         # Scripts for ONNX graph construction
├── tests/            # Validation suite
├── .github/          # CI/CD Workflows (Binary builds, Testing)
├── CMakeLists.txt    # C++ Build Configuration
└── pyproject.toml    # Python Packaging Metadata
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
