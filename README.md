# Weather Graph

[![CI](https://github.com/keisler-engine/keisler-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/keisler-engine/keisler-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-red.svg)](https://en.cppreference.com/w/cpp/20)

A high-performance, memory-efficient C++ engine for global weather prediction using Graph Neural Networks (GNNs). Optimized for low-resource environments (e.g., **1GB RAM** instances) and large-scale scientific data processing.

## 🚀 Key Features

-   **Zero-Copy Inference:** Direct memory mapping between Python (`xarray`/`numpy`) and the C++ ONNX core using the `pybind11` Buffer Protocol.
-   **Low Memory Footprint:** Specifically engineered to run global weather models on `e2.micro` (1GB RAM) nodes.
-   **In-Graph Normalization:** Pre-processing (Z-score) is baked into the ONNX graph for maximum performance.
-   **Out-of-Core Processing:** Native integration with `Dask` for processing multi-gigabyte ERA5 archives on limited hardware.
-   **Multi-Model Support:** Schema-driven architecture capable of running Keisler 2022, GraphCast, and other GNN architectures via ONNX.

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
git clone https://github.com/keisler-engine/keisler-engine
cd keisler-engine
pip install .
```

### Download Pre-built Binaries
Check the [Releases](https://github.com/keisler-engine/keisler-engine/releases) page for pre-compiled wheels for Linux, macOS, and Windows.

---

## 📊 Quick Start

### Basic Prediction
```python
import xarray as xr
from keisler_engine import GraphWeatherModel

# Initialize engine
model = GraphWeatherModel(
    model_path="models/keisler_2022.onnx",
    weights_dir="data/weights"
)

# Load ERA5 state
ds = xr.open_dataset("initial_state.nc")

# 6-hour forecast
prediction = model.predict_one_step(ds)
```

### Multi-Day Rollout
```python
# 10-day (40 step) auto-regressive forecast
forecast_steps = model.forecast(ds, steps=40)
```

---

## 🧪 Testing & Reliability

We employ a triple-vector validation suite:

1.  **Mathematical Parity:** Verified against original JAX/PyTorch implementations (`atol=1e-5`).
2.  **Memory Safety:** Zero-leak during infinite rollouts (verified via `ASan` and `psutil`).
3.  **Physical Robustness:** Deterministic handling of `NaN`/`Inf` and extreme weather events (Category 5 hurricanes).

Run tests locally:
```bash
pytest tests/
```

---

## 🛠️ Project Structure

```text
├── src/cpp/          # C++ Core (ONNX Runtime, pybind11)
├── keisler_engine/   # Python Package & xarray wrapper
├── exporter/         # Scripts for ONNX graph construction
├── tests/            # Validation suite
├── .github/          # CI/CD Workflows (Binary builds, Testing)
├── CMakeLists.txt    # C++ Build Configuration
└── pyproject.toml    # Python Packaging Metadata
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
