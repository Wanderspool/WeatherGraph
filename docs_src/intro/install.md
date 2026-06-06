# Installation & Dependencies

This page guides you through installing WeatherGraph and configuring its system dependencies for both Python-only usage and C++ source compilation.

---

## System Requirements

Before starting, ensure your host environment meets the following baseline requirements:

*   **Operating System**: Linux (CentOS/RHEL 8+, Ubuntu 20.04+, Debian 11+)
*   **Python Version**: Python 3.10, 3.11, or 3.12 (check with `python3 --version`)
*   **C++ Compiler**: A compiler supporting C++17 or newer (e.g., `gcc` 9+, `clang` 10+)
*   **Build Tools**: CMake (version 3.18 or newer) and `make` (or `ninja`)
*   **ONNX Runtime SDK**: WeatherGraph includes a C++ binding to ONNX Runtime. The build system automatically downloads the appropriate ONNX Runtime headers and libraries for CPU targets, but custom hardware setup may require local paths.

---

## Quick Install (Pre-built Wheels)

> [!NOTE]
> Pre-built wheels containing the compiled C++ shared libraries are available for standard Linux x86_64 architectures.

To install the latest stable version of WeatherGraph directly from the package registry:

```bash
pip install weathergraph
```

This installs the core package along with its primary dependencies: `numpy`, `xarray`, `pandas`, `dask`, `netCDF4`, and `zarr`.

---

## Installing Optional Dependencies

WeatherGraph features a modular design. To keep the base installation lightweight, integrations with the wider climate ecosystem and specialized cloud storages are organized as package extras:

| Extra Name | Description | Install Command |
| :--- | :--- | :--- |
| **`vis`** | Interactive mapping (Folium) and animations (Matplotlib, imageio) | `pip install 'weathergraph[vis]'` |
| **`cloud`** | Cloud-hosted Zarr store support (AWS S3 and GCS filesystems) | `pip install 'weathergraph[cloud]'` |
| **`metpy`** | Thermodynamic and coordinate utilities using MetPy | `pip install 'weathergraph[metpy]'` |
| **`xcdat`** | Structured climate analysis and coordinate bounds via xCDAT | `pip install 'weathergraph[xcdat]'` |
| **`climate`** | Full climate stack (MetPy + xCDAT + xESMF) | `pip install 'weathergraph[climate]'` |
| **`test`** | Testing suite dependencies (pytest, memory-profiler, psutil) | `pip install 'weathergraph[test]'` |
| **`docs`** | Documentation builder dependencies (MkDocs, Material, MkDocstrings) | `pip install 'weathergraph[docs]'` |

To install multiple extras at once, separate them with commas:

```bash
pip install 'weathergraph[vis,cloud,metpy]'
```

---

## Installing from Source (C++ Compilation)

If you are developing custom core features, modifying C++ graphs, or compiling for non-x86 architectures, you should build the package from source using the provided `CMakeLists.txt` build-system.

### 1. Install System Build Tools
On Debian/Ubuntu-based systems, run:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake patchelf python3-dev
```

### 2. Clone the Repository
Clone the project repository and navigate into the root directory:

```bash
git clone https://github.com/Wanderspool/WeatherGraph.git
cd WeatherGraph
```

### 3. Build & Install in Editable Mode
We recommend using an editable install (`-e`) in a virtual environment for development. This uses `scikit-build-core` under the hood to invoke CMake and compile the C++ pybind11 extension (`weathergraph_backend`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The build backend compiles the C++ code, builds the `weathergraph_backend.so` shared library, and places it inside the `weathergraph/core/` package path alongside `libonnxruntime.so` so that it is immediately importable.

### 4. GPU Build Configuration (Optional)
By default, the C++ backend compiles for the standard CPU execution provider. To compile with support for the CUDA execution provider:

```bash
CMAKE_ARGS="-DONNXRUNTIME_CUDA=ON" pip install -e .
```

For more detailed information on GPU acceleration, see the [Execution Providers](../advanced/execution-providers.md) section.

---

## Verifying the Installation

To verify that WeatherGraph is correctly installed and that the C++ inference engine is functional, run the following verification commands:

### CLI Verification
Run the version check command via the command line interface:

```bash
weathergraph --version
```
This should output the current version of the engine (e.g., `0.1.0`).

### Python Verification
Launch a Python session and verify that the package, the C++ backend, and the xarray accessor load without errors:

```python
import weathergraph as wg
import xarray as xr

# 1. Print the version
print(f"WeatherGraph version: {wg.__version__}")

# 2. Check the C++ backend library
from weathergraph.model import WeatherGraphModel
print("C++ Backend Engine is successfully linked!")

# 3. Verify the xarray accessor registration
ds = xr.Dataset()
if hasattr(ds, "weathergraph"):
    print("Xarray accessor is active and registered!")
```

If these steps complete without raising an `ImportError` or `ModuleNotFoundError`, your WeatherGraph installation is complete and ready for use.