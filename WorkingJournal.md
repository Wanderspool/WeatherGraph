# High-Performance Weather GNN Engine (C++/ONNX)

This project is a high-performance C++ library/engine for weather prediction using Graph Neural Networks (GNNs). It is optimized for memory-aware CPU inference, but the public baseline should be a workstation-class deployment for 0.25° global models rather than a blanket 1 GB micro-instance claim.

## Core Vision & Mandates

- **Memory-Aware Efficiency:** Minimize copies and keep coarse reference artifacts usable on constrained CPUs while sizing high-resolution runs against realistic workstation RAM.
- **Zero-Copy Performance:** Uses `pybind11` Buffer Protocol for direct memory access between Python (numpy/xarray) and the C++ engine.
- **Optional Low-Memory Runtime:** Allow disabling the ONNX Runtime CPU arena allocator and memory-pattern reuse when RSS is the limiting resource.
- **Exact Tiled Execution:** Support graph-aware tiling only through explicit tile bundles; do not implement approximate node slicing.
- **Scientific Flexibility:** A schema-driven architecture that supports multiple models (the reference 2022 model, GraphCast, etc.) via external configuration and ONNX weights.
- **Enterprise Standards:** Mathematical verification (zero-degradation), memory safety (ASan/MSan), and robust packaging (scikit-build-core).

---

## Technical Stack

- **Inference:** ONNX Runtime (C++ API).
- **Bindings:** pybind11 (Zero-copy tensor transfer).
- **Core:** Modern C++ (C++20/23) with SIMD (AVX2) optimizations.
- **Python Integration:** xarray, numpy, pandas, Dask (for out-of-core processing).
- **Build System:** CMake + scikit-build-core.

---

## Architecture: Control vs. Data Plane

### 1. Data Plane (C++)
Focuses on performance and memory management.
- **`WeatherGraphEngine`**: Manages the ONNX session and executes inference.
- **Memory:** RAII for deterministic destruction of heavy ONNX objects.
- **SIMD:** Accelerated normalization $(x - \mu) / \sigma$ directly in C++.

### 2. Control Plane (Python)
Focuses on flexibility and scientific usability.
- **`WeatherGraphModel`**: Wrapper class for `xarray` integration.
- **Schema-Driven**: Dynamically handles variable mapping, resolution, and normalization constants.
- **Out-of-Core**: Utilizes Dask to process large NetCDF archives while avoiding scheduler-driven memory spikes on CPU-only deployments.

---

## Implementation Status

Completed in the current repository state:

1. **Artifact and runtime extraction path:** Normalization stats, graph topology artifacts, and the reference ONNX loading path are wired into the WeatherGraph runtime.
2. **C++ backend hardening:** `WeatherGraphEngine` now reads output metadata from ONNX, supports configurable intra-op threads, and exposes optional low-memory ONNX Runtime flags.
3. **Python control-plane hardening:** `WeatherGraphModel` forwards runtime config, enforces contiguous input buffers, supports streamed rollout/export paths, and rejects unsupported latent-output artifacts.
4. **Exact tiled runtime contract:** `spatial_tiling`, `tile_bundle_path`, `tile_state_backend`, and `tile_state_dir` are implemented as exact-only runtime controls that require graph-aware tile bundles.
5. **Configurable reference-grid export metadata:** `reference_grid_shape` and `reference_grid_resolution_degrees` remove the old hardcoded 1-degree export assumption from the Python control plane.
6. **Pipeline propagation:** Generic execution-provider knobs, low-memory mode, exact tiling, and reference-grid controls are wired through README/examples, notebook workflows, reusable GitHub Actions, Ansible, Terraform/cloud-init, GCP Batch, AWS Batch, and the simulation runner.
7. **Researcher-facing CLI:** Installing the package exposes a supported `weathergraph` CLI for runtime inspection, source discovery, forecast execution, and tile-bundle packaging.
8. **Tile-bundle packaging path:** `weathergraph build-tile-bundle` and `exporter/build_tile_bundle.py` can derive halo-expanded input indices, owned output partitions, and bundle manifests from graph topology plus pre-exported per-tile ONNX files.
9. **Validation coverage:** Backend and wrapper tests cover allocator flags, runtime propagation, dynamic output shapes, exact tile-bundle stitching, configurable reference-grid export, CLI parsing, bundle generation, streaming export dispatch, and fail-fast behavior for unsupported artifacts.

Still pending by design:

1. **Per-tile ONNX export:** Bundle metadata packaging and halo/index generation are implemented, but exporting the per-tile ONNX artifacts themselves is still a separate step.
2. **Benchmark publication:** Peak RSS / latency profiles for allocator-disabled and tiled runs still need benchmark-backed reporting.

---

## User Guide: Weather Modeling with WeatherGraph

This guide describes how to perform global weather modeling using the optimized C++ backend.

### 1. Environment Setup

Ensure you are using the provided virtual environment which contains all necessary dependencies (`onnxruntime`, `xarray`, `dask`, `netCDF4`).

```bash
# Activate the environment
source weathergraph/venv/bin/activate

# Add the C++ core to LD_LIBRARY_PATH (required for ONNX Runtime)
export LD_LIBRARY_PATH=$(pwd)/weathergraph/core:$LD_LIBRARY_PATH
```

### 2. Basic Usage: Single-Step Prediction

The engine expects an `xarray.Dataset` containing ERA5 variables.

```python
import xarray as xr
from weathergraph import WeatherGraphModel

# 1. Initialize the model
model = WeatherGraphModel(
    model_path="weather_gnn.onnx",
    weights_dir="exporter/weights",
    disable_cpu_mem_arena=True,
    disable_mem_pattern=True,
)

# 2. Load your data (e.g., from NetCDF)
ds = xr.open_dataset("path/to/era5_initial_state.nc")

# 3. Predict the state 6 hours into the future
# Returns a numpy array [1, 71042, 78]
prediction = model.predict_one_step(ds)
```

### 2.5. Supported CLI Usage

When the package is installed, the supported front door for researchers is the
`weathergraph` CLI:

```bash
weathergraph list-sources

weathergraph inspect \
    --model-path models/weather_gnn.onnx \
    --weights-dir data \
    --execution-provider cuda \
    --execution-device-id 0

weathergraph forecast \
    --model-path models/weather_gnn.onnx \
    --weights-dir data \
    --data-source era5_netcdf \
    --input-path path/to/era5_initial_state.nc \
    --steps 40 \
    --output-format zarr \
    --output-path forecast_out
```

From a source checkout without installation, use `python -m weathergraph.cli`.

### 3. Advanced Usage: 10-Day Rollout (Autoregressive)

For multi-day forecasts, use `iter_forecast` or `forecast_export` when memory pressure matters. `forecast` remains available but materializes the full trajectory in RAM.

```python
# Perform a 40-step rollout (10 days total)
forecast_steps = model.forecast(ds, steps=40)

# 'forecast_steps' is a list of predicted states
final_state = forecast_steps[-1]
```

### 4. Working with Large Datasets (Out-of-Core)

To process massive ERA5 archives on CPU machines with tight memory envelopes, integrate with Dask. The engine is pre-configured for synchronous execution to prevent worker-pool memory spikes, and the runtime now also exposes optional low-memory ONNX Runtime flags when reserved RSS matters more than throughput.

```python
import dask
dask.config.set(scheduler='synchronous')

# Open dataset with chunking
ds = xr.open_dataset("large_archive.nc", chunks={'time': 1})

# Process each time slice one-by-one
for i in range(len(ds.time)):
    time_slice = ds.isel(time=i).compute()
    result = model.predict_one_step(time_slice)
    # Save result or append to file
    time_slice.close()
```

### 5. Data Contract & Variable Ordering

The engine strictly requires 78 atmospheric channels. Your input `xarray.Dataset` must contain the following variables:
- **Pressure Level Variables (at 13 levels):** `z` (geopotential), `q` (humidity), `t` (temperature), `u` (wind_u), `v` (wind_v), `w` (vertical_velocity).
- **Standard Levels (hPa):** 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000.

---

## Testing & Validation
 Strategy

The project adheres to three independent vectors of testing to ensure scientific and operational reliability:

### 1. Mathematical Determinism (Accuracy)
- **Error Propagation:** Verify that 40+ step auto-regressive rollouts do not exhibit exponential error growth compared to the JAX reference (`np.allclose` with `atol=1e-5`).
- **Impulse Response:** Use Dirac delta inputs to verify graph topology (senders/receivers) and ensure symmetrical physical wave propagation.
- **Normalization Boundaries:** Validate in-graph normalization against extreme values to prevent catastrophic cancellation or overflow.

### 2. Hardware Stability & FFI (Profile-Specific Constraints)
- **Memory Integrity:** All C++ FFI calls must detect non-contiguous memory (strides) and either handle them or raise a `ValueError`, preventing Segfaults.
- **Infinite Rollout Check:** Continuous 10,000+ step rollouts must maintain a flat memory profile (verified via `memory_profiler`).
- **Dask Resilience:** Force synchronous scheduling (`dask.config.set(scheduler='synchronous')`) to reduce out-of-core memory spikes.

Recommended sizing guidance:
- **1° experimental/reference artifact:** treat as a backend validation target and coarse demo path. Single-step and short-rollout runs may fit near 1-2 GB RAM, but this is not the main scientific deployment promise.
- **0.25° workstation baseline:** plan around at least 16 GB RAM for single-step or short-rollout inference, and 32-64 GB RAM for 40-step rollouts or full-trajectory export. Use `iter_forecast`, `forecast_export`, allocator-disabled mode, and exact tiling where appropriate to reduce working-set pressure.
- **Finer than 0.25°:** require separate capacity planning and profile-specific benchmarks before making public claims.

Current runtime status:
- `disable_cpu_mem_arena` and `disable_mem_pattern` are implemented and optional.
- Exact `spatial_tiling` is implemented at runtime via tile bundles.
- Naive node slicing remains intentionally unsupported.

### 3. Physical Resilience (Meteorological Anomalies)
- **NaN/Inf Handling:** Ensure the engine processes missing data (NaN) deterministically without crashing (SIGFPE).
- **Extreme Events:** Validate model stability during Category 5 hurricane or sudden stratospheric warming scenarios.

---

## Development & Quality Standards

- **Zero-Copy:** NEVER copy tensors between Python and C++. Use `ptr` from `buffer_info`.
- **Accuracy:** All optimizations must pass verification against reference models within $10^{-5}$ tolerance.
- **Safety:** All C++ code must be verified via AddressSanitizer (ASan) and MemorySanitizer (MSan).
- **Documentation:** Doxygen for C++ core; Type-hinted Python docstrings for the API.

---

## Project Status: Migration in Progress

Current Rust and Web codebases have been removed. Development is now focused on the consolidated root structure using `src/cpp` and the `weathergraph` Python package.

## Working Log

### 2026-05-13: Workspace Consolidation & Cleanup
- **Legacy Removal:** Deleted `keisler-rust-mcp` (Rust/Web) and redundant `keisler_engine` (root), `src/cpp` (root old), `tests` (root old), `exporter` (root old), and `onnxruntime-sdk` (root old).
- **Directory Restoration:** Mistakenly moved the project to `/root/`, which polluted the home directory. The project has been immediately moved back to its proper home in `keisler_weather_project/`.
- **Model Organization:** Moved `keisler_full_engine.onnx` to `models/` for consistent artifact management.
- **Verification:** C++ source is located at `src/cpp/main.cpp` and Python package is at `weathergraph/`.
- **Model Compatibility Note:** Identified that the current `exporter/build_gnn_graph.py` generates a prototype model that is currently incompatible with the `WeatherGraphModel` wrapper. The 40-step rollout tests fail as expected due to this prototype status. The core C++ engine and integration logic are verified via 30/32 passing tests using dummy artifacts.

