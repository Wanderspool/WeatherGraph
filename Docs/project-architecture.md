# WeatherGraph Project Architecture

## Purpose

This document explains how the WeatherGraph codebase is organized, how data moves through it, and which parts of the repository own which responsibilities. It is intended for engineers who need to read, extend, debug, or integrate the project.

## High-Level Design

WeatherGraph uses a split architecture:

- The data plane lives in C++ and owns ONNX Runtime execution.
- The control plane lives in Python and owns scientific input handling, rollout logic, export, and pipeline integration.

This separation is deliberate.

- C++ handles the performance-sensitive inference path.
- Python stays responsible for domain-friendly orchestration and integration with `xarray`, `numpy`, Dask, and automation surfaces.

## Repository Layout

The most important directories are:

- `src/cpp/`: the C++ ONNX Runtime backend exposed through pybind11.
- `weathergraph/`: the Python package that users import.
- `exporter/`: scripts related to graph extraction or ONNX conversion experiments.
- `examples/`: runnable examples and deployment/playbook templates.
- `tests/`: runtime, contract, and validation tests.
- `data/`: normalization arrays and graph-related artifacts used by the reference runtime.
- `.github/`: reusable CI actions and workflows.
- `onnxruntime-sdk/`: vendored or downloaded ONNX Runtime library payload used during builds.

## Core Runtime Path

The main runtime path starts in Python and crosses into C++ only for inference.

### Step 1. User entry point

Main files: `weathergraph/__init__.py`, `weathergraph/accessor.py`

The package exports `WeatherGraphModel`, the data-source adapter API, and the Xarray accessor (`.weathergraph`).

Users can start inference in two ways:
1. Object-oriented: instantiate `WeatherGraphModel(model_path=...)`
2. Data-oriented: call `ds.weathergraph.predict()` directly on an `xr.Dataset`

Installing the package also provides the supported `weathergraph` CLI for source discovery, runtime inspection, and forecast execution.

### Step 2. Python model orchestration

Main file: `weathergraph/model.py`

`WeatherGraphModel` owns:

- runtime option storage
- engine creation
- tile-bundle loading
- dataset resolution through adapters
- tensor preparation
- autoregressive looping
- export to NetCDF4, Zarr, or NPZ

Important constructor arguments:

- `model_path`
- `weights_dir`
- `intra_op_threads`
- `execution_provider`
- `execution_device_id`
- `execution_memory_limit`
- `execution_provider_options`
- `disable_cpu_ep_fallback`
- `disable_cpu_mem_arena`
- `disable_mem_pattern`
- `spatial_tiling`
- `tile_bundle_path`
- `reference_grid_shape`
- `reference_grid_resolution_degrees`
- `tile_state_backend`
- `tile_state_dir`

### Step 3. Dataset resolution

Main file: `weathergraph/model.py`

`_resolve_dataset()` accepts either:

- an already loaded `xarray.Dataset`
- a `DataSourceAdapter` instance

If the input is an adapter, the adapter's `load()` method is called lazily.

This is the point where the runtime stays independent from the origin of the data.

### Step 4. Input tensor preparation

Main file: `weathergraph/model.py`

`_prepare_input()` performs the model-specific flattening logic:

- iterate through pressure levels in a fixed order
- iterate through variables in a fixed order
- flatten each 2D field
- stack them into `[nodes, 78]`
- add the batch dimension to produce `[1, nodes, 78]`
- force a contiguous `float32` buffer

This function is where the scientific ordering contract becomes a concrete tensor contract.

### Step 5. Runtime engine creation

Main files:

- `weathergraph/model.py`
- `src/cpp/main.cpp`

Normal path:

- `WeatherGraphModel._create_engine()` instantiates `weathergraph_backend.WeatherGraphEngine`.

Tiled path:

- `WeatherGraphModel` creates a `_TiledEngineAdapter` which calls `_predict_tiled()` instead of a single global backend session.

### Step 6. Inference execution in C++

Main file: `src/cpp/main.cpp`

The backend:

- creates an ONNX Runtime session
- configures session options
- optionally adds one of the supported accelerator execution providers
- optionally disables CPU execution-provider fallback for strict accelerator placement
- optionally disables the CPU arena allocator
- optionally disables memory-pattern reuse
- discovers input and output names from the ONNX graph
- discovers the real output shape from ONNX metadata
- validates that input buffers are C-contiguous
- wraps Python NumPy buffers as ONNX Runtime tensors
- runs `session->Run(...)`

The pybind11 module exports the C++ class as `WeatherGraphEngine`.

## Autoregressive Forecast Path

Main file: `weathergraph/model.py`

The forecast flow is straightforward.

1. Prepare the initial state as `[1, nodes, 78]`.
2. Run one inference step.
3. Feed the output back in as the next input.
4. Repeat for the requested number of steps.

The public methods differ only in how results are surfaced.

- `predict_one_step()` returns one output tensor.
- `iter_forecast()` yields each step lazily.
- `forecast()` materializes every step into a Python list or CF-compliant `xr.Dataset`.
- `forecast_export()` streams or writes the rollout into a chosen file format with CF-1.11 metadata injection.

## Export Architecture

Main file: `weathergraph/model.py`

There are two export families.

### Raw trajectory export

Format: `npz`

Behavior:

- stores the raw `[steps, nodes, 78]` trajectory
- does not reshape into lat/lon files
- materializes the trajectory before compression

### Reference-grid scientific export

Formats:

- `netcdf4`
- `zarr`

Behavior:

- uses configured reference-grid metadata when reshaping the reference-node subset
- writes one file per variable and pressure level
- streams step-by-step to disk to reduce peak memory usage
- injects CF-1.11 standards (`standard_name`, `units`) via `weathergraph.cf_meta`

This part of the architecture still requires reference-grid metadata rather than a fully generic mesh-to-grid export contract.

## Exact Tiling Architecture

Main file: `weathergraph/model.py`

The tiling subsystem is controlled in Python, not in the raw C++ backend.

Why:

- tile orchestration requires manifest parsing
- different tile models may be loaded per partition
- the global tensor must be reassembled with explicit ownership metadata

The tile bundle contract includes:

- global input and output shapes
- per-tile model path
- per-tile input index array
- per-tile output index array

Validation rules enforced at load time:

- shapes must be 3D
- batch size must be 1
- channel count must be 78
- output node ownership must be unique
- output node coverage must be complete

Runtime sequence:

1. Slice the global input to the tile input indices.
2. Run the tile model.
3. Validate the tile output shape.
4. Scatter the tile output into the global output buffer.

This design keeps the C++ layer simple and leaves graph partition semantics in the Python control plane.

## Data-Source Architecture

Main file: `weathergraph/data_sources.py`

The data-source layer is a registry of adapters behind a common abstract base class.

Key idea:

- the runtime expects a consistent atmospheric dataset
- each adapter is responsible for fetching or loading data and returning that normalized view

Built-in adapters cover both local files and remote APIs.

- local ERA5 NetCDF
- ECMWF open data
- Copernicus CDS
- NOAA GFS
- Open-Meteo
- Zarr store (local and cloud)
- custom schema-driven mapping

This is the main extension point when a new upstream provider must be supported.

## Build and Packaging Architecture

Main files:

- `pyproject.toml`
- `CMakeLists.txt`
- `Makefile`
- `.github/actions/setup-weathergraph/action.yml`

Build flow:

1. Install Python dependencies including `pybind11`.
2. Download or reuse the ONNX Runtime SDK.
3. Configure the C++ build with CMake.
4. Build the shared object `weathergraph_backend.so`.
5. Copy the backend and ONNX Runtime shared libraries into `weathergraph/core/`.

This is the same core build pattern reused across local work, CI, and deployment playbooks.

## Exporter Architecture

Main directory: `exporter/`

The exporter area is not the main runtime path. It contains graph extraction and ONNX-related scripts.

Important distinction:

- the runtime wrapper expects an autoregressive ONNX artifact with output shape `[1, nodes, 78]`
- the prototype graph-construction path in `exporter/build_gnn_graph.py` is marked as experimental and emits a latent-output graph that is not directly compatible with the normal `WeatherGraphModel` autoregressive wrapper

This distinction exists to keep experimental exporter work from being mistaken for a production inference contract.

## Testing Architecture

Main files:

- `tests/test_cpp_backend.py`
- `tests/test_historical_validation.py`
- `tests/test_memory_leak.py`

Responsibilities:

- backend and wrapper contract verification
- dynamic output-shape behavior
- low-memory option forwarding
- exact tile-bundle execution
- historical smoke and extended validation modes
- repeated-inference memory behavior

These tests represent the contract boundary for most future runtime changes.

## Operational Surfaces

Main directory: `examples/`

The examples are not separate architectures. They are automation shells around the same runtime.

They differ mostly in where the run happens and who owns the environment:

- local script for manual use
- notebook for interactive exploration
- GitHub Actions for CI validation
- Ansible for an existing Linux host
- Terraform for provisioning a single cloud instance
- GCP Batch for managed batch execution
- AWS Batch via CloudFormation for AWS-managed batch execution

## Extension Points

If you need to extend the project, the cleanest extension points are:

- add a new `DataSourceAdapter` in `weathergraph/data_sources.py`
- add a new operational playbook in `examples/playbooks/`
- add new tests in `tests/` when changing runtime contracts
- add a new exporter script in `exporter/` if the runtime contract is kept explicit

If you need to change model tensor semantics, the most important places to audit together are:

- `weathergraph/model.py`
- `src/cpp/main.cpp`
- `tests/test_cpp_backend.py`
- any exporter that claims compatibility with the wrapper

## Design Invariants

Several invariants define the current code architecture.

- The wrapper expects autoregressive output shaped as `[1, nodes, 78]`.
- The current scientific contract is based on 78 atmospheric channels.
- Tiling is exact-only and must be manifest-driven.
- NetCDF4 and Zarr export currently assume the reference grid layout.
- Dask is forced to synchronous scheduling to avoid worker-pool memory spikes in CPU-oriented deployments.
- Low-memory runtime flags are optional and off by default.

Understanding these invariants is the fastest way to avoid accidental architectural regressions.