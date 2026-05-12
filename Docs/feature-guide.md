# WeatherGraph Feature Guide

## Purpose

This document explains the major runtime features that are already implemented in WeatherGraph and the constraints that matter when you use them in practice. It is meant to complement the top-level README by going one step deeper into what each feature does, where it lives in the codebase, and when it should or should not be used.

## What WeatherGraph Provides Today

WeatherGraph is a two-layer weather inference stack:

- A C++ data plane that executes ONNX models through ONNX Runtime.
- A Python control plane that accepts `xarray.Dataset` inputs, prepares tensors, drives autoregressive rollout, and exports results.

The current repository state already includes the following implemented capabilities.

## 1. C++ ONNX Runtime Backend

Main code: `src/cpp/main.cpp`

What it does:

- Loads an ONNX model with ONNX Runtime.
- Reads the real output tensor shape from model metadata instead of assuming input and output shapes are identical.
- Exposes a Python module named `weathergraph_backend` with the class `WeatherGraphEngine`.
- Accepts NumPy-backed input arrays and writes prediction output directly into a Python-owned output buffer.

Why it matters:

- This is the performance-critical path.
- It keeps Python orchestration separate from low-level inference.
- It makes low-memory tuning possible at the runtime level.

## 2. Zero-Copy Python to C++ Boundary

Main code: `src/cpp/main.cpp`, `weathergraph/model.py`

What it does:

- The Python layer prepares a contiguous `float32[1, nodes, 78]` tensor.
- The C++ backend wraps the NumPy memory buffer directly with ONNX Runtime tensor objects.
- Output is also written directly into a Python NumPy array.

Important constraint:

- Input arrays must be C-contiguous.
- The backend rejects strided input rather than risking incorrect reads or crashes.

## 3. Scientific Python Control Plane

Main code: `weathergraph/model.py`

What it does:

- Accepts either an `xarray.Dataset` or a `DataSourceAdapter`.
- Converts atmospheric variables into the strict 78-channel model order.
- Creates and owns the runtime engine.
- Provides single-step prediction, autoregressive rollout, and export helpers.
- Applies exact tile-bundle orchestration when tiled inference is enabled.

Main public entry points:

- `predict_one_step()`
- `forecast()`
- `iter_forecast()`
- `forecast_export()`

## 4. Strict Atmospheric Input Contract

Main code: `weathergraph/model.py`, `weathergraph/data_sources.py`

Current reference contract:

- Variables: `z`, `q`, `t`, `u`, `v`, `w`
- Pressure levels: `50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000`
- Channel count: `6 variables x 13 levels = 78 channels`

Operational meaning:

- If your upstream data does not match this contract, it must be adapted before inference.
- The adapters in `weathergraph/data_sources.py` exist to normalize external sources into this expected shape.

## 5. Multiple Data-Source Adapters

Main code: `weathergraph/data_sources.py`

Implemented adapters:

- `era5_netcdf`: local ERA5 NetCDF files.
- `ecmwf_open`: ECMWF open-data forecast download.
- `cds_era5`: Copernicus CDS reanalysis download.
- `gfs`: NOAA GFS download through public data access.
- `open_meteo`: single-point forecast source.
- `custom`: custom file and variable mapping.

Why it matters:

- The model runtime does not need to care where the initial state came from.
- Pipelines can swap data sources without rewriting inference code.

## 6. Forecast Modes

Main code: `weathergraph/model.py`

### `predict_one_step()`

Use this when you need exactly one 6-hour prediction from a prepared state.

### `forecast()`

Use this when you want a Python list containing every autoregressive step.

Trade-off:

- Easy to inspect in Python.
- Highest memory cost because the full trajectory is materialized in RAM.

### `iter_forecast()`

Use this when you want to process steps one-by-one.

Trade-off:

- Lower peak memory than `forecast()`.
- Better choice for long rollouts or streaming post-processing.

### `forecast_export()`

Use this when the real goal is an output artifact on disk, not an in-memory Python list.

Supported formats:

- `netcdf4`
- `zarr`
- `npz`

Behavior:

- `netcdf4` and `zarr` stream step-by-step to disk.
- `npz` still materializes the trajectory first because the archive format is not append-friendly.

## 7. Optional Low-Memory ONNX Runtime Mode

Main code: `src/cpp/main.cpp`, `weathergraph/model.py`

Implemented runtime knobs:

- `intra_op_threads`
- `disable_cpu_mem_arena`
- `disable_mem_pattern`

What the low-memory switches do:

- `disable_cpu_mem_arena=True` disables ONNX Runtime's CPU arena allocator.
- `disable_mem_pattern=True` disables ONNX Runtime memory-pattern reuse.

Why this exists:

- On memory-constrained machines, reserved RSS can matter more than raw throughput.
- These switches make it possible to lower static memory reservation in exchange for slower execution.

When to use it:

- Constrained CPU inference.
- Troubleshooting memory headroom or false OOM conditions.
- Pipeline runs where latency is less important than staying within a memory budget.

When not to use it by default:

- Throughput-oriented workstation runs.
- Stable environments where allocator reuse is beneficial.

## 8. Exact Spatial Tiling

Main code: `weathergraph/model.py`

Implemented runtime knobs:

- `spatial_tiling`
- `tile_bundle_path`

What this feature is:

- A graph-aware tiled execution path.
- It is exact-only.
- It requires a tile bundle manifest plus per-tile ONNX artifacts and explicit input/output node-index arrays.

What it is not:

- It is not naive node slicing.
- It is not an approximation mode.
- It does not auto-generate tile bundles from an arbitrary global model.

How it works at runtime:

- The Python layer loads the tile manifest.
- Each tile receives the correct input node subset.
- Each tile produces output for its owned nodes.
- The global output buffer is reconstructed by writing each tile result into the declared output indices.

Fail-fast behavior:

- If tiling is requested without a tile bundle, the model raises an error.
- If output coverage overlaps or leaves gaps, the bundle is rejected.
- If a tile produces the wrong output shape, execution stops immediately.

## 9. Reference-Grid Streaming Export

Main code: `weathergraph/model.py`

The current export path assumes the current 1-degree ERA5 reference-grid layout for reshaping node outputs into `(lat, lon, channel)` form.

What this means:

- NetCDF4 and Zarr export are production-usable for the current reference artifact.
- Generic export for arbitrary graph layouts is not implemented yet.
- Raw `npz` export remains the more general fallback when you only need the trajectory tensor.

## 10. Pipeline Integrations

Operational surfaces already wired to the current runtime:

- `examples/simulate_meteorologist.py`
- `examples/playbooks/jupiter_notebook.ipynb`
- `examples/playbooks/github_actions.yml`
- `examples/playbooks/ansible/site.yml`
- `examples/playbooks/terraform/`
- `examples/playbooks/gcp/batch-job.yaml`
- `examples/playbooks/aws/cloudformation.yaml`
- `.github/workflows/model-pipeline.yml`

These surfaces already accept the important runtime controls such as low-memory flags and exact tiling parameters.

## 11. Validation and Safety Features

Main code: `tests/`

Current validation coverage includes:

- backend output-shape handling
- low-memory flag forwarding
- contiguous-input enforcement
- dynamic runtime configuration
- exact tile-bundle stitching
- streaming export dispatch
- historical validation separation into smoke and optional extended checks
- repeated-inference memory behavior testing

## Practical Deployment Expectations

WeatherGraph should be explained using two different expectations instead of one generic RAM claim.

### Experimental 1-degree reference profile

Use this for:

- backend validation
- exporter experiments
- coarse global smoke tests

### Recommended 0.25-degree workstation profile

Use this mental model for serious global inference.

Practical implication:

- single-step and short-rollout runs should be planned with workstation-level memory
- long rollouts and full-trajectory exports need materially more headroom
- `iter_forecast()`, `forecast_export()`, and optional low-memory mode are important tools, but they do not change the fact that high-resolution inference remains a workstation-class workload

## Intentional Non-Goals in the Current Tree

Not implemented yet:

- automatic tile-bundle generation from a global ONNX model
- approximate spatial slicing
- generic graph-to-grid export metadata for arbitrary node layouts
- benchmark publication for all allocator-disabled and tiled runtime profiles

## Feature-to-Code Map

- Runtime backend: `src/cpp/main.cpp`
- Python orchestration: `weathergraph/model.py`
- Data ingestion adapters: `weathergraph/data_sources.py`
- Exporter experiments: `exporter/`
- Tests and contract verification: `tests/`
- Operational usage examples: `examples/`