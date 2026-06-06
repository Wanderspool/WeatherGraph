# CLI Reference & Operations

The `weathergraph` executable command-line interface provides a scriptable, operational wrapper around all core modules. This reference guide describes every command, parameter, and flag.

---

## 1. Global CLI Parameters

The CLI supports the following global option groupings which are shared across subcommands like `inspect`, `forecast`, `ensemble`, and `pipeline`:

### Runtime Configuration
*   `--model-path PATH` (default: `"models/weather_gnn.onnx"`): Path to the primary autoregressive ONNX model.
*   `--weights-dir DIR` (default: `"data"`): Directory containing normalization means and standard deviations.
*   `--intra-op-threads INT` (default: `1`): Number of threads allocated inside ONNX Runtime to execute individual graph operators.
*   `--execution-provider EP` (default: `"cpu"`): Target backend for inference. Options: `"cpu"`, `"cuda"`, `"tensorrt"`, `"rocm"`, `"openvino"`.
*   `--execution-device-id INT` (default: `0`): GPU card ordinal index.
*   `--execution-memory-limit BYTES` (default: `0`): Memory allocation cap in bytes (0 = unlimited).
*   `--execution-provider-options JSON` (default: `None`): Key-value pair options passed to the ONNX Runtime provider.
*   `--disable-cpu-ep-fallback`: Force failure if a node in the model graph cannot be executed on the accelerator.
*   `--disable-cpu-mem-arena`: Turn off ONNX Runtime's memory allocator arena to reduce virtual memory footprint.
*   `--disable-mem-pattern`: Disable memory allocation pattern caching inside ONNX Runtime sessions.

### Grid Reference Options
*   `--reference-grid-shape SHAPE`: Spatial resolution formatted as `LATxLON` (e.g., `181x360`).
*   `--reference-grid-resolution-degrees FLOAT`: Regular grid resolution in degrees (e.g. `0.25`, `0.1`), automatically deriving grid shape.

### Spatial Tiling Options
*   `--spatial-tiling`: Enable graph-aware spatial decomposition.
*   `--tile-bundle-path PATH`: Path to the directory or manifest JSON containing tile partitions.
*   `--tile-state-backend {ram,memmap}` (default: `"ram"`): Memory mapping configuration for inter-tile state arrays.
*   `--tile-state-dir PATH`: Directory for disk-mapped inter-tile arrays (when using `memmap`).

---

## 2. Ingest Source Options

The following parameters control the data sources used during the `forecast`, `ensemble`, and `pipeline` subcommands:

*   `--data-source NAME` (default: `"era5_netcdf"`): Choose from registered adapters (e.g. `gfs`, `cds_era5`, `custom`).
*   `--input-path PATH`: Shorthand for file-backed sources (sets `path=PATH`).
*   `--source-kwargs JSON`: Pass structured configurations (e.g. `{"latitude": 51.5, "longitude": -0.1}`).
*   `--source-arg KEY=VALUE`: Repeated flags for quick command configurations.

---

## 3. Subcommand Reference

### `list-sources`
Lists registered ingestion data sources.

### `inspect`
Creates a mock model session and estimates CPU/GPU and system memory.
*   `--json`: Output results as a single line JSON string.

### `forecast`
Runs an autoregressive forecast rollout.
*   `--steps INT` (default: `40`): Number of 6-hour prediction cycles.
*   `--output-format {none,netcdf4,zarr,npz}` (default: `"none"`): Output file format.
*   `--output-path PATH`: Save directory.
*   `--start-time ISO_DATE`: Anchor date for the time dimension (e.g. `2026-06-06T12`).
*   `--json`: Print summary report in JSON format.

### `ensemble`
Runs multiple perturbed trajectories and aggregates statistics.
*   `--members INT` (default: `50`): Ensemble size.
*   `--perturbation-scale SCALE`: Noise scale factor (float or JSON dict mapping variables).
*   `--threshold NAME=EXPR`: Probabilistic thresholds (e.g., `freeze=t@1000<273.15`). Can be repeated.
*   `--aggregate-steps LIST`: Comma-separated step index list (e.g., `9,19,29,39`) to save RAM.
*   `--seed INT` (default: `0`): PRNG seed (0 for non-deterministic).
*   `--output-format {none,netcdf4,zarr}`: Statistical export format.
*   `--output-path PATH`: Output directory.

### `visualize`
Creates spatial figures and animations from forecast runs.
*   `--input PATH` (required): Path to the NetCDF file.
*   `--variable VAR` (required): Variable to display (e.g. `t`, `z`).
*   `--format {html,mp4,gif}` (default: `"mp4"`): Visualization type.
*   `--output PATH` (required): Path to save the map/animation file.
*   `--cmap NAME` (default: `"viridis"`): Colormap name.
*   `--time-index INT` (default: `0`): Time frame for HTML maps.
*   `--resolution {high,medium,low}` (default: `"medium"`): Animation resolution.

### `build-tile-bundle`
Generates a tile bundle from graph partition layouts.
*   `--output-dir DIR` (required): Save location.
*   `--senders-path PATH` (required): Path to the global graph senders array.
*   `--receivers-path PATH` (required): Path to the global graph receivers array.
*   `--tile-model-dir DIR` (required): Location of partitioned ONNX files.
*   `--tile-grid-shape SHAPE`: Regular grid shape layout for tiling.
*   `--halo-hops INT` (default: `1`): Border hop overlaps.

### `pipeline`
Runs complete downloading, ONNX compilation, forecasting, and MP4 generation.
*   `--model-url URL` (required): Package weights link.
*   `--vis-variable VAR` (default: `"t"`): Animation variable.
*   `--work-dir DIR`: Directory for cache and results.

---

## 4. Real-World Operational Recipes

Below are concrete, multi-line shell scripting recipes demonstrating how to orchestrate WeatherGraph CLI tools for typical operational pipelines.

### Recipe A: End-to-End Forecast and MP4 Visualization
This script fetches the analysis state, initializes the ONNX session with CUDA acceleration, runs a 10-step forecast, and generates an animation of air temperature.

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Inspect target model requirements
echo "=== Inspecting Model Contract ==="
weathergraph inspect \
  --model-path models/weather_gnn.onnx \
  --json

# 2. Run the autoregressive forecast (60 hours)
echo "=== Running Autoregressive Forecast Rollout ==="
weathergraph forecast \
  --model-path models/weather_gnn.onnx \
  --weights-dir data/normalization \
  --execution-provider cuda \
  --execution-memory-limit 8589934592 \
  --data-source era5_netcdf \
  --input-path data/era5_archives/init.nc \
  --steps 10 \
  --output-format netcdf4 \
  --output-path output/forecast_run.nc \
  --start-time 2026-06-06T12:00:00

# 3. Create video visualization of the temperature variable
echo "=== Generating MP4 Visualization Map ==="
weathergraph visualize \
  --input output/forecast_run.nc \
  --variable t \
  --format mp4 \
  --output output/t_forecast.mp4 \
  --cmap thermal \
  --resolution high
```

### Recipe B: Generating Tiles and Running Low-Memory Partitions
For workstations with limited RAM or older GPU hardware, the global mesh is split into regional subgrids (tiling). This script shows how to build the tile bundle and run tiled forecasts.

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Compile tile configurations from adjacency arrays
echo "=== Building Spatial Tile Bundles ==="
weathergraph build-tile-bundle \
  --output-dir data/tiles_manifest_0.25deg \
  --senders-path data/graph/senders.npy \
  --receivers-path data/graph/receivers.npy \
  --tile-model-dir models/partitioned_onnx/ \
  --tile-grid-shape 4x4 \
  --halo-hops 2

# 2. Execute tiled inference using memory-mapped swap spaces
echo "=== Running Low-Memory Tiled Forecast ==="
weathergraph forecast \
  --model-path models/weather_gnn.onnx \
  --spatial-tiling \
  --tile-bundle-path data/tiles_manifest_0.25deg/manifest.json \
  --tile-state-backend memmap \
  --tile-state-dir /tmp/weathergraph_mmap_swap \
  --data-source gfs \
  --source-arg run_hour=06 \
  --steps 24 \
  --output-format zarr \
  --output-path output/tiled_zarr_archive.zarr
```