# Quickstart (CLI)

The `weathergraph` command-line interface provides operational engineers and researchers with direct control over the weather forecasting pipeline, from inspection to rollout, visualization, and validation—without writing any Python code.

---

## 1. Subcommands Overview

The CLI supports the following subcommands:

| Subcommand | Description |
| :--- | :--- |
| **`list-sources`** | Lists all registered data-source adapters (local and remote). |
| **`inspect`** | Instantiates the ONNX model and prints a detailed runtime and memory sizing report. |
| **`forecast`** | Executes single-step, iterative, or exported multi-step autoregressive rollouts. |
| **`ensemble`** | Runs $O(1)$-memory perturbed ensemble rollouts with real-time statistics. |
| **`visualize`** | Generates Leaflet interactive maps (HTML) or animations (MP4, GIF). |
| **`pipeline`** | Runs a complete end-to-end download, compile, forecast, and visualization flow. |
| **`build-tile-bundle`** | Prepares graph-aware spatial tiling bundle manifests and index files. |
| **`download-model`** | Downloads a serialized model package from a remote URI. |
| **`compile-model`** | Converts weights archives (.pkl) into optimized ONNX computation graphs. |

---

## 2. Setting Up the Environment

When you run commands that interact with remote APIs (like ECMWF Open Data or NOAA GFS), the CLI automatically reads environment variables from a local `.env` file stored in your working directory.

You can configure credentials by creating a `.env` file:

```env
# Credentials for Copernicus CDS reanalysis API
CDSAPI_URL=https://cds.climate.copernicus.eu/api/v2
CDSAPI_KEY=12345:abcdef-1234-abcd-5678-ef1234567890
```

---

## 3. Command Examples

Here are the most common operational workflows performed via the CLI:

### List Registered Sources
List all built-in adapters to see what formats and data portals are ready for ingestion:

```bash
weathergraph list-sources
```

### Inspect a Model
Construct a session, load model variables, and output a detailed hardware and memory-usage report:

```bash
weathergraph inspect \
  --model-path models/weather_gnn.onnx \
  --weights-dir data \
  --execution-provider cpu
```

Add the `--json` flag to receive structured output suitable for monitoring pipelines:

```bash
weathergraph inspect --model-path models/weather_gnn.onnx --json
```

### Run a Standard Forecast
Run a 10-step (60-hour) autoregressive forecast using the local NetCDF adapter, and export the resulting trajectory directly to NetCDF4 files:

```bash
weathergraph forecast \
  --model-path models/weather_gnn.onnx \
  --weights-dir data \
  --data-source era5_netcdf \
  --input-path data/era5_archives/init.nc \
  --steps 10 \
  --output-format netcdf4 \
  --output-path output/forecast_run_01
```

### Run a Probabilistic Ensemble
Execute a 40-step rollout using 50 ensemble members, injecting additive Gaussian noise at each step, and calculating frost risk probabilities:

```bash
weathergraph ensemble \
  --model-path models/weather_gnn.onnx \
  --weights-dir data \
  --data-source gfs \
  --source-arg date=2026-06-01 \
  --steps 40 \
  --members 50 \
  --perturbation-scale '{"t": 0.5, "q": 0.001}' \
  --threshold frost=t@850<273.15 \
  --output-format netcdf4 \
  --output-path output/ensemble_run_01
```

### Visualize the Forecast
Generate an MP4 wind animation from the generated forecast NetCDF dataset:

```bash
weathergraph visualize \
  --input output/forecast_run_01/u_850hPa.nc \
  --variable u \
  --format mp4 \
  --output output/wind_animation.mp4 \
  --cmap jet
```

Or generate an interactive HTML map for temperature at step index 4 (24 hours):

```bash
weathergraph visualize \
  --input output/forecast_run_01/t_1000hPa.nc \
  --variable t \
  --format html \
  --time-index 4 \
  --output output/temp_map.html
```

---

## 4. One-Click Pipeline Execution

For testing and rapid onboarding, you can run a complete end-to-end workflow using the `pipeline` subcommand. This command downloads a model package, compiles it to ONNX format, runs a rollout forecast, and renders an MP4 animation—all in a single execution:

```bash
weathergraph pipeline \
  --model-url https://assets.wanderspool.org/models/weathergraph_weights.pkl \
  --data-source ecmwf_open \
  --source-arg date=2026-06-05 \
  --steps 4 \
  --vis-variable t \
  --work-dir ./pipeline_test
```
All intermediate model weights, ONNX files, NetCDF forecasts, and output animations will be saved in the `./pipeline_test` directory.