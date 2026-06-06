# Troubleshooting & FAQ

This troubleshooting guide addresses the most common configuration errors, runtime performance bottlenecks, and scientific calculation issues encountered while deploying WeatherGraph.

---

## 1. ONNX Runtime & Accelerator Failures

### `ImportError: libonnxruntime.so.1.18.0: cannot open shared object file: No such file or directory`
*   **Cause**: The compiled C++ backend extension (`weathergraph_backend.so`) is unable to locate the ONNX Runtime dynamic library file at runtime.
*   **Resolution**: 
    If you built the project from source, ensure you ran the installation in editable mode or compiled the distribution package:
    ```bash
    pip install -e .
    ```
    On custom Linux installations, the dynamic loader path might not include `/opt/weathergraph/weathergraph/core`. You can manually inject the load path using the `LD_LIBRARY_PATH` environment variable:
    ```bash
    export LD_LIBRARY_PATH=/opt/weathergraph/weathergraph/core:$LD_LIBRARY_PATH
    ```

### `RuntimeError: ONNX Runtime Error: [Invalid Argument] Fail to load model`
*   **Cause**: The ONNX file does not conform to the expected WeatherGraph input/output contract (shape `[1, nodes, 78]`).
*   **Resolution**:
    Validate that the GNN ONNX graph was compiled correctly. You can check the expected input and output tensor shapes using the `inspect` subcommand:
    ```bash
    weathergraph inspect --model-path models/my_unsupported_model.onnx
    ```
    If the shapes mismatch, verify the tracing code inside the exporter scripts. Note that latent-space GNN models are not directly compatible with the main autoregressive rollout wrapper.

### GPU Execution Provider Fallback Warnings
*   **Symptom**: The engine logs warnings stating that operators are falling back to the CPU execution provider.
*   **Cause**: You initialized the model with `execution_provider="cuda"`, but some custom operators (or model layout configurations) are not supported by the CUDA runtime version installed on the host.
*   **Resolution**:
    To detect exactly which nodes are falling back and causing performance bottlenecks, instantiate the model with CPU fallbacks disabled:
    ```python
    model = WeatherGraphModel(
        model_path="models/weather_gnn.onnx",
        weights_dir="data",
        execution_provider="cuda",
        disable_cpu_ep_fallback=True  # Will raise an error on fallback
    )
    ```
    This triggers an explicit error at session load time showing the incompatible operators.

---

## 2. Ingestion & Data Source Errors

### `FileNotFoundError: ERA5 NetCDF file not found`
*   **Cause**: The local data file path passed to the `era5_netcdf` data adapter is incorrect, or the CDS server did not write the target file.
*   **Resolution**:
    Check that the file exists and is readable. If you are using remote downloading, verify that your credentials are set:
    *   **Copernicus CDS**: Ensure your `~/.cdsapirc` contains your correct personal API credentials:
        ```text
        url: https://cds.climate.copernicus.eu/api/v2
        key: 12345:abcdef-1234-abcd-5678-ef1234567890
        ```

### GFS GRIB2 Indexing Timeouts
*   **Symptom**: The GFS adapter hangs or crashes with index parsing errors.
*   **Cause**: NOAA's real-time GFS GRIB2 files are hosted on public AWS S3 buckets. When requesting a very recent forecast step, NOAA might not have finished uploading the complete GRIB2 index file (`.idx`).
*   **Resolution**:
    Add a retry buffer or request the previous run's analysis step (e.g. asking for the 18:00 run instead of the 00:00 run) to ensure index files are complete.

### `TypeError: CustomAdapter.__init__() got an unexpected keyword argument`
*   **Cause**: Passing invalid key names to `load_source("custom", ...)` constructor.
*   **Resolution**:
    The variable mapping parameter is called `variable_map` (matching standard Python dictionaries), not `variable_mapping`. Use:
    ```python
    adapter = load_source("custom", path="init.nc", variable_map={"z": "geopotential"})
    ```

---

## 3. Memory & Performance Issues

### CPU Workstation Out-Of-Memory (OOM) Crashes
*   **Symptom**: The operating system terminates the Python script (`Killed` / exit code 137) during long forecasts.
*   **Cause**: The standard `model.forecast()` method caches the entire multi-day trajectory in system RAM. At high resolution, this quickly exhausts system memory.
*   **Resolution**:
    Switch from in-memory forecasting to streaming export mode. The `forecast_export` method streams predictions directly to disk:
    ```python
    model.forecast_export(
        initial_ds,
        steps=56,
        output_format="netcdf4",
        output_path="output/long_run"
    )
    ```
    If you are developing custom pipelines, process steps iteratively using Python generators:
    ```python
    for step_idx, step_ds in enumerate(model.iter_forecast(initial_ds, steps=56)):
        # Process step_ds and discard it to free RAM
        pass
    ```

### GPU Virtual Memory Bloat (VRAM OOM)
*   **Symptom**: CUDA runs out of memory during initialization.
*   **Cause**: ONNX Runtime pre-allocates an execution memory arena containing static execution paths.
*   **Resolution**:
    Configure a strict memory limit on the CUDA execution provider using the `execution_memory_limit` parameter:
    ```python
    model = WeatherGraphModel(
        model_path="models/weather_gnn.onnx",
        weights_dir="data",
        execution_provider="cuda",
        execution_memory_limit=6 * 1024 * 1024 * 1024  # Limit to 6 GB
    )
    ```

---

## 4. Scientific & Numerical Issues

### Visual Seam Discontinuities on Tiled Forecasts
*   **Symptom**: Predictions show artificial straight lines or coordinate offsets at the borders of spatial partitions.
*   **Cause**: Spatial partitions are stitched using hard coordinate transitions (no overlaps), or the number of Message-Passing hops in the GNN exceeds the `halo-hops` configured during tile bundle generation.
*   **Resolution**:
    1.  Re-generate the tile bundle with a larger `halo-hops` value matching the exact model architecture (typically `--halo-hops 2` or `3`).
    2.  Provide a blending weight array (`output_weights_path` in the manifest) to compute soft cosine transitions rather than hard cutoffs.

### Exploding Values or NaNs after 10+ Steps
*   **Symptom**: Physical variables like temperature or humidity diverge to infinity or NaN values during long autoregressive rollouts.
*   **Cause**: GNNs can drift outside their training distributions over long rollout horizons.
*   **Resolution**:
    Enforce physical limits by loading a constraints projection model (such as non-negativity of humidity or bounds on air pressure) directly in C++:
    ```python
    model = WeatherGraphModel(
        model_path="models/weather_gnn.onnx",
        weights_dir="data",
        constraints_model_path="models/physics_projection.onnx"
    )
    ```

---

## 5. Edge-Case Operational Incidents

This section covers rare error reports from distributed cloud deployments, complex storage mapping layouts, and specialized data assimilation scenarios.

### distributed.worker - WARNING - Memory limit exceeded
*   **Symptom**: Dask cluster workers crash or report warnings about high memory usage, followed by task restarts.
*   **Cause**: When using WeatherGraph within a distributed Dask context (e.g., using `dask.distributed`), workers loading chunked Zarr fields for forecast output verification may load entire variables into local heap spaces if the chunk sizes are too large or if workers are running multi-threaded.
*   **Resolution**:
    1. Re-chunk input and verification files so that each chunk corresponds to exactly one time step and is aligned with spatial GNN node layout structures.
    2. Configure workers to use process-based memory separation instead of threads:
       ```bash
       dask-worker tcp://scheduler:8786 --nthreads 1 --nworkers 4
       ```

### `RuntimeError: C++ core failed: TensorRT engine initialization failed`
*   **Symptom**: Model loading crashes with C++ pybind runtime errors specifically when initializing the TensorRT Execution Provider.
*   **Cause**: TensorRT creates dynamic execution kernels optimized for the specific GPU architecture. If the model is loaded on a GPU with a different compute capability than the one used during ONNX conversion, or if there is a mismatch in CUDA compute compatibility variables, compilation fails.
*   **Resolution**:
    1. Delete the cached TensorRT profile directories (usually placed under `.tensorrt_cache/` or `/tmp/`).
    2. Run the forecast CLI command with `--execution-provider-options` targeting engine rebuilding:
       ```bash
       weathergraph forecast \
         --execution-provider tensorrt \
         --execution-provider-options '{"trt_force_sequential_engine_build": "1", "trt_cache_path": "/tmp/trt_cache"}'
       ```

### `NetCDF: HDF error` / File Lock Exceptions
*   **Symptom**: Writing forecast steps streams raises HDF5 file lock exceptions, especially on network file systems (NFS) or high-performance Lustre storage arrays.
*   **Cause**: The HDF5 library used by netCDF4 enforces strict multi-process write locks. If multiple concurrent tasks or trailing threads attempt to access the same directory or file pointer, the library crashes.
*   **Resolution**:
    1. Set the following environment variables to disable HDF5 file locking:
       ```bash
       export HDF5_USE_FILE_LOCKING=FALSE
       ```
    2. Ensure that you write separate NetCDF4 files for each forecast step rather than writing to a single file from multiple asynchronous worker threads. Use:
       ```python
       # Safe iterative write pattern
       for idx, step_ds in enumerate(model.iter_forecast(initial_ds, steps=10)):
           step_ds.to_netcdf(f"output/step_{idx:03d}.nc")
       ```

