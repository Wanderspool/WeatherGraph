# weathergraph.model

The `weathergraph.model` module contains the primary user-facing classes and functions to configure, load, and run neural weather prediction models. It integrates the Python-level data structures (`xarray`, `numpy`, and `dask`) with the high-performance C++ backend (`weathergraph_backend`).

---

## Architectural Principles & Design

The model engine operates on a GNN-based weather representation:
1.  **State Arrays**: Tensors are structured as shape `[1, nodes, 78]`, representing the batch dimension, graph nodes (coordinates), and 78 physical channels ($13 \text{ levels} \times 6 \text{ variables}$).
2.  **Autoregressive Rollout**: Forecasts are generated iteratively:
    $$\mathbf{x}_{t+1} = f_\theta(\mathbf{x}_t)$$
    where $\mathbf{x}_t$ is the flattened atmospheric state at step $t$ and $f_\theta$ is the GNN model.
3.  **Tiling Engine**: For high-resolution meshes, the tiling engine partitions the global graph into regional subgraphs, executes inference per tile, and performs boundary exchange with soft weighted blending.

---

## Technical Examples

### Example 1: Standard CPU/GPU Forecasting
The following script demonstrates the standard pattern for instantiating the model on CUDA and running a 12-step rollout:

```python
import weathergraph as wg
import xarray as xr

# 1. Load data via CDS or local archives
adapter = wg.load_source("era5_netcdf", path="data/era5_archives/init.nc")
initial_ds = adapter.load()

# 2. Initialize the model on GPU with custom memory limits
model = wg.WeatherGraphModel(
    model_path="models/weather_gnn.onnx",
    weights_dir="data/normalization",
    execution_provider="cuda",
    execution_memory_limit=8 * 1024 * 1024 * 1024  # 8 GB VRAM limit
)

# 3. Predict the 72-hour forecast trajectory (12 steps of 6 hours)
forecast_ds = model.forecast(
    initial_ds,
    steps=12,
    as_dataset=True
)

# 4. Save to a CF-compliant NetCDF file
forecast_ds.to_netcdf("output/forecast_72h.nc")
```

### Example 2: Memory-Efficient Generator Streaming
For low-resource nodes, you can yield forecast steps iteratively to avoid caching the entire rollout history in RAM:

```python
import weathergraph as wg

model = wg.WeatherGraphModel(
    model_path="models/weather_gnn.onnx",
    weights_dir="data/normalization",
    execution_provider="cpu",
    disable_cpu_mem_arena=True  # Release RSS immediately
)

# Iterate over prediction steps, saving them and freeing RAM
for step_idx, step_tensor in enumerate(model.iter_forecast(initial_ds, steps=24)):
    # Convert step tensor to dataset
    step_ds = model.to_dataset(step_tensor, time_step_idx=step_idx)
    step_ds.to_netcdf(f"output/step_{step_idx:03d}.nc")
    
    # Python GC takes care of releasing memory
    del step_ds
    del step_tensor
```

---

## Classes & References

::: weathergraph.model.WeatherGraphModel
    options:
      show_source: true
      show_bases: true
      members:
        - __init__
        - forecast
        - forecast_export
        - predict_one_step
        - predict_ensemble
        - iter_forecast
        - estimate_state_bytes
        - estimate_tiled_memory_report

---

::: weathergraph.model.EnsembleStats
    options:
      show_source: true
      show_bases: true
