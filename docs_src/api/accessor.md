# weathergraph.accessor

The `weathergraph.accessor` module implements a custom namespace registration that attaches WeatherGraph inference interfaces directly onto all `xarray.Dataset` objects as `.weathergraph`.

---

## Technical Overview & Accessor Patterns

WeatherGraph registers an `xarray` dataset accessor under the `.weathergraph` namespace. This design pattern integrates forecast capabilities into standard climate data pipelines. 

By utilizing the accessor, researchers can:
1.  **Validate Layout**: Verify that standard pressure levels are present and sorted in the correct vertical direction (descending hPa pressure).
2.  **Bind Active Engines**: Load and bind a `WeatherGraphModel` instance directly to a dataset container, maintaining state alongside coordinates.
3.  **Perform Seamless Inference**: Execute predictions directly on the loaded dataset, automatically deriving physical parameters and returning aligned coordinate grids.
4.  **Interop with MetPy & xCDAT**: Format coordinate metadata variables to compile with standard packages without manual renaming.

---

## Code Examples

### Example 1: Loading, Re-ordering, and Inference
Datasets downloaded from third-party APIs can have pressure coordinates sorted in ascending order (from 50 to 1000 hPa) or descending order. WeatherGraph models require consistent ordering. The accessor automates this:

```python
import xarray as xr
import weathergraph  # Registers the .weathergraph accessor namespace

# 1. Load an unaligned raw dataset
ds = xr.open_dataset("data/raw_era5.nc")

# 2. Sort the vertical level coordinates to descend (1000hPa -> 50hPa)
ds = ds.weathergraph.ensure_pressure_order(ascending=False)

# 3. Load the model and attach it directly to the dataset context
ds.weathergraph.load_model(
    model_path="models/weather_gnn.onnx",
    weights_dir="data/normalization",
    execution_provider="cpu"
)

# 4. Execute prediction using the attached model instance
forecast_ds = ds.weathergraph.predict(steps=12)
```

### Example 2: Interoperability Formats
To pass forecast arrays to metpy or xcdat for spatial analysis, variables must carry correct CF conventions and attributes. The accessor handles this conversion:

```python
# Convert dataset attributes to comply with MetPy's unit resolver
metpy_ds = forecast_ds.weathergraph.prepare_for_metpy()

# Convert coordinates to comply with xcdat spatial operations
xcdat_ds = forecast_ds.weathergraph.prepare_for_xcdat()
```

---

## Xarray Accessor Interface

::: weathergraph.accessor.WeatherGraphAccessor
    options:
      show_source: true
      members:
        - __init__
        - load_model
        - predict
        - ensure_pressure_order
        - prepare_for_metpy
        - prepare_for_xcdat
        - to_zarr
