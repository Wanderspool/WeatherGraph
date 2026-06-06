# Climate Ecosystem Integration

WeatherGraph integrates directly with the standard scientific Python climate analysis ecosystem. By registering a custom namespace on `xarray.Dataset` objects and providing best-effort helpers for libraries like `MetPy` and `xCDAT`, WeatherGraph simplifies scientific analysis.

---

## 1. The Xarray Accessor (`.weathergraph`)

When you import `weathergraph`, the package automatically registers a custom accessor namespace named `weathergraph` on all `xarray.Dataset` objects.

```python
import weathergraph as wg
import xarray as xr

# Open your initial dataset
ds = xr.open_dataset("data/era5_archives/init.nc")

# Run a 40-step forecast directly from the dataset object
forecast_ds = ds.weathergraph.predict(steps=40)
```

### Advantages of the Accessor Path
1.  **Chaining**: Write clean, declarative pipelines:
    ```python
    (
        xr.open_dataset("data/init.nc")
        .weathergraph.predict(steps=20, execution_provider="cuda")
        .weathergraph.prepare_for_metpy()
    )
    ```
2.  **In-Memory Session Caching**: Loading a 2 GB global weather GNN ONNX graph takes several seconds. The accessor caches the instantiated `WeatherGraphModel` in memory. Repeated calls to `ds.weathergraph.predict()` reuse the same backend session, avoiding weight reload overhead.
3.  **Automatic Argument Mapping**: Extra keyword arguments passed to `predict()` are forwarded directly to the `WeatherGraphModel` constructor.

---

## 2. MetPy Integration

[MetPy](https://unidata.github.io/MetPy/) is a collection of tools in Python for reading, visualizing, and performing calculations on weather data. Integrating GNN outputs with MetPy requires handling coordinate alignments and variable ordering.

### Pressure Level Sorting
WeatherGraph executes calculations with pressure levels sorted from top to bottom (50 hPa to 1000 hPa). However, MetPy's thermodynamic profile calculations (such as CAPE, CIN, and Lifted Index) require pressure levels sorted from highest pressure (surface) to lowest (top of the atmosphere).

The `prepare_for_metpy()` function manages this sorting and attaches coordinate metadata:

```python
from weathergraph.integrations import prepare_for_metpy
import metpy.calc as mpcalc

# Prepare forecast dataset for MetPy analysis
metpy_ds = forecast_ds.weathergraph.prepare_for_metpy()

# Calculate geostrophic wind using MetPy's quantified coordinates
z_quantified = metpy_ds["z"].metpy.quantify()
u_geo, v_geo = mpcalc.geostrophic_wind(z_quantified)
```

#### Under the Hood
1.  Calls `ensure_pressure_order(ascending=False)` to reverse the pressure coordinate axis to `[1000, 925, ..., 50]` hPa.
2.  Injects CF standard names and units attributes.
3.  Registers Cartopy Coordinate Reference System (CRS) metadata using `metpy.parse_cf()`.

---

## 3. xCDAT Integration

[xCDAT](https://xcdat.readthedocs.io/) is designed for climate data analysis on structured grids, providing utilities for spatial averaging, temporal averaging, and regridding.

To calculate area-weighted spatial statistics, xCDAT requires explicit coordinate bounds (e.g. `lat_bnds` and `lon_bnds`) outlining each grid cell. GNN models only output point values.

```python
from weathergraph.integrations import prepare_for_xcdat

# Add coordinate bounds and format time coordinates for xCDAT
xcdat_ds = forecast_ds.weathergraph.prepare_for_xcdat()

# Calculate area-weighted global average air temperature
global_mean_temp = xcdat_ds.spatial.average("t")
```

#### Under the Hood
1.  Checks if latitude/longitude coordinates exist.
2.  Calculates halfway boundary coordinates and adds `lat_bnds` and `lon_bnds` variables using `xcdat.bounds.add_bounds()`.
3.  Applies frequency-based bounds (`time_bnds`) to the time dimension for climate averaging.

---

## 4. Derived Diagnostics

Raw model outputs include wind vector components ($u$ and $v$) and geopotential ($z$). To calculate wind speeds or standard geopotential heights (in meters), WeatherGraph provides optimized numpy-backed diagnostic formulas:

```python
from weathergraph.integrations import compute_derived_diagnostics

# Add wind_speed (m/s) and geopotential_height (m) variables to the dataset
diagnostics_ds = compute_derived_diagnostics(forecast_ds)

# Access the newly generated diagnostic variables
print("Wind Speed:\n", diagnostics_ds["wind_speed"])
print("Geopotential Height (m):\n", diagnostics_ds["geopotential_height"])
```

*   `compute_wind_speed(ds)`: Computes $V_{\text{speed}} = \sqrt{u^2 + v^2}$ and injects CF attributes.
*   `compute_geopotential_height(ds)`: Computes $Z_{\text{gph}} = \frac{z}{g}$ where $g = 9.80665\text{ m/s}^2$ is the standard acceleration of gravity.