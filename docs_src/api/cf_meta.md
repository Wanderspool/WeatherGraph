# weathergraph.cf_meta

The `weathergraph.cf_meta` module defines Climate and Forecast (CF-1.11) metadata standards and exposes utilities to inject standard descriptions, variable attributes, physical units, and coordinate geometries into exported forecast files.

---

## Technical Overview: CF-1.11 Integration

In weather forecasting and climatology, output files must be shareable and self-describing. WeatherGraph automatically conforms to the **CF-1.11 metadata conventions** by mapping model variables to their standard physical definitions and UDUNITS units.

The following coordinate and variable attribute mappings are exported:

### Coordinate Standard Attributes
*   `time`: standard_name = `"time"`, axis = `"T"`
*   `latitude`: standard_name = `"latitude"`, units = `"degrees_north"`, axis = `"Y"`
*   `longitude`: standard_name = `"longitude"`, units = `"degrees_east"`, axis = `"X"`
*   `level`: standard_name = `"air_pressure"`, units = `"hPa"`, axis = `"Z"`

### Variable Standard Mappings
*   `z`: standard_name = `"geopotential"`, units = `"m2 s-2"`, long_name = `"Geopotential"`
*   `q`: standard_name = `"specific_humidity"`, units = `"kg kg-1"`, long_name = `"Specific humidity"`
*   `t`: standard_name = `"air_temperature"`, units = `"K"`, long_name = `"Temperature"`
*   `u`: standard_name = `"eastward_wind"`, units = `"m s-1"`, long_name = `"Eastward component of wind"`
*   `v`: standard_name = `"northward_wind"`, units = `"m s-1"`, long_name = `"Northward component of wind"`
*   `w`: standard_name = `"lagrangian_tendency_of_air_pressure"`, units = `"Pa s-1"`, long_name = `"Vertical velocity"`

---

## Technical Example

The following script demonstrates how to construct a CF-compliant dataset manually or inject attributes into an existing NetCDF file on disk:

```python
import weathergraph.cf_meta as cf
import xarray as xr
import numpy as np

# 1. Manually build a dataset with coordinate structures
lat = np.linspace(90, -90, 181)
lon = np.linspace(0, 359, 360)
times = ["2026-06-06T12:00:00"]

# 2. Build dataset wrapper
ds = cf.build_cf_dataset(
    data_array=np.zeros((1, 181, 360, 78), dtype=np.float32),
    times=times,
    lat=lat,
    lon=lon,
    levels=[50, 500, 1000]
)

# 3. Save as netCDF on disk
ds.to_netcdf("output/unformatted.nc")

# 4. Inject CF-1.11 attributes directly into the netCDF file layout
cf.inject_cf_attrs_netcdf(
    path="output/unformatted.nc",
    variable="t",
    level=500
)
```

---

## Constants

*   **`CF_VARIABLE_ATTRS`** (`dict[str, dict[str, str]]`): Maps standard short names (`z`, `q`, `t`, `u`, `v`, `w`) to long names, CF standard names, and UDUNITS units.
*   **`CF_COORDINATE_ATTRS`** (`dict[str, dict[str, str]]`): Maps coordinate dimensions (`latitude`, `longitude`, `time`, `level`) to CF attributes.

---

## Dataset Builders

::: weathergraph.cf_meta.build_cf_dataset
    options:
      show_source: true

---

## Metadata Injection Helpers

::: weathergraph.cf_meta.inject_cf_attrs_netcdf
::: weathergraph.cf_meta.inject_cf_attrs_zarr

---

## Coordinate Sorting Utilities

::: weathergraph.cf_meta.ensure_pressure_order
    options:
      show_source: true
