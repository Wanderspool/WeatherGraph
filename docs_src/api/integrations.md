# weathergraph.integrations

The `weathergraph.integrations` module contains helper functions designed to bridge WeatherGraph output datasets with operational meteorological and climate analysis ecosystems, such as MetPy and xCDAT.

---

## Technical Overview & Physical Diagnostics

Raw GNN weather models output variables in standard SI units or specific meteorological definitions (e.g. Geopotential instead of Geopotential Height). To make these outputs compatible with downstream visualization and calculations, the integrations module handles two categories of operations:

### 1. Metadata Normalization
Packages like **MetPy** require coordinate metadata to be explicitly defined with CF conventions so that unit and dimensional parsing works. Similarly, **xCDAT** requires specific latitude/longitude name conventions to execute spatial average regridding. This module injects standard names, units, and axes into target arrays.

### 2. Derived Diagnostics
*   **Wind Speed ($m/s$)**: Computed from eastward wind ($u$) and northward wind ($v$) using the Euclidean norm:
    $$ws = \sqrt{u^2 + v^2}$$
*   **Geopotential Height ($gpm$)**: Derived from geopotential ($z$) by dividing by standard gravity ($g = 9.80665\text{ m/s}^2$):
    $$h_{gpm} = \frac{z}{9.80665}$$

---

## Technical Example

This example loads a forecast dataset, computes derived diagnostics, prepares it for MetPy, and uses MetPy to compute the precipitable water parameter:

```python
import xarray as xr
import weathergraph.integrations as integrations
from metpy.units import units
import metpy.calc as mpcalc

# 1. Load forecast dataset
ds = xr.open_dataset("output/forecast.nc")

# 2. Compute wind speed and geopotential height
ds_extended = integrations.compute_derived_diagnostics(ds)

# 3. Format dataset attributes for MetPy compatibility
ds_metpy = integrations.prepare_for_metpy(ds_extended)

# 4. Extract variables with MetPy unit arrays attached
temperature = ds_metpy["t"].metpy.unit_array
specific_humidity = ds_metpy["q"].metpy.unit_array
pressure = ds_metpy["level"].metpy.unit_array * units.hPa

# 5. Execute advanced MetPy computation (e.g., dewpoint temperature)
relative_humidity = mpcalc.relative_humidity_from_specific_humidity(
    pressure,
    temperature,
    specific_humidity
)
```

---

## Meteorological Ecosystem Preparations

::: weathergraph.integrations.prepare_for_metpy
::: weathergraph.integrations.prepare_for_xcdat

---

## Derived Diagnostic Computations

::: weathergraph.integrations.compute_wind_speed
::: weathergraph.integrations.compute_geopotential_height
::: weathergraph.integrations.compute_derived_diagnostics
