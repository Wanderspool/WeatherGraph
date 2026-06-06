# Working with Different Data Sources

WeatherGraph features a pluggable data ingestion architecture. The framework provides built-in adapters to download, parse, and normalize initial atmospheric conditions from various global meteorological offices and formats into the exact vertical levels and variables expected by the model.

---

## 1. Listing Available Sources

You can list all registered data sources programmatically using the Python API:

```python
from weathergraph.data_sources import list_sources

list_sources()
```

This outputs a clean directory of registered adapters:

```text
Name                  Auth?   Description
----------------------------------------------------------------------
era5_netcdf           no      Local ERA5 NetCDF file
ecmwf_open            no      ECMWF Open Data — free real-time forecast
cds_era5              yes     Copernicus CDS — ERA5 reanalysis via API
gfs                   no      NOAA GFS — global forecast, free via AWS S3
open_meteo            no      Open-Meteo — free multi-model NWP API
zarr                  no      Zarr store — local or cloud (GCS/S3)
custom                no      Custom adapter — map any format/variable names
```

---

## 2. Ingesting Local NetCDF Data

The simplest data source is a local NetCDF file containing ERA5 reanalysis fields. Below is a tested example showing how to initialize and load a local NetCDF source:

```python
--8<-- "tests/doc_examples/test_adapters.py:imports"

--8<-- "tests/doc_examples/test_adapters.py:era5_adapter"
```

The adapter verifies that the file contains variables corresponding to:
- Geopotential (`z`)
- Specific Humidity (`q`)
- Temperature (`t`)
- $U$-component of Wind (`u`)
- $V$-component of Wind (`v`)
- Vertical Velocity (`w`)

---

## 3. Custom Variable Mappings

If your input data comes from an unstructured NetCDF or Zarr file with non-standard variable names, use the `custom` adapter to explicitly map the dataset's variables to the names expected by the model.

Below is a tested example showing how to map variables from a custom file:

```python
--8<-- "tests/doc_examples/test_adapters.py:custom_adapter"
```

### Advanced Custom Mapping Options
The `CustomAdapter` constructor accepts several parameters to help normalize non-standard files:
- `level_dim`: Name of the pressure-level dimension (default `"level"`).
- `lat_dim` / `lon_dim`: Names of the latitude/longitude dimensions (default `"latitude"`, `"longitude"`).
- `level_unit`: Set to `"Pa"` if pressure is in Pascals; it will automatically be divided by 100 to convert to hPa.
- `geopot_in_meters`: Set to `True` if your geopotential height is stored in meters (gpm) instead of geopotential ($m^2/s^2$); it will be multiplied by $g = 9.80665$ automatically.

---

## 4. Downloading Real-Time Forecasts

WeatherGraph provides adapters that automatically handle downloading and converting operational forecast models.

### NOAA GFS (Global Forecast System)
Ingest NOAA's real-time GFS analysis from AWS Open Data (no API key required):

```python
from weathergraph.data_sources import load_source

# Fetch GFS initialization for a specific UTC date/time
gfs_source = load_source(
    "gfs",
    date="2026-06-01 00:00",
    fxx=0,          # 0 = analysis, >0 = forecast hour lead time
    source="aws"    # Use AWS S3 mirror
)
gfs_dataset = gfs_source.load()
```
> [!TIP]
> GFS downloads GRIB2 files using the `Herbie` python package. Under the hood, this adapter converts geopotential height ($gpm$) to geopotential ($m^2/s^2$) and structures coordinates to match the ERA5 vertical grid.

### ECMWF Open Data
Retrieve real-time ECMWF Integrated Forecasting System (IFS) outputs:

```python
ecmwf_source = load_source(
    "ecmwf_open",
    date="2026-06-05",
    time=0,          # 00:00 UTC model run
    step=0,          # Lead time step in hours
    target="data/ecmwf_init.grib2"
)
ecmwf_dataset = ecmwf_source.load()
```

### Copernicus CDS (Climate Data Store)
Fetch archive-quality ERA5 reanalysis fields using the Copernicus CDS API:

```python
cds_source = load_source(
    "cds_era5",
    date="2024-01-01",
    time="12:00",
    target="data/cds_init.nc"
)
cds_dataset = cds_source.load()
```
> [!IMPORTANT]
> The `cds_era5` adapter requires a free registration at [Copernicus CDS](https://cds.climate.copernicus.eu/) and a configured `~/.cdsapirc` file on your host machine.