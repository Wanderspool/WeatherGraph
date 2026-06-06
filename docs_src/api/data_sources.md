# weathergraph.data_sources

The `weathergraph.data_sources` module manages ingestion adapters and remote meteorological client interfaces, exposing them through a common registration factory pattern.

---

## Technical Overview & Lifecycle

All data adapters inherit from the abstract base class `DataSourceAdapter` and are registered in a global module registry. The workflow for loading data sources is as follows:
1.  **Instantiation**: The factory function `load_source(name, **kwargs)` looks up the registry key `name` and constructs the registered subclass.
2.  **Configuration**: Parameters are validated by the adapter's constructor (e.g. bounding boxes, dates, variables list).
3.  **Loading**: Calling `.load()` initiates the transfer. Depending on the adapter, this may read local files on disk (NetCDF/Zarr) or download slices from remote APIs (ECMWF, Copernicus CDS, GFS, Open-Meteo).
4.  **Dataset Return**: Returns an `xarray.Dataset` standardized to coordinates expected by the model engine.

---

## Writing a Custom Data Adapter

You can extend WeatherGraph's ingestion capabilities by implementing the `DataSourceAdapter` interface. Register your adapter subclass using the `@register_source` decorator:

```python
import weathergraph as wg
import xarray as xr
from weathergraph.data_sources import DataSourceAdapter, register_source

@register_source("my_local_sensor")
class LocalSensorAdapter(DataSourceAdapter):
    """Loads historical weather data from a local JSON sensor log."""
    
    def __init__(self, sensor_id: str, log_path: str, **kwargs):
        super().__init__(**kwargs)
        self.sensor_id = sensor_id
        self.log_path = log_path

    def load(self) -> xr.Dataset:
        # Load your custom file
        import pandas as pd
        df = pd.read_json(self.log_path)
        
        # Convert to xarray.Dataset and ensure coordinates match model expectations
        ds = df.to_xarray()
        return ds
```

Once registered, users can instantiate and load it using the high-level API:
```python
# The factory locates the custom registered source
adapter = wg.load_source(
    "my_local_sensor",
    sensor_id="station_012",
    log_path="data/sensors/station_012.json"
)
dataset = adapter.load()
```

---

## Configuration & Credentials

### 1. Copernicus Climate Data Store (CDS)
The `cds_era5` adapter queries the Copernicus API. Ensure you have the `cdsapi` python client configured and credentials written to `~/.cdsapirc`:
```ini
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
```

### 2. NOAA GFS (Amazon S3)
The `gfs` adapter streams real-time forecasts directly from NOAA's public bucket on AWS S3. It uses `s3fs` under the hood. To access it, you do not need AWS credentials; the adapter is pre-configured to establish anonymous connection sessions.

---

## Factory Functions

::: weathergraph.data_sources.load_source
::: weathergraph.data_sources.list_sources

---

## Abstract Base Class

::: weathergraph.data_sources.DataSourceAdapter
    options:
      show_source: true

---

## Ingestion Adapters

::: weathergraph.data_sources.ERA5NetCDFAdapter
::: weathergraph.data_sources.ECMWFOpenDataAdapter
::: weathergraph.data_sources.CopernicusCDSAdapter
::: weathergraph.data_sources.GFSAdapter
::: weathergraph.data_sources.OpenMeteoAdapter
::: weathergraph.data_sources.ZarrStoreAdapter
::: weathergraph.data_sources.CustomAdapter
