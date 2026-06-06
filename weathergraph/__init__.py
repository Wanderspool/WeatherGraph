from .model import WeatherGraphModel, EnsembleStats
from .data_sources import (
    DataSourceAdapter,
    ERA5NetCDFAdapter,
    ECMWFOpenDataAdapter,
    CopernicusCDSAdapter,
    GFSAdapter,
    OpenMeteoAdapter,
    ZarrStoreAdapter,
    CustomAdapter,
    REGISTRY,
    list_sources,
    load_source,
)
from .cf_meta import (
    build_cf_dataset,
    ensure_pressure_order,
    CF_VARIABLE_ATTRS,
    CF_COORDINATE_ATTRS,
)

# Register the Xarray accessor so that `ds.weathergraph.*` is available
# as soon as `import weathergraph` is executed.
from . import accessor  # noqa: F401 — side-effect import

__version__ = "0.1.0"
