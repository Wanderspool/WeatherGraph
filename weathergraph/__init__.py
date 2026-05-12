from .model import WeatherGraphModel
from .data_sources import (
    DataSourceAdapter,
    ERA5NetCDFAdapter,
    ECMWFOpenDataAdapter,
    CopernicusCDSAdapter,
    GFSAdapter,
    OpenMeteoAdapter,
    CustomAdapter,
    REGISTRY,
    list_sources,
    load_source,
)

__version__ = "0.1.0"
