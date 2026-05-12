"""
weathergraph.data_sources
==========================

Pluggable data-source adapters for the WeatherGraph engine.

Each adapter exposes a single ``load()`` method that returns an
``xarray.Dataset`` conforming to the current reference-model input contract:

  Variables  : z, q, t, u, v, w
  Levels (hPa): 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
    Grid        : 181 × 360  (1° ERA5 lat/lon for the current reference artifact)
  Coords      : latitude  (-90 … 90),  longitude (0 … 359),  level

Adapters that fetch remote data accept keyword arguments in ``__init__``
(API keys, date/time, etc.) and perform the download lazily in ``load()``.

Quick start
-----------
List all available sources::

    from weathergraph.data_sources import list_sources
    list_sources()

Pick one and load data::

    from weathergraph.data_sources import load_source

    # ERA5 local NetCDF (default, no installation needed)
    adapter = load_source("era5_netcdf", path="data/era5_archives/init.nc")

    # ECMWF open-data forecast  (free, no API key)
    adapter = load_source("ecmwf_open", date="2024-01-01")

    # Copernicus CDS ERA5 reanalysis  (free registration + API key)
    adapter = load_source("cds_era5", date="2024-01-01")

    # NOAA GFS via AWS Open Data  (free, no key)
    adapter = load_source("gfs", date="2024-01-01 00:00")

    # Open-Meteo multi-model NWP  (free, no key, single-point)
    adapter = load_source("open_meteo", latitude=51.5, longitude=-0.1)

    # Custom file with renamed variables  (any format)
    adapter = load_source(
        "custom",
        path="forecast.nc",
        variable_map={"z": "geopotential", "t": "air_temperature"},
    )

    ds = adapter.load()   # → xr.Dataset with z, q, t, u, v, w
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr


# ── Model contract ─────────────────────────────────────────────────────────────

MODEL_LEVELS: List[int] = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
MODEL_VARS:   List[str] = ["z", "q", "t", "u", "v", "w"]

_G = 9.80665  # standard gravity [m s⁻²]


# ── Abstract base ──────────────────────────────────────────────────────────────

class DataSourceAdapter(ABC):
    """Base class for all data-source adapters.

    Sub-classes must implement :meth:`load` and set the class-level
    ``name``, ``description``, and ``requires_auth`` attributes.
    """

    #: Short registry identifier (used with :func:`load_source`).
    name: str = ""

    #: Human-readable one-line description shown by :func:`list_sources`.
    description: str = ""

    #: Whether this source requires an API key or paid subscription.
    requires_auth: bool = False

    @abstractmethod
    def load(self) -> xr.Dataset:
        """Return an xarray.Dataset with variables z, q, t, u, v, w
        indexed by a ``level`` coordinate (hPa) plus lat/lon."""

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _require(package: str, install_hint: str) -> Any:
        """Import *package* or raise a clear install error."""
        import importlib
        try:
            return importlib.import_module(package)
        except ImportError:
            raise ImportError(
                f"Package '{package}' is required for this adapter.\n"
                f"Install it with:  {install_hint}"
            ) from None


# ── ERA5 local NetCDF ──────────────────────────────────────────────────────────

class ERA5NetCDFAdapter(DataSourceAdapter):
    """Load a local ERA5 NetCDF file (existing default behaviour).

    Parameters
    ----------
    path : str
        Path to a NetCDF file containing ERA5 pressure-level data with
        variables ``z``, ``q``, ``t``, ``u``, ``v``, ``w`` and a
        ``level`` coordinate in hPa.

    Examples
    --------
    >>> adapter = ERA5NetCDFAdapter("data/era5_archives/katrina_20050823_init.nc")
    >>> ds = adapter.load()
    """

    name          = "era5_netcdf"
    description   = "Local ERA5 NetCDF file (Copernicus CDS download or ECMWF archive)"
    requires_auth = False

    def __init__(self, path: str):
        self.path = path

    def load(self) -> xr.Dataset:
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"ERA5 NetCDF file not found: {self.path}\n"
                "Download ERA5 reanalysis from https://cds.climate.copernicus.eu/"
            )
        return xr.open_dataset(self.path)


# ── ECMWF Open Data (free, no key) ────────────────────────────────────────────

class ECMWFOpenDataAdapter(DataSourceAdapter):
    """Fetch the latest ECMWF open-data forecast — free, no authentication.

    Uses the ``ecmwf-opendata`` Python library which downloads GRIB2 slices
    from ECMWF's public HTTPS endpoint (updated twice daily).

    Install::

        pip install ecmwf-opendata cfgrib eccodes

    Parameters
    ----------
    date : str, optional
        Forecast initialisation date ``"YYYY-MM-DD"`` (default: today).
    time : int, optional
        Model run in UTC hours: 0 or 12 (default: 0).
    step : int, optional
        Forecast lead time in hours (0 = analysis, default 0).
    target : str, optional
        Local path where the downloaded GRIB2 file is saved
        (default: ``"/tmp/ecmwf_open.grib2"``).

    Examples
    --------
    >>> adapter = ECMWFOpenDataAdapter(date="2024-06-01", step=24)
    >>> ds = adapter.load()
    """

    name          = "ecmwf_open"
    description   = "ECMWF Open Data — free real-time forecast, no API key required"
    requires_auth = False

    def __init__(
        self,
        date:   Optional[str] = None,
        time:   int           = 0,
        step:   int           = 0,
        target: str           = "/tmp/ecmwf_open.grib2",
    ):
        self.date   = date
        self.time   = time
        self.step   = step
        self.target = target

    def load(self) -> xr.Dataset:
        ecmwf_od = self._require("ecmwf.opendata", "pip install ecmwf-opendata")
        self._require("cfgrib",                    "pip install cfgrib eccodes")

        client = ecmwf_od.Client()
        request: Dict[str, Any] = {
            "type":     "fc",
            "step":     self.step,
            "param":    ["z", "q", "t", "u", "v", "w"],
            "levtype":  "pl",
            "levelist": MODEL_LEVELS,
        }
        if self.date:
            request["date"] = self.date
        if self.time is not None:
            request["time"] = self.time

        client.retrieve(request=request, target=self.target)
        return xr.open_dataset(self.target, engine="cfgrib")


# ── Copernicus CDS / ECMWF (registered, API key) ─────────────────────────────

class CopernicusCDSAdapter(DataSourceAdapter):
    """Download ERA5 reanalysis from Copernicus CDS using ``cdsapi``.

    Requires a free CDS account at https://cds.climate.copernicus.eu/ and a
    ``~/.cdsapirc`` file, or the environment variables ``CDSAPI_URL`` and
    ``CDSAPI_KEY``.

    Install::

        pip install cdsapi

    Parameters
    ----------
    date : str
        Date string ``"YYYY-MM-DD"`` (e.g. ``"2024-01-01"``).
    time : str, optional
        Hour string ``"HH:00"`` (default: ``"00:00"``).
    target : str, optional
        Local path for the downloaded NetCDF file.
    dataset : str, optional
        CDS dataset name (default: ``"reanalysis-era5-pressure-levels"``).

    Examples
    --------
    >>> adapter = CopernicusCDSAdapter(date="2024-01-01", time="12:00")
    >>> ds = adapter.load()
    """

    name          = "cds_era5"
    description   = "Copernicus CDS — ERA5 reanalysis via API key (free registration)"
    requires_auth = True

    def __init__(
        self,
        date:    str,
        time:    str = "00:00",
        target:  str = "/tmp/cds_era5.nc",
        dataset: str = "reanalysis-era5-pressure-levels",
    ):
        self.date    = date
        self.time    = time
        self.target  = target
        self.dataset = dataset

    def load(self) -> xr.Dataset:
        cdsapi = self._require("cdsapi", "pip install cdsapi")

        c = cdsapi.Client()
        c.retrieve(
            self.dataset,
            {
                "product_type": "reanalysis",
                "variable": [
                    "geopotential",
                    "specific_humidity",
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "vertical_velocity",
                ],
                "pressure_level": [str(lv) for lv in MODEL_LEVELS],
                "year":  self.date[:4],
                "month": self.date[5:7],
                "day":   self.date[8:10],
                "time":  self.time,
                "format": "netcdf",
            },
            self.target,
        )
        ds = xr.open_dataset(self.target)
        # CDS returns short names (z, q, t, u, v, w) in ERA5 — no rename needed
        return ds


# ── NOAA GFS (free, public) ───────────────────────────────────────────────────

class GFSAdapter(DataSourceAdapter):
    """Download a NOAA GFS analysis or forecast from AWS Open Data.

    Uses ``Herbie`` for GRIB2 access. Install::

        pip install herbie-data

    Parameters
    ----------
    date : str
        Forecast run date/time ``"YYYY-MM-DD HH:MM"``
        (e.g. ``"2024-01-01 00:00"``).
    fxx : int, optional
        Forecast hour (0 = analysis, default 0).
    source : str, optional
        Data mirror: ``"aws"`` (default), ``"nomads"``, ``"google"``,
        ``"azure"``.

    Notes
    -----
    GFS uses geopotential *height* (m). This adapter automatically converts
    it to geopotential (m²/s²) by multiplying by g = 9.80665.

    Examples
    --------
    >>> adapter = GFSAdapter(date="2024-01-01 00:00", fxx=6)
    >>> ds = adapter.load()
    """

    name          = "gfs"
    description   = "NOAA GFS — global forecast, free via AWS S3 Open Data"
    requires_auth = False

    # GFS GRIB2 short name → model variable
    _VAR_MAP: Dict[str, str] = {
        "gh":   "z",   # geopotential height  (convert → geopotential via ×g)
        "spfh": "q",   # specific humidity
        "t":    "t",   # temperature
        "u":    "u",   # u-wind
        "v":    "v",   # v-wind
        "w":    "w",   # vertical velocity (Pa/s)
    }

    def __init__(self, date: str, fxx: int = 0, source: str = "aws"):
        self.date   = date
        self.fxx    = fxx
        self.source = source

    def load(self) -> xr.Dataset:
        herbie_mod = self._require("herbie", "pip install herbie-data")
        Herbie = herbie_mod.Herbie

        level_pattern = "|".join(str(lv) for lv in MODEL_LEVELS)
        datasets: List[xr.Dataset] = []

        for gfs_var, model_var in self._VAR_MAP.items():
            H = Herbie(self.date, model="gfs", fxx=self.fxx, source=self.source)
            regex = f":{gfs_var.upper()}:(?:{level_pattern}) mb:"
            try:
                ds_var = H.xarray(regex)
                src_name = list(ds_var.data_vars)[0]
                datasets.append(ds_var.rename({src_name: model_var}))
            except Exception:
                # Variable not available at this step — skip gracefully
                pass

        merged = xr.merge(datasets)

        # GFS geopotential height (m) → geopotential (m²/s²)
        if "z" in merged:
            merged["z"] = merged["z"] * _G

        return merged


# ── Open-Meteo (free, no key) ─────────────────────────────────────────────────

class OpenMeteoAdapter(DataSourceAdapter):
    """Fetch pressure-level forecast data from Open-Meteo (free, no key).

    https://open-meteo.com/ — a global NWP aggregator supporting ECMWF IFS,
    GFS, ICON, and others.

    Parameters
    ----------
    latitude : float
        Point latitude in degrees (−90 … 90).
    longitude : float
        Point longitude in degrees (0 … 359 or −180 … 180).
    forecast_days : int, optional
        Forecast horizon in days (1–16, default 10).
    model : str, optional
        NWP model: ``"best_match"`` (default), ``"ecmwf_ifs025"``,
        ``"gfs_seamless"``, ``"icon_seamless"``.

    Notes
    -----
    Open-Meteo provides single-point forecasts. For global grid runs,
    use :class:`ERA5NetCDFAdapter` or :class:`GFSAdapter` instead.

    Examples
    --------
    >>> adapter = OpenMeteoAdapter(latitude=51.5, longitude=-0.1)
    >>> ds = adapter.load()
    """

    name          = "open_meteo"
    description   = "Open-Meteo — free multi-model NWP API, no authentication needed"
    requires_auth = False

    _API_VARS: Dict[str, str] = {
        "geopotential":     "z",
        "specific_humidity":"q",
        "temperature":      "t",
        "windspeed_u":      "u",
        "windspeed_v":      "v",
        "vertical_velocity":"w",
    }

    def __init__(
        self,
        latitude:      float,
        longitude:     float,
        forecast_days: int = 10,
        model:         str = "best_match",
    ):
        self.latitude      = latitude
        self.longitude     = longitude
        self.forecast_days = forecast_days
        self.model         = model

    def load(self) -> xr.Dataset:
        import json
        import urllib.request

        hourly_vars = ",".join(
            f"{api_var}_{lv}hPa"
            for api_var in self._API_VARS
            for lv in MODEL_LEVELS
        )
        params = (
            f"latitude={self.latitude}"
            f"&longitude={self.longitude}"
            f"&forecast_days={self.forecast_days}"
            f"&hourly={hourly_vars}"
            f"&models={self.model}"
        )
        url = f"https://api.open-meteo.com/v1/forecast?{params}"

        req = urllib.request.Request(url, headers={"User-Agent": "weathergraph/1.0"})
        with urllib.request.urlopen(req) as resp:  # nosec — public HTTPS endpoint
            data = json.loads(resp.read())

        hourly = data["hourly"]
        arrays: Dict[str, xr.DataArray] = {}
        for api_name, model_var in self._API_VARS.items():
            level_data, valid_levels = [], []
            for lv in MODEL_LEVELS:
                key = f"{api_name}_{lv}hPa"
                if key in hourly:
                    level_data.append(hourly[key])
                    valid_levels.append(lv)
            if level_data:
                arrays[model_var] = xr.DataArray(
                    np.array(level_data, dtype=np.float32),
                    dims=["level", "time"],
                    coords={"level": valid_levels, "time": hourly["time"]},
                )

        return xr.Dataset(arrays)


# ── Custom / constructor adapter ───────────────────────────────────────────────

class CustomAdapter(DataSourceAdapter):
    """
    "Constructor" adapter — map *any* file or dataset to the model contract.

    Supports every file format readable by xarray (NetCDF, GRIB2 via cfgrib,
    Zarr, HDF5) as well as raw NumPy arrays. Supply a ``variable_map`` to
    translate your variable names to the engine's expected names
    (z, q, t, u, v, w).

    Parameters
    ----------
    path : str, optional
        Path to the source file. Omit when passing ``data`` directly.
    data : xr.Dataset or np.ndarray, optional
        Pre-loaded dataset or a raw ``float32[1, nodes, 78]`` array for a
        compatible ONNX artifact.
        When given, ``path`` and ``format`` are ignored.
    format : str, optional
        File format hint when ``path`` is given:
        ``"netcdf4"`` (default), ``"zarr"``, ``"grib2"``, ``"hdf5"``.
    variable_map : dict, optional
        Mapping ``{model_name: source_name}`` for variable renaming.
        Example: ``{"z": "geopotential", "t": "air_temperature"}``.
        Omit variables that already have the correct names.
    level_dim : str, optional
        Name of the pressure-level dimension in your file
        (default: ``"level"``).
    lat_dim : str, optional
        Name of the latitude dimension (default: ``"latitude"``).
    lon_dim : str, optional
        Name of the longitude dimension (default: ``"longitude"``).
    level_unit : str, optional
        Unit of pressure levels: ``"hPa"`` (default) or ``"Pa"``
        (values are divided by 100 automatically).
    geopot_in_meters : bool, optional
        Set ``True`` when your geopotential field is in metres (gpm) rather
        than m²/s² — it is multiplied by g = 9.80665.
    engine : str, optional
        xarray engine override (e.g. ``"cfgrib"``, ``"h5netcdf"``).

    Examples
    --------
    From a local NetCDF with renamed variables::

        adapter = CustomAdapter(
            path="my_nwp_output.nc",
            variable_map={
                "z": "geopotential",
                "q": "specific_humidity",
                "t": "air_temperature",
                "u": "eastward_wind",
                "v": "northward_wind",
                "w": "vertical_velocity",
            },
            level_dim="pressure",
            lat_dim="lat",
            lon_dim="lon",
        )
        ds = adapter.load()

    From a GRIB2 file::

        adapter = CustomAdapter(
            path="icon_forecast.grb2",
            format="grib2",
            variable_map={"z": "z_height", "t": "t2"},
            geopot_in_meters=True,
        )

    From a pre-built NumPy array (bypasses all file I/O)::

        raw = np.load("my_state.npy")   # shape [1, nodes, 78]
        adapter = CustomAdapter(data=raw)

    From a plain dict schema (useful for config-file driven pipelines)::

        schema = {
            "source": "forecast.nc",
            "variable_map": {"z": "geopotential", "t": "temperature"},
            "level_dim": "pressure",
        }
        adapter = CustomAdapter.from_schema(schema)
    """

    name          = "custom"
    description   = "Custom adapter — map any format / variable names to the model contract"
    requires_auth = False

    def __init__(
        self,
        path:             Optional[str]              = None,
        data:             Optional[Any]              = None,
        format:           str                        = "netcdf4",
        variable_map:     Optional[Dict[str, str]]   = None,
        level_dim:        str                        = "level",
        lat_dim:          str                        = "latitude",
        lon_dim:          str                        = "longitude",
        level_unit:       str                        = "hPa",
        geopot_in_meters: bool                       = False,
        engine:           Optional[str]              = None,
    ):
        if path is None and data is None:
            raise ValueError("Provide either 'path' or 'data'.")
        self.path             = path
        self.data             = data
        self.format           = format
        self.variable_map     = variable_map or {}
        self.level_dim        = level_dim
        self.lat_dim          = lat_dim
        self.lon_dim          = lon_dim
        self.level_unit       = level_unit
        self.geopot_in_meters = geopot_in_meters
        self.engine           = engine

    def load(self) -> xr.Dataset:
        # ── Raw array / pre-loaded Dataset path ───────────────────────────────
        if self.data is not None:
            if isinstance(self.data, np.ndarray):
                return xr.Dataset({"_raw": xr.DataArray(self.data)})
            return self.data  # assume xr.Dataset

        # ── File path ─────────────────────────────────────────────────────────
        open_kw: Dict[str, Any] = {}

        if self.engine:
            open_kw["engine"] = self.engine
        elif self.format == "grib2":
            self._require("cfgrib", "pip install cfgrib eccodes")
            open_kw["engine"] = "cfgrib"
        elif self.format == "hdf5":
            open_kw["engine"] = "h5netcdf"

        if self.format == "zarr":
            ds = xr.open_zarr(self.path)
        else:
            ds = xr.open_dataset(self.path, **open_kw)

        # ── Rename dimensions ──────────────────────────────────────────────────
        dim_rename: Dict[str, str] = {}
        if self.level_dim != "level":
            dim_rename[self.level_dim] = "level"
        if self.lat_dim != "latitude":
            dim_rename[self.lat_dim] = "latitude"
        if self.lon_dim != "longitude":
            dim_rename[self.lon_dim] = "longitude"
        if dim_rename:
            ds = ds.rename(dim_rename)

        # ── Rename variables ───────────────────────────────────────────────────
        if self.variable_map:
            # variable_map = {model_name: source_name}
            # we need to rename source_name → model_name
            reverse = {src: dst for dst, src in self.variable_map.items()}
            present = {k: v for k, v in reverse.items() if k in ds}
            if present:
                ds = ds.rename(present)

        # ── Unit corrections ───────────────────────────────────────────────────
        if self.level_unit == "Pa" and "level" in ds.coords:
            ds = ds.assign_coords(level=ds.coords["level"] / 100.0)

        if self.geopot_in_meters and "z" in ds:
            ds["z"] = ds["z"] * _G

        return ds

    @classmethod
    def from_schema(cls, schema: Dict[str, Any]) -> "CustomAdapter":
        """Construct a :class:`CustomAdapter` from a plain dictionary.

        The schema accepts all ``__init__`` keyword arguments plus a
        ``"source"`` key as an alias for ``path``.

        Parameters
        ----------
        schema : dict
            Configuration dictionary. Example::

                {
                    "source":       "data/my_forecast.nc",
                    "format":       "netcdf4",
                    "variable_map": {"z": "geopotential", "t": "temperature"},
                    "level_dim":    "pressure",
                    "lat_dim":      "lat",
                    "lon_dim":      "lon",
                }

        Returns
        -------
        CustomAdapter
        """
        kw = dict(schema)
        if "source" in kw:
            kw["path"] = kw.pop("source")
        return cls(**kw)


# ── Registry ───────────────────────────────────────────────────────────────────

#: All built-in adapter classes, keyed by their short name.
REGISTRY: Dict[str, type] = {
    "era5_netcdf": ERA5NetCDFAdapter,
    "ecmwf_open":  ECMWFOpenDataAdapter,
    "cds_era5":    CopernicusCDSAdapter,
    "gfs":         GFSAdapter,
    "open_meteo":  OpenMeteoAdapter,
    "custom":      CustomAdapter,
}


def list_sources() -> None:
    """Print a table of all registered data-source adapters."""
    col = 20
    print(f"{'Name':<{col}}  {'Auth?':<6}  Description")
    print("-" * 70)
    for name, cls in REGISTRY.items():
        auth = "yes" if cls.requires_auth else "no "
        print(f"{name:<{col}}  {auth:<6}  {cls.description}")


def load_source(name: str, **kwargs) -> DataSourceAdapter:
    """Instantiate a data-source adapter by registry name.

    Parameters
    ----------
    name : str
        Registry key — use :func:`list_sources` to see all options.
    **kwargs
        Forwarded to the adapter's ``__init__``.

    Returns
    -------
    DataSourceAdapter
        A ready-to-use adapter. Call ``.load()`` to fetch the dataset.

    Raises
    ------
    KeyError
        If *name* is not in the registry.

    Examples
    --------
    >>> adapter = load_source("era5_netcdf", path="init.nc")
    >>> adapter = load_source("ecmwf_open", date="2024-01-01", step=24)
    >>> adapter = load_source("cds_era5", date="2024-01-01")
    >>> adapter = load_source("gfs", date="2024-01-01 00:00", fxx=6)
    >>> adapter = load_source("open_meteo", latitude=51.5, longitude=-0.1)
    >>> adapter = load_source("custom", path="forecast.nc",
    ...     variable_map={"z": "geopotential"})
    >>> ds = adapter.load()
    """
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown data source '{name}'.\n"
            f"Available names: {list(REGISTRY.keys())}\n"
            "For a custom source use load_source('custom', ...) or CustomAdapter."
        )
    return REGISTRY[name](**kwargs)
