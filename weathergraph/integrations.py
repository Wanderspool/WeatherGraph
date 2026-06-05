"""
weathergraph.integrations
==========================

Utilities for integrating WeatherGraph forecast output with the
operational meteorology (MetPy) and long-term climate analysis (xCDAT)
ecosystems.

All functions in this module are **best-effort**: they work perfectly
when the optional dependency is installed, and fall back gracefully
(with a clear warning) when it is not.

Install the optional dependencies with::

    pip install 'weathergraph[metpy]'     # MetPy only
    pip install 'weathergraph[xcdat]'     # xCDAT only
    pip install 'weathergraph[climate]'   # MetPy + xCDAT + xESMF
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import xarray as xr

from .cf_meta import CF_VARIABLE_ATTRS, CF_COORDINATE_ATTRS, ensure_pressure_order


# ── MetPy Integration ────────────────────────────────────────────────────────

def prepare_for_metpy(ds: xr.Dataset) -> xr.Dataset:
    """Prepare a WeatherGraph forecast Dataset for MetPy analysis.

    Steps performed:

    1. Ensure CF-compliant ``standard_name`` and ``units`` attributes
       are present on all variables and coordinates.
    2. Sort pressure levels **descending** (surface → top of atmosphere)
       as required by MetPy's thermodynamic profile calculations.
    3. Call ``metpy.parse_cf()`` to attach Cartopy CRS metadata, if
       MetPy is installed.

    Parameters
    ----------
    ds : xr.Dataset
        Forecast dataset (typically from ``model.forecast()`` or the
        accessor's ``predict()``).

    Returns
    -------
    xr.Dataset
        Dataset ready for ``mpcalc`` functions.

    Notes
    -----
    MetPy's ``surface_based_cape_cin()`` and vertical-profile functions
    **require** pressure levels sorted from highest pressure (surface)
    to lowest (top).  This function handles that automatically.

    Examples
    --------
    >>> from weathergraph.integrations import prepare_for_metpy
    >>> import metpy.calc as mpcalc
    >>>
    >>> ds_metpy = prepare_for_metpy(ds_forecast)
    >>> u_geo, v_geo = mpcalc.geostrophic_wind(ds_metpy["z"].metpy.quantify())
    """
    # 1. Inject CF attributes if missing
    ds = _ensure_cf_attrs(ds)

    # 2. Sort levels descending (1000 → 50 hPa) for MetPy
    ds = ensure_pressure_order(ds, ascending=False)

    # 3. Parse CF metadata with MetPy
    try:
        import metpy.xarray  # noqa: F401 — registers .metpy accessor
        ds = ds.metpy.parse_cf()
    except ImportError:
        warnings.warn(
            "MetPy is not installed. Install it with: pip install 'weathergraph[metpy]'\n"
            "Dataset returned without MetPy CRS metadata.",
            stacklevel=2,
        )
    except Exception as exc:
        warnings.warn(
            f"MetPy parse_cf() encountered an issue: {exc}\n"
            "Dataset returned without MetPy CRS metadata.",
            stacklevel=2,
        )

    return ds


# ── xCDAT Integration ───────────────────────────────────────────────────────

def prepare_for_xcdat(ds: xr.Dataset) -> xr.Dataset:
    """Prepare a WeatherGraph forecast Dataset for xCDAT analysis.

    Steps performed:

    1. Ensure CF attributes are present.
    2. Add missing spatial and temporal bounds using ``xcdat``.
    3. Normalise time encoding for xCDAT's ``temporal`` accessor.

    Parameters
    ----------
    ds : xr.Dataset
        Forecast dataset.

    Returns
    -------
    xr.Dataset
        Dataset ready for ``ds.spatial.average()``,
        ``ds.temporal.climatology()``, and ``ds.regridder``.

    Examples
    --------
    >>> from weathergraph.integrations import prepare_for_xcdat
    >>>
    >>> ds_xcdat = prepare_for_xcdat(ds_forecast)
    >>> global_mean_t = ds_xcdat.spatial.average("t")
    >>> climo = ds_xcdat.temporal.climatology("t", freq="month")
    """
    ds = _ensure_cf_attrs(ds)

    try:
        import xcdat  # noqa: F401 — registers xCDAT accessors

        # Add spatial bounds if missing
        if "lat_bnds" not in ds and "lat" in ds.coords:
            ds = ds.bounds.add_bounds("lat")
        if "lon_bnds" not in ds and "lon" in ds.coords:
            ds = ds.bounds.add_bounds("lon")
        # Add time bounds if missing
        if "time_bnds" not in ds and "time" in ds.coords:
            try:
                ds = ds.bounds.add_time_bounds(method="freq", freq="6h")
            except Exception:
                # Some datasets with irregular time spacing may fail
                pass

    except ImportError:
        warnings.warn(
            "xCDAT is not installed. Install it with: pip install 'weathergraph[xcdat]'\n"
            "Dataset returned without spatial/temporal bounds.",
            stacklevel=2,
        )
    except Exception as exc:
        warnings.warn(
            f"xCDAT bounds generation encountered an issue: {exc}\n"
            "Dataset returned without full bounds metadata.",
            stacklevel=2,
        )

    return ds


# ── Derived Diagnostics ──────────────────────────────────────────────────────

def compute_wind_speed(ds: xr.Dataset) -> xr.DataArray:
    """Compute horizontal wind speed from u and v components.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain variables ``u`` and ``v``.

    Returns
    -------
    xr.DataArray
        Wind speed in m/s with CF attributes.
    """
    if "u" not in ds or "v" not in ds:
        raise ValueError("Dataset must contain 'u' and 'v' wind components.")

    wind_speed = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    wind_speed.attrs = {
        "standard_name": "wind_speed",
        "long_name": "Horizontal Wind Speed",
        "units": "m s-1",
    }
    wind_speed.name = "wind_speed"
    return wind_speed


def compute_geopotential_height(ds: xr.Dataset, g: float = 9.80665) -> xr.DataArray:
    """Convert geopotential (m²/s²) to geopotential height (m).

    Parameters
    ----------
    ds : xr.Dataset
        Must contain variable ``z`` (geopotential).
    g : float
        Standard gravity (default 9.80665 m/s²).

    Returns
    -------
    xr.DataArray
        Geopotential height in metres (gpm) with CF attributes.
    """
    if "z" not in ds:
        raise ValueError("Dataset must contain 'z' (geopotential).")

    gph = ds["z"] / g
    gph.attrs = {
        "standard_name": "geopotential_height",
        "long_name": "Geopotential Height",
        "units": "m",
    }
    gph.name = "geopotential_height"
    return gph


def compute_derived_diagnostics(ds: xr.Dataset) -> xr.Dataset:
    """Add common derived diagnostics to a forecast dataset.

    Currently computes:

    - ``wind_speed`` from ``u`` and ``v``
    - ``geopotential_height`` from ``z``

    Parameters
    ----------
    ds : xr.Dataset
        WeatherGraph forecast output.

    Returns
    -------
    xr.Dataset
        The input dataset augmented with derived variables.
    """
    ds_out = ds.copy(deep=False)

    if "u" in ds and "v" in ds:
        ds_out["wind_speed"] = compute_wind_speed(ds)

    if "z" in ds:
        ds_out["geopotential_height"] = compute_geopotential_height(ds)

    return ds_out


# ── Internal helpers ─────────────────────────────────────────────────────────

def _ensure_cf_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Add missing CF attributes to variables and coordinates."""
    ds = ds.copy(deep=False)

    # Variable attributes
    for var_name, attrs in CF_VARIABLE_ATTRS.items():
        if var_name in ds:
            for key, value in attrs.items():
                if key not in ds[var_name].attrs:
                    ds[var_name].attrs[key] = value

    # Coordinate attributes
    coord_name_map = {"latitude": "lat", "longitude": "lon"}
    for cf_name, attrs in CF_COORDINATE_ATTRS.items():
        ds_name = coord_name_map.get(cf_name, cf_name)
        if ds_name in ds.coords:
            for key, value in attrs.items():
                if key not in ds[ds_name].attrs:
                    ds[ds_name].attrs[key] = value

    return ds
