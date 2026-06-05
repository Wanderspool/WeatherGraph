"""
Test suite for weathergraph.integrations module.

Validates MetPy/xCDAT preparation utilities, derived diagnostics
(wind speed, geopotential height), and the ZarrStoreAdapter.
"""

import numpy as np
import pytest
import xarray as xr

from weathergraph.integrations import (
    compute_derived_diagnostics,
    compute_geopotential_height,
    compute_wind_speed,
    prepare_for_metpy,
    prepare_for_xcdat,
    _ensure_cf_attrs,
)
from weathergraph.cf_meta import CF_VARIABLE_ATTRS
from weathergraph.data_sources import ZarrStoreAdapter, REGISTRY


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def forecast_ds():
    """Minimal forecast-like Dataset with u, v, z, t, q, w."""
    rng = np.random.default_rng(42)
    levels = [850, 500, 200]
    lat = np.linspace(90, -90, 10)
    lon = np.linspace(0, 350, 20)

    data_vars = {}
    for var in ["z", "q", "t", "u", "v", "w"]:
        data_vars[var] = xr.DataArray(
            rng.standard_normal((2, 3, 10, 20)).astype(np.float32),
            dims=["time", "level", "lat", "lon"],
            coords={
                "level": levels,
                "lat": lat,
                "lon": lon,
            },
        )

    return xr.Dataset(data_vars)


# ── _ensure_cf_attrs ─────────────────────────────────────────────────────────

class TestEnsureCFAttrs:
    """CF attributes must be injected into variables and coordinates."""

    def test_adds_standard_name(self, forecast_ds):
        result = _ensure_cf_attrs(forecast_ds)
        assert result["t"].attrs["standard_name"] == "air_temperature"

    def test_adds_units(self, forecast_ds):
        result = _ensure_cf_attrs(forecast_ds)
        assert result["u"].attrs["units"] == "m s-1"

    def test_does_not_overwrite_existing(self, forecast_ds):
        forecast_ds["t"].attrs["units"] = "degC"  # non-standard
        result = _ensure_cf_attrs(forecast_ds)
        assert result["t"].attrs["units"] == "degC", "Should not overwrite existing"

    def test_coordinate_attrs(self, forecast_ds):
        result = _ensure_cf_attrs(forecast_ds)
        assert result["lat"].attrs["units"] == "degrees_north"
        assert result["lon"].attrs["units"] == "degrees_east"


# ── prepare_for_metpy ────────────────────────────────────────────────────────

class TestPrepareForMetPy:
    """MetPy preparation must sort levels descending and add CF attrs."""

    def test_levels_descending(self, forecast_ds):
        result = prepare_for_metpy(forecast_ds)
        levels = result["level"].values
        assert levels[0] > levels[-1], "Levels should be descending for MetPy"

    def test_cf_attrs_present(self, forecast_ds):
        result = prepare_for_metpy(forecast_ds)
        assert "standard_name" in result["t"].attrs

    def test_handles_no_level_dim(self):
        ds = xr.Dataset({"x": xr.DataArray([1, 2, 3])})
        result = prepare_for_metpy(ds)
        assert isinstance(result, xr.Dataset)


# ── prepare_for_xcdat ────────────────────────────────────────────────────────

class TestPrepareForXcdat:
    """xCDAT preparation must add CF attrs and handle gracefully."""

    def test_cf_attrs_present(self, forecast_ds):
        result = prepare_for_xcdat(forecast_ds)
        assert "standard_name" in result["t"].attrs

    def test_returns_dataset(self, forecast_ds):
        result = prepare_for_xcdat(forecast_ds)
        assert isinstance(result, xr.Dataset)


# ── Derived Diagnostics ──────────────────────────────────────────────────────

class TestWindSpeed:
    """Wind speed from u and v components."""

    def test_wind_speed_computation(self, forecast_ds):
        ws = compute_wind_speed(forecast_ds)
        assert ws.name == "wind_speed"
        assert ws.attrs["units"] == "m s-1"
        assert ws.attrs["standard_name"] == "wind_speed"

    def test_wind_speed_values(self):
        ds = xr.Dataset({
            "u": xr.DataArray([3.0], dims=["x"]),
            "v": xr.DataArray([4.0], dims=["x"]),
        })
        ws = compute_wind_speed(ds)
        np.testing.assert_allclose(ws.values, [5.0])

    def test_missing_component_raises(self):
        ds = xr.Dataset({"u": xr.DataArray([1.0])})
        with pytest.raises(ValueError, match="must contain 'u' and 'v'"):
            compute_wind_speed(ds)


class TestGeopotentialHeight:
    """Geopotential height from geopotential."""

    def test_computation(self):
        g = 9.80665
        ds = xr.Dataset({"z": xr.DataArray([g * 5500.0], dims=["x"])})
        gph = compute_geopotential_height(ds)
        np.testing.assert_allclose(gph.values, [5500.0], rtol=1e-5)

    def test_attrs(self):
        ds = xr.Dataset({"z": xr.DataArray([100.0], dims=["x"])})
        gph = compute_geopotential_height(ds)
        assert gph.attrs["standard_name"] == "geopotential_height"
        assert gph.attrs["units"] == "m"

    def test_missing_z_raises(self):
        ds = xr.Dataset({"t": xr.DataArray([300.0])})
        with pytest.raises(ValueError, match="must contain 'z'"):
            compute_geopotential_height(ds)


class TestDerivedDiagnostics:
    """compute_derived_diagnostics adds wind_speed and geopotential_height."""

    def test_adds_wind_speed(self, forecast_ds):
        result = compute_derived_diagnostics(forecast_ds)
        assert "wind_speed" in result

    def test_adds_geopotential_height(self, forecast_ds):
        result = compute_derived_diagnostics(forecast_ds)
        assert "geopotential_height" in result

    def test_preserves_original_vars(self, forecast_ds):
        result = compute_derived_diagnostics(forecast_ds)
        for var in ["z", "q", "t", "u", "v", "w"]:
            assert var in result


# ── ZarrStoreAdapter ─────────────────────────────────────────────────────────

class TestZarrStoreAdapter:
    """Zarr adapter must be in registry and handle local stores."""

    def test_registered(self):
        assert "zarr" in REGISTRY
        assert REGISTRY["zarr"] is ZarrStoreAdapter

    def test_init(self):
        adapter = ZarrStoreAdapter(store="test.zarr")
        assert adapter.name == "zarr"
        assert adapter.store == "test.zarr"
        assert adapter.consolidated is True

    def test_load_local_zarr(self, tmp_path):
        """Write a small Zarr and read it back."""
        zarr_path = str(tmp_path / "test.zarr")
        ds = xr.Dataset({
            "t": xr.DataArray(
                np.random.randn(2, 5, 5).astype(np.float32),
                dims=["time", "lat", "lon"],
            )
        })
        ds.to_zarr(zarr_path, consolidated=True)

        adapter = ZarrStoreAdapter(store=zarr_path)
        loaded = adapter.load()
        assert "t" in loaded
        np.testing.assert_array_equal(loaded["t"].values, ds["t"].values)

    def test_requires_auth_false(self):
        assert ZarrStoreAdapter.requires_auth is False
