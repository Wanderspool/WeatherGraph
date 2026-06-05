"""
Test suite for CF-1.11 convention compliance in WeatherGraph output.

Validates that all variables and coordinates in forecast output carry
the correct standard_name, units, and axis attributes required for
interoperability with MetPy, xCDAT, and the broader climate-science
Python ecosystem.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weathergraph.cf_meta import (
    CF_COORDINATE_ATTRS,
    CF_VARIABLE_ATTRS,
    build_cf_dataset,
    ensure_pressure_order,
    inject_cf_attrs_zarr,
    _global_attrs,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
LEVEL_VARS = ["z", "q", "t", "u", "v", "w"]
LAT_COUNT, LON_COUNT = 181, 360
NODE_COUNT = LAT_COUNT * LON_COUNT


@pytest.fixture
def dummy_trajectory():
    """Produce a 3-step trajectory of shape [1, nodes, 78]."""
    rng = np.random.default_rng(42)
    return [rng.standard_normal((1, NODE_COUNT, 78)).astype(np.float32) for _ in range(3)]


@pytest.fixture
def lat_lon():
    lat = np.linspace(90.0, -90.0, LAT_COUNT, dtype=np.float64)
    lon = np.linspace(0.0, 359.0, LON_COUNT, dtype=np.float64)
    return lat, lon


@pytest.fixture
def cf_dataset(dummy_trajectory, lat_lon):
    lat, lon = lat_lon
    return build_cf_dataset(
        trajectory=dummy_trajectory,
        lat=lat,
        lon=lon,
        levels=LEVELS,
        level_vars=LEVEL_VARS,
    )


# ── CF Variable Attribute Tests ───────────────────────────────────────────────

class TestCFVariableAttributes:
    """Every data variable must carry CF-1.11 standard attributes."""

    def test_all_variables_present(self, cf_dataset):
        for var in LEVEL_VARS:
            assert var in cf_dataset, f"Variable '{var}' missing from CF dataset"

    @pytest.mark.parametrize("var_name", LEVEL_VARS)
    def test_standard_name(self, cf_dataset, var_name):
        expected = CF_VARIABLE_ATTRS[var_name]["standard_name"]
        assert cf_dataset[var_name].attrs["standard_name"] == expected

    @pytest.mark.parametrize("var_name", LEVEL_VARS)
    def test_units(self, cf_dataset, var_name):
        expected = CF_VARIABLE_ATTRS[var_name]["units"]
        assert cf_dataset[var_name].attrs["units"] == expected

    @pytest.mark.parametrize("var_name", LEVEL_VARS)
    def test_long_name(self, cf_dataset, var_name):
        expected = CF_VARIABLE_ATTRS[var_name]["long_name"]
        assert cf_dataset[var_name].attrs["long_name"] == expected


# ── CF Coordinate Attribute Tests ─────────────────────────────────────────────

class TestCFCoordinateAttributes:
    """Coordinates must carry standard_name, units, and axis."""

    def test_latitude_attrs(self, cf_dataset):
        lat_attrs = cf_dataset["lat"].attrs
        assert lat_attrs["standard_name"] == "latitude"
        assert lat_attrs["units"] == "degrees_north"
        assert lat_attrs["axis"] == "Y"

    def test_longitude_attrs(self, cf_dataset):
        lon_attrs = cf_dataset["lon"].attrs
        assert lon_attrs["standard_name"] == "longitude"
        assert lon_attrs["units"] == "degrees_east"
        assert lon_attrs["axis"] == "X"

    def test_level_attrs(self, cf_dataset):
        level_attrs = cf_dataset["level"].attrs
        assert level_attrs["standard_name"] == "air_pressure"
        assert level_attrs["units"] == "hPa"
        assert level_attrs["axis"] == "Z"

    def test_time_attrs(self, cf_dataset):
        time_attrs = cf_dataset["time"].attrs
        assert time_attrs["standard_name"] == "time"
        assert time_attrs["axis"] == "T"


# ── Global Attributes ────────────────────────────────────────────────────────

class TestGlobalAttributes:
    """Dataset-level attributes must declare CF convention version."""

    def test_conventions(self, cf_dataset):
        assert cf_dataset.attrs["Conventions"] == "CF-1.11"

    def test_source(self, cf_dataset):
        assert "WeatherGraph" in cf_dataset.attrs["source"]

    def test_references(self, cf_dataset):
        assert "Keisler" in cf_dataset.attrs["references"]


# ── Dataset Shape Tests ──────────────────────────────────────────────────────

class TestDatasetShape:
    """Output dimensions must be (time, level, lat, lon)."""

    def test_dims(self, cf_dataset):
        for var in LEVEL_VARS:
            assert cf_dataset[var].dims == ("time", "level", "lat", "lon")

    def test_time_length(self, cf_dataset):
        assert len(cf_dataset["time"]) == 3  # 3-step trajectory

    def test_level_length(self, cf_dataset):
        assert len(cf_dataset["level"]) == len(LEVELS)

    def test_lat_length(self, cf_dataset):
        assert len(cf_dataset["lat"]) == LAT_COUNT

    def test_lon_length(self, cf_dataset):
        assert len(cf_dataset["lon"]) == LON_COUNT


# ── Pressure Level Ordering ──────────────────────────────────────────────────

class TestPressureOrdering:
    """ensure_pressure_order must correctly sort levels."""

    def test_ascending_order(self, cf_dataset):
        ds = ensure_pressure_order(cf_dataset, ascending=True)
        levels = ds["level"].values
        assert np.all(np.diff(levels) > 0), "Levels should be ascending (50 → 1000)"

    def test_descending_order(self, cf_dataset):
        ds = ensure_pressure_order(cf_dataset, ascending=False)
        levels = ds["level"].values
        assert np.all(np.diff(levels) < 0), "Levels should be descending (1000 → 50)"

    def test_no_level_dim_passthrough(self):
        """Datasets without a level dim should pass through unchanged."""
        ds = xr.Dataset({"x": xr.DataArray([1, 2, 3])})
        result = ensure_pressure_order(ds)
        xr.testing.assert_identical(ds, result)


# ── Zarr CF Injection ────────────────────────────────────────────────────────

class TestZarrCFInjection:
    """inject_cf_attrs_zarr must add CF metadata to single-variable datasets."""

    def test_injects_variable_attrs(self):
        ds = xr.Dataset(
            {"t": xr.DataArray(
                np.zeros((1, 10, 10), dtype=np.float32),
                dims=["time", "lat", "lon"],
            )},
        )
        result = inject_cf_attrs_zarr(ds, "t", 850)
        assert result["t"].attrs["standard_name"] == "air_temperature"
        assert result["t"].attrs["units"] == "K"
        assert result["t"].attrs["level_hPa"] == 850

    def test_injects_global_attrs(self):
        ds = xr.Dataset(
            {"z": xr.DataArray(np.zeros((1, 5, 5)), dims=["time", "lat", "lon"])}
        )
        result = inject_cf_attrs_zarr(ds, "z", 500)
        assert result.attrs["Conventions"] == "CF-1.11"


# ── _global_attrs helper ─────────────────────────────────────────────────────

class TestGlobalAttrsHelper:
    """_global_attrs must return a well-formed dictionary."""

    def test_keys(self):
        attrs = _global_attrs()
        assert "Conventions" in attrs
        assert "source" in attrs
        assert "history" in attrs

    def test_custom_source(self):
        attrs = _global_attrs(source="test_source")
        assert attrs["source"] == "test_source"
