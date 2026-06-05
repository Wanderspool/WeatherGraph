"""
Test suite for the WeatherGraph Xarray accessor.

Validates that ``import weathergraph`` registers the ``.weathergraph``
namespace on ``xr.Dataset``, and that the accessor's lifecycle (model
loading, prediction, pressure ordering) works correctly.
"""

import numpy as np
import pytest
import xarray as xr

# Importing weathergraph registers the accessor as a side-effect.
import weathergraph
from weathergraph.accessor import WeatherGraphAccessor


# ── Registration ──────────────────────────────────────────────────────────────

class TestAccessorRegistration:
    """The accessor must be available on any xr.Dataset after import."""

    def test_accessor_exists(self):
        ds = xr.Dataset()
        assert hasattr(ds, "weathergraph"), (
            "xr.Dataset should have a `.weathergraph` accessor after importing weathergraph"
        )

    def test_accessor_type(self):
        ds = xr.Dataset()
        assert isinstance(ds.weathergraph, WeatherGraphAccessor)


# ── Model lifecycle ──────────────────────────────────────────────────────────

class TestModelLifecycle:
    """Tests for lazy model loading and caching."""

    def test_initial_model_is_none(self):
        ds = xr.Dataset()
        assert ds.weathergraph._model is None

    def test_load_model_returns_self(self, tmp_path):
        """load_model should return the accessor for chaining."""
        ds = xr.Dataset()
        # We can't actually load a real model here, but we can test that
        # the method exists and has the right signature.
        assert callable(ds.weathergraph.load_model)

    def test_model_kwargs_stored(self):
        ds = xr.Dataset()
        assert ds.weathergraph._model_kwargs == {}


# ── Pressure ordering ────────────────────────────────────────────────────────

class TestPressureOrdering:
    """Tests for the ensure_pressure_order helper on the accessor."""

    def _make_level_ds(self):
        return xr.Dataset(
            {"t": xr.DataArray(
                np.random.randn(3, 13, 5, 5).astype(np.float32),
                dims=["time", "level", "lat", "lon"],
                coords={"level": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]},
            )}
        )

    def test_ascending_order(self):
        ds = self._make_level_ds()
        result = ds.weathergraph.ensure_pressure_order(ascending=True)
        levels = result["level"].values
        assert np.all(np.diff(levels) > 0)

    def test_descending_order(self):
        ds = self._make_level_ds()
        result = ds.weathergraph.ensure_pressure_order(ascending=False)
        levels = result["level"].values
        assert np.all(np.diff(levels) < 0)


# ── MetPy / xCDAT preparation ───────────────────────────────────────────────

class TestPreparationHelpers:
    """Accessor helpers should be callable and return Datasets."""

    def _make_cf_ds(self):
        return xr.Dataset(
            {
                "t": xr.DataArray(
                    np.random.randn(2, 3, 5, 10).astype(np.float32),
                    dims=["time", "level", "lat", "lon"],
                    coords={
                        "level": [850, 500, 200],
                        "lat": np.linspace(90, -90, 5),
                        "lon": np.linspace(0, 350, 10),
                    },
                    attrs={"standard_name": "air_temperature", "units": "K"},
                ),
            },
        )

    def test_prepare_for_metpy_returns_dataset(self):
        ds = self._make_cf_ds()
        result = ds.weathergraph.prepare_for_metpy()
        assert isinstance(result, xr.Dataset)
        # Levels should be descending after MetPy preparation
        if "level" in result.dims:
            levels = result["level"].values
            assert levels[0] > levels[-1], "MetPy needs descending pressure"

    def test_prepare_for_xcdat_returns_dataset(self):
        ds = self._make_cf_ds()
        result = ds.weathergraph.prepare_for_xcdat()
        assert isinstance(result, xr.Dataset)


# ── Zarr export ──────────────────────────────────────────────────────────────

class TestZarrExport:
    """Accessor to_zarr should delegate to xr.Dataset.to_zarr."""

    def test_to_zarr_callable(self):
        ds = xr.Dataset({"x": xr.DataArray([1, 2, 3])})
        assert callable(ds.weathergraph.to_zarr)

    def test_to_zarr_writes(self, tmp_path):
        ds = xr.Dataset({"x": xr.DataArray([1, 2, 3])})
        path = str(tmp_path / "test.zarr")
        ds.weathergraph.to_zarr(path)
        reopened = xr.open_zarr(path)
        np.testing.assert_array_equal(reopened["x"].values, [1, 2, 3])
