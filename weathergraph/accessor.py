"""
weathergraph.accessor
======================

Custom Xarray accessor that embeds the WeatherGraph inference engine
directly into ``xarray.Dataset`` objects.

After importing ``weathergraph``, every Dataset gains a ``.weathergraph``
namespace with methods for loading model weights, running forecasts, and
preparing output for downstream climate libraries (MetPy, xCDAT).

Usage
-----
>>> import xarray as xr
>>> import weathergraph                       # registers the accessor
>>>
>>> ds = xr.open_zarr("gs://weatherbench2/datasets/era5/2023-01-01-00.zarr")
>>> ds_forecast = ds.weathergraph.predict(steps=40)
>>> ds_forecast.to_zarr("s3://my-bucket/forecast.zarr")

The accessor caches the heavy ONNX model (~2 GB) so repeated calls to
``predict()`` on the same Dataset don't reload weights from disk.

See Also
--------
:class:`weathergraph.model.WeatherGraphModel`
    Underlying engine class.
:mod:`weathergraph.cf_meta`
    CF-convention metadata injected into forecast outputs.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import xarray as xr

from .cf_meta import ensure_pressure_order


@xr.register_dataset_accessor("weathergraph")
class WeatherGraphAccessor:
    """Xarray Dataset accessor for WeatherGraph inference.

    Attributes
    ----------
    _obj : xr.Dataset
        The Dataset this accessor is attached to.
    _model : WeatherGraphModel or None
        Lazily-initialised engine instance (cached across calls).
    _model_kwargs : dict
        Keyword arguments used to construct the cached model.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self._model = None
        self._model_kwargs: dict[str, Any] = {}

    # ── Model lifecycle ───────────────────────────────────────────────────────

    def load_model(
        self,
        model_path: str = "models/weather_gnn.onnx",
        weights_dir: str = "data",
        *,
        force_reload: bool = False,
        **kwargs: Any,
    ) -> "WeatherGraphAccessor":
        """Lazily load the ONNX inference engine.

        The engine is cached on the accessor instance.  Subsequent calls
        with the same arguments return immediately.  Pass
        ``force_reload=True`` to discard the cached engine and reload.

        Parameters
        ----------
        model_path : str
            Path to the ONNX artifact.
        weights_dir : str
            Directory containing ``means.npy`` and ``stds.npy``.
        force_reload : bool
            Discard and rebuild the engine even if one is cached.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`~weathergraph.model.WeatherGraphModel`.

        Returns
        -------
        WeatherGraphAccessor
            ``self`` for method-chaining.

        Examples
        --------
        >>> ds.weathergraph.load_model(
        ...     model_path="models/weather_gnn.onnx",
        ...     weights_dir="data",
        ...     execution_provider="cuda",
        ... )
        """
        new_kwargs = {"model_path": model_path, "weights_dir": weights_dir, **kwargs}

        if self._model is not None and not force_reload and new_kwargs == self._model_kwargs:
            return self

        from .model import WeatherGraphModel

        self._model = WeatherGraphModel(**new_kwargs)
        self._model_kwargs = new_kwargs
        return self

    @property
    def model(self):
        """Return the cached model, auto-loading from env vars if needed."""
        if self._model is None:
            self.load_model(
                model_path=os.getenv("WEATHERGRAPH_ONNX_MODEL", "models/weather_gnn.onnx"),
                weights_dir=os.getenv("WEATHERGRAPH_WEIGHTS_DIR", "data"),
            )
        return self._model

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        steps: int = 40,
        *,
        model_path: Optional[str] = None,
        weights_dir: Optional[str] = None,
        as_dataset: bool = True,
        **model_kwargs: Any,
    ) -> xr.Dataset:
        """Run an autoregressive forecast and return a CF-compliant Dataset.

        This is the primary entrypoint for climatologists.  All C++
        inference, spatial tiling, and memory optimisation is hidden
        behind this single call.

        Parameters
        ----------
        steps : int
            Number of 6-hour forecast steps (40 = 10 days).
        model_path : str or None
            ONNX model path.  If ``None`` the cached / env-var model is used.
        weights_dir : str or None
            Weights directory.  If ``None`` the cached / env-var path is used.
        as_dataset : bool
            If ``True`` (default), return a full CF ``xr.Dataset``.
            If ``False``, return a raw ``list[np.ndarray]``.
        **model_kwargs
            Extra keyword arguments forwarded to
            :meth:`load_model` (e.g., ``execution_provider``).

        Returns
        -------
        xr.Dataset
            Forecast with dimensions ``(time, level, lat, lon)`` and
            CF-1.11 metadata.

        Examples
        --------
        >>> ds_forecast = ds.weathergraph.predict(steps=40)
        >>> ds_forecast["t"].sel(level=850, time="2024-01-03").plot()
        """
        if model_path or weights_dir or model_kwargs:
            self.load_model(
                model_path=model_path or os.getenv("WEATHERGRAPH_ONNX_MODEL", "models/weather_gnn.onnx"),
                weights_dir=weights_dir or os.getenv("WEATHERGRAPH_WEIGHTS_DIR", "data"),
                **model_kwargs,
            )

        mdl = self.model
        return mdl.forecast(self._obj, steps=steps, as_dataset=as_dataset)

    # ── Data preparation helpers ──────────────────────────────────────────────

    def ensure_pressure_order(self, ascending: bool = True) -> xr.Dataset:
        """Sort the pressure-level axis.

        MetPy requires levels sorted surface→top (descending pressure).
        Call with ``ascending=False`` before passing to MetPy.

        Parameters
        ----------
        ascending : bool
            ``True``: 50 → 1000 hPa (default, model order).
            ``False``: 1000 → 50 hPa (MetPy-compatible).

        Returns
        -------
        xr.Dataset
        """
        return ensure_pressure_order(self._obj, ascending=ascending)

    def prepare_for_metpy(self) -> xr.Dataset:
        """Return the Dataset prepared for MetPy analysis.

        This is a convenience wrapper that:

        1. Sorts pressure levels descending (surface → top).
        2. Calls ``metpy.parse_cf()`` if MetPy is installed.
        3. Falls back gracefully if MetPy is not available.

        Returns
        -------
        xr.Dataset
        """
        from .integrations import prepare_for_metpy

        return prepare_for_metpy(self._obj)

    def prepare_for_xcdat(self) -> xr.Dataset:
        """Return the Dataset prepared for xCDAT analysis.

        Adds spatial bounds and normalises time coordinates.

        Returns
        -------
        xr.Dataset
        """
        from .integrations import prepare_for_xcdat

        return prepare_for_xcdat(self._obj)

    # ── Export helpers ────────────────────────────────────────────────────────

    def to_zarr(self, path: str, **kwargs: Any) -> None:
        """Export the Dataset to Zarr with CF metadata.

        Parameters
        ----------
        path : str
            Zarr store path (local, ``s3://``, ``gs://``).
        **kwargs
            Forwarded to :meth:`xr.Dataset.to_zarr`.
        """
        self._obj.to_zarr(path, **kwargs)
