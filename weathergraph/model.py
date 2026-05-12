import json
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
from contextlib import ExitStack
from pathlib import Path
import dask

# Force synchronous Dask to avoid worker-pool memory spikes on CPU deployments.
dask.config.set(scheduler='synchronous')

# Ensure the core module is discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
try:
    import weathergraph_backend
except ImportError:
    # If not in the package, try local build dir (for dev)
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
    import weathergraph_backend

class _TiledEngineAdapter:
    def __init__(self, model):
        self._model = model

    def output_shape(self):
        return list(self._model.output_shape)

    def cpu_mem_arena_enabled(self):
        return not self._model.disable_cpu_mem_arena

    def mem_pattern_enabled(self):
        return not self._model.disable_mem_pattern

    def predict(self, input_data):
        return self._model._predict_tiled(input_data)

class WeatherGraphModel:
    def __init__(self,
                 model_path,
                 weights_dir,
                 intra_op_threads=1,
                 disable_cpu_mem_arena=False,
                 disable_mem_pattern=False,
                 spatial_tiling=False,
                 tile_bundle_path=None):
        """
        Initialize the high-performance C++ engine.

        Parameters
        ----------
        model_path : str
            Path to the ONNX artifact.
        weights_dir : str
            Directory containing normalization statistics.
        intra_op_threads : int, optional
            ONNX Runtime intra-op thread count. Keep this small on
            memory-constrained machines; raise it on workstation-class CPUs
            when throughput matters more than absolute memory minimization.
        disable_cpu_mem_arena : bool, optional
            Disable ONNX Runtime's CPU arena allocator to reduce reserved RSS
            at the cost of additional allocation overhead.
        disable_mem_pattern : bool, optional
            Disable ONNX Runtime memory-pattern reuse. Useful for low-memory
            runs that prioritize lower reservation over peak throughput.
        spatial_tiling : bool, optional
            Enable exact graph-aware tiled inference. This requires a tile bundle
            with partition metadata and per-tile ONNX artifacts.
        tile_bundle_path : str | None, optional
            Path to a tile-bundle manifest or directory containing one.
        """
        self.model_path = model_path
        self.intra_op_threads = intra_op_threads
        self.disable_cpu_mem_arena = disable_cpu_mem_arena
        self.disable_mem_pattern = disable_mem_pattern
        self.spatial_tiling = spatial_tiling or bool(tile_bundle_path)
        self.tile_bundle_path = tile_bundle_path
        self.tile_bundle = None
        self._tile_engines = {}
        self.runtime_options = {
            "intra_op_threads": intra_op_threads,
            "disable_cpu_mem_arena": disable_cpu_mem_arena,
            "disable_mem_pattern": disable_mem_pattern,
            "spatial_tiling": self.spatial_tiling,
            "tile_bundle_path": tile_bundle_path,
        }

        if self.spatial_tiling:
            if not tile_bundle_path:
                raise ValueError(
                    "spatial_tiling requires tile_bundle_path with exact graph-aware tile metadata."
                )
            self.tile_bundle = self._load_tile_bundle(tile_bundle_path)
            self.output_shape = tuple(self.tile_bundle["global_output_shape"])
            self.engine = _TiledEngineAdapter(self)
            self.cpu_mem_arena_enabled = not disable_cpu_mem_arena
            self.mem_pattern_enabled = not disable_mem_pattern
        else:
            self.engine = self._create_engine(model_path)
            self.output_shape = tuple(self.engine.output_shape())
            self.cpu_mem_arena_enabled = getattr(
                self.engine,
                "cpu_mem_arena_enabled",
                lambda: not disable_cpu_mem_arena,
            )()
            self.mem_pattern_enabled = getattr(
                self.engine,
                "mem_pattern_enabled",
                lambda: not disable_mem_pattern,
            )()
        if len(self.output_shape) != 3 or self.output_shape[0] != 1 or self.output_shape[-1] != 78:
            raise ValueError(
                "WeatherGraphModel requires an autoregressive ONNX artifact with output shape [1, nodes, 78]. "
                "Prototype latent-output exporters are not supported by this wrapper."
            )
        
        # Load normalization constants (used if not in-graph)
        self.means = np.load(os.path.join(weights_dir, "means.npy")).astype(np.float32)
        self.stds = np.load(os.path.join(weights_dir, "stds.npy")).astype(np.float32)
        
        # ERA5 Variable ordering contract (Total 78 channels)
        # Order: z, q, t, u, v, w at each of the 13 levels
        self.level_vars = ['z', 'q', 't', 'u', 'v', 'w']
        self.levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    def _create_engine(self, model_path):
        return weathergraph_backend.WeatherGraphEngine(
            model_path,
            intra_op_threads=self.intra_op_threads,
            disable_cpu_mem_arena=self.disable_cpu_mem_arena,
            disable_mem_pattern=self.disable_mem_pattern,
        )

    def _load_tile_bundle(self, tile_bundle_path):
        manifest_path = Path(tile_bundle_path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Tile bundle manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        bundle_root = manifest_path.parent
        global_input_shape = tuple(manifest.get("global_input_shape", manifest.get("global_output_shape", ())))
        global_output_shape = tuple(manifest.get("global_output_shape", global_input_shape))
        if len(global_input_shape) != 3 or len(global_output_shape) != 3:
            raise ValueError("Tile bundle must define 3D global_input_shape and global_output_shape.")
        if global_input_shape[0] != 1 or global_output_shape[0] != 1:
            raise ValueError("Tile bundle currently supports batch size 1 only.")
        if global_input_shape[2] != 78 or global_output_shape[2] != 78:
            raise ValueError("Tile bundle must use WeatherGraph-compatible 78-channel state tensors.")
        if global_input_shape[1] != global_output_shape[1]:
            raise ValueError("Tile bundle must preserve the global node count between input and output.")

        tile_specs = []
        covered_nodes = np.zeros(global_output_shape[1], dtype=bool)
        for index, tile in enumerate(manifest.get("tiles", [])):
            input_indices = np.asarray(
                np.load(bundle_root / tile["input_indices_path"]),
                dtype=np.int64,
            )
            output_indices = np.asarray(
                np.load(bundle_root / tile["output_indices_path"]),
                dtype=np.int64,
            )
            if input_indices.ndim != 1 or output_indices.ndim != 1:
                raise ValueError("Tile bundle indices must be 1D arrays.")
            if output_indices.size == 0:
                raise ValueError("Tile bundle output indices cannot be empty.")
            if np.any(output_indices < 0) or np.any(output_indices >= global_output_shape[1]):
                raise ValueError("Tile bundle output indices must lie within the global node range.")
            if np.unique(output_indices).size != output_indices.size:
                raise ValueError("Tile bundle output indices must be unique within each tile.")
            if covered_nodes[output_indices].any():
                raise ValueError("Tile bundle output coverage must not overlap across tiles.")
            covered_nodes[output_indices] = True
            tile_specs.append(
                {
                    "id": tile.get("id", f"tile_{index:03d}"),
                    "model_path": str(bundle_root / tile["model_path"]),
                    "input_indices": input_indices,
                    "output_indices": output_indices,
                }
            )

        if not tile_specs:
            raise ValueError("Tile bundle must define at least one tile.")
        if not covered_nodes.all():
            raise ValueError("Tile bundle output indices must cover every global node exactly once.")

        return {
            "manifest_path": str(manifest_path),
            "global_input_shape": global_input_shape,
            "global_output_shape": global_output_shape,
            "tiles": tile_specs,
        }

    def _get_tile_engine(self, model_path):
        if model_path not in self._tile_engines:
            self._tile_engines[model_path] = self._create_engine(model_path)
        return self._tile_engines[model_path]

    def _predict_tiled(self, input_buffer):
        input_buffer = np.ascontiguousarray(input_buffer, dtype=np.float32)
        expected_input_shape = tuple(self.tile_bundle["global_input_shape"])
        if input_buffer.shape != expected_input_shape:
            raise ValueError(
                f"Tiled inference expected input shape {expected_input_shape}, got {input_buffer.shape}."
            )

        output_buffer = np.empty(self.tile_bundle["global_output_shape"], dtype=np.float32)
        for tile in self.tile_bundle["tiles"]:
            tile_input = np.ascontiguousarray(input_buffer[:, tile["input_indices"], :])
            tile_output = self._get_tile_engine(tile["model_path"]).predict(tile_input)

            expected_tile_shape = (1, tile["output_indices"].shape[0], self.output_shape[-1])
            if tile_output.shape != expected_tile_shape:
                raise ValueError(
                    f"Tile '{tile['id']}' produced shape {tile_output.shape}, expected {expected_tile_shape}."
                )
            output_buffer[:, tile["output_indices"], :] = tile_output

        return output_buffer

    # ── Data-source helpers ────────────────────────────────────────────────────

    def _resolve_dataset(self, source):
        """Accept either an xr.Dataset *or* a DataSourceAdapter instance.

        Parameters
        ----------
        source : xr.Dataset or DataSourceAdapter
            When a :class:`~weathergraph.data_sources.DataSourceAdapter` is
            passed its ``.load()`` method is called automatically.

        Returns
        -------
        xr.Dataset
        """
        try:
            from .data_sources import DataSourceAdapter
            if isinstance(source, DataSourceAdapter):
                return source.load()
        except ImportError:
            pass
        return source  # assume xr.Dataset

    def _prepare_input(self, ds):
        """
        Extract variables in strict scientific order and flatten to
        ``float32[1, nodes, 78]``.

        The 78-channel ordering is part of the WeatherGraph reference-model
        contract. The node count depends on the ONNX artifact and upstream grid.
        """
        # Step 1: Selection and ordering
        # Efficiently extract all required slices
        data_slices = []
        for level in self.levels:
            for var in self.level_vars:
                # Get the 2D grid and flatten it
                # Assumes ds is already loaded for the specific time slice
                val = ds[var].sel(level=level).values.flatten()
                data_slices.append(val)
        
        # Step 2: Stack to [Nodes, 78]
        # Transpose to get [Nodes, Channels]
        input_tensor = np.ascontiguousarray(
            np.stack(data_slices, axis=-1).astype(np.float32)
        )
        
        # Return with batch dim [1, Nodes, 78]
        return np.ascontiguousarray(input_tensor[np.newaxis, ...])

    def iter_forecast(self, initial_ds, steps=12):
        """
        Yield each 6-hour autoregressive forecast step as soon as it is produced.

        This is the memory-efficient rollout path for workstation-scale runs.
        Use :meth:`forecast` only when you truly need the full trajectory
        materialized in RAM.
        """
        input_buffer = self._prepare_input(self._resolve_dataset(initial_ds))

        for _ in range(steps):
            output_buffer = self.engine.predict(input_buffer)
            input_buffer = output_buffer
            yield output_buffer

    def _iter_reference_grid_forecast(self, initial_ds, steps):
        """Yield `(step_index, era5_step)` for the current 1° reference artifact."""
        for step_index, output_buffer in enumerate(self.iter_forecast(initial_ds, steps=steps)):
            if output_buffer.shape[0] != 1 or output_buffer.shape[2] != 78:
                raise ValueError(
                    "Reference export requires model outputs shaped as [1, nodes, 78]."
                )
            if output_buffer.shape[1] < 65160:
                raise ValueError(
                    "Reference export requires at least 65160 ERA5 nodes for the current 1° grid path."
                )

            era5_step = output_buffer[0, :65160, :].reshape(181, 360, 78)
            yield step_index, era5_step

    def _stream_netcdf_export(self, output_path, times, lat, lon, steps, step_iterator):
        import netCDF4

        os.makedirs(output_path, exist_ok=True)
        time_units = f"hours since {times[0].strftime('%Y-%m-%d %H:%M:%S')}"
        time_values = np.arange(steps, dtype=np.float64) * 6.0

        with ExitStack() as stack:
            writers = {}
            for li, level in enumerate(self.levels):
                for vi, var in enumerate(self.level_vars):
                    filename = os.path.join(output_path, f"{var}_{level}hPa.nc")
                    dataset = stack.enter_context(netCDF4.Dataset(filename, mode="w", format="NETCDF4"))
                    dataset.createDimension("time", steps)
                    dataset.createDimension("lat", len(lat))
                    dataset.createDimension("lon", len(lon))

                    time_var = dataset.createVariable("time", "f8", ("time",))
                    time_var.units = time_units
                    time_var.calendar = "proleptic_gregorian"
                    time_var[:] = time_values

                    lat_var = dataset.createVariable("lat", "f8", ("lat",))
                    lon_var = dataset.createVariable("lon", "f8", ("lon",))
                    lat_var[:] = lat
                    lon_var[:] = lon

                    data_var = dataset.createVariable(
                        var,
                        "f4",
                        ("time", "lat", "lon"),
                        zlib=True,
                        complevel=1,
                    )
                    data_var.setncattr("level_hPa", int(level))
                    dataset.setncattr("WeatherGraph", "forecast")
                    dataset.setncattr("steps", int(steps))
                    dataset.setncattr("level_hPa", int(level))
                    writers[(li, vi)] = data_var

            n_vars = len(self.level_vars)
            for step_index, era5_step in step_iterator:
                for li, _level in enumerate(self.levels):
                    for vi, _var in enumerate(self.level_vars):
                        ch = li * n_vars + vi
                        writers[(li, vi)][step_index, :, :] = era5_step[:, :, ch].astype(np.float32)

    def _stream_zarr_export(self, output_path, times, lat, lon, step_iterator):
        os.makedirs(output_path, exist_ok=True)
        n_vars = len(self.level_vars)

        for step_index, era5_step in step_iterator:
            time_value = [times[step_index]]
            for li, level in enumerate(self.levels):
                for vi, var in enumerate(self.level_vars):
                    ch = li * n_vars + vi
                    data = era5_step[:, :, ch][np.newaxis, :, :].astype(np.float32)
                    ds_out = xr.Dataset(
                        {var: xr.DataArray(
                            data,
                            dims=["time", "lat", "lon"],
                            coords={"time": time_value, "lat": lat, "lon": lon},
                            attrs={"level_hPa": int(level)},
                        )},
                        attrs={"WeatherGraph": "forecast", "level_hPa": int(level)},
                    )
                    out = os.path.join(output_path, f"{var}_{level}hPa.zarr")
                    if step_index == 0:
                        ds_out.to_zarr(out, mode="w")
                    else:
                        ds_out.to_zarr(out, mode="a", append_dim="time")

    def forecast(self, initial_ds, steps=12):
        """
        Perform a 6-hour auto-regressive rollout.

        Parameters
        ----------
        initial_ds : xr.Dataset or DataSourceAdapter
            Initial atmospheric state. Pass a
            :class:`~weathergraph.data_sources.DataSourceAdapter` to load
            data from any supported source automatically.
        steps : int
            Number of 6-hour steps.

        Returns a list of numpy arrays representing the atmospheric state
        at each step.
        """
        return list(self.iter_forecast(initial_ds, steps=steps))

    def forecast_export(self, initial_ds, steps=40, output_path="forecast",
                        fmt="netcdf4", t0=None):
        """
        Run an autoregressive rollout and export results in a structured
        scientific format, split by variable and pressure level.

        Parameters
        ----------
        initial_ds  : xr.Dataset or DataSourceAdapter
                      ERA5 initial state or any supported adapter instance.
        steps       : int             — Number of 6-hour forecast steps.
        output_path : str             — Output directory (netcdf4/zarr) or
                                        file path (.npz).
        fmt         : str             — Output format: "netcdf4" | "zarr" | "npz"
        t0          : str | None      — ISO-8601 start time (e.g. "2024-01-01T00").
                                        Used to build the time axis.

            Output layout (netcdf4 / zarr)
        ──────────────────────────────
        <output_path>/
          z_50hPa.nc   z_100hPa.nc  ... (one file per variable × level)
          q_50hPa.nc   ...
          t_50hPa.nc   ...
          ...
        Each file has dimensions (time, lat, lon) with a proper time axis.

            NetCDF4 and Zarr exports stream each forecast step to disk and avoid
            materializing the full trajectory in memory.

            Output layout (npz)
        ───────────────────
        <output_path>.npz
          trajectory  — float32[steps, nodes, 78]   raw model output
          lat         — float32[181]
          lon         — float32[360]
          levels      — int32[13]
          variables   — str[6]

        The NPZ path still materializes the trajectory in memory before
        compression because the archive format is not append-friendly.
        """
        resolved_source = self._resolve_dataset(initial_ds)

        # ── NPZ — raw export, no reshape ──────────────────────────────────────
        if fmt == "npz":
            arr = np.stack([t[0] for t in self.iter_forecast(resolved_source, steps=steps)], axis=0)
            lat  = np.linspace(90, -90, 181, dtype=np.float32)
            lon  = np.linspace(0, 359, 360, dtype=np.float32)
            np.savez_compressed(
                output_path,
                trajectory=arr,
                lat=lat,
                lon=lon,
                levels=np.array(self.levels, dtype=np.int32),
                variables=np.array(self.level_vars),
            )
            print(f"[export] npz → {output_path}.npz  shape={arr.shape}")
            return

        if fmt not in {"netcdf4", "zarr"}:
            raise ValueError("fmt must be one of 'netcdf4', 'zarr', or 'npz'.")

        # ── Grid reshape: current reference artifact assumes ERA5 1° nodes ───
        # Nodes 0..65159 correspond to the ERA5 1° grid (lat outer, lon inner).
        # Keep this export path tied to the reference artifact until a generic
        # grid metadata contract is introduced.

        lat  = np.linspace(90, -90, 181, dtype=np.float64)
        lon  = np.linspace(0, 359, 360, dtype=np.float64)
        times = pd.date_range(
            start=t0 if t0 else "2000-01-01",
            periods=steps,
            freq="6h",
        )

        step_iterator = self._iter_reference_grid_forecast(resolved_source, steps=steps)
        if fmt == "zarr":
            self._stream_zarr_export(output_path, times, lat, lon, step_iterator)
        else:
            self._stream_netcdf_export(output_path, times, lat, lon, steps, step_iterator)

        print(f"[export] {fmt} → {output_path}/  ({len(self.levels) * len(self.level_vars)} files, {steps} steps, streaming)")

    def predict_one_step(self, ds):
        """Single step prediction.

        Parameters
        ----------
        ds : xr.Dataset or DataSourceAdapter
        """
        input_data = self._prepare_input(self._resolve_dataset(ds))
        # Note: In our current specialized ONNX graph, we pass means/stds as separate inputs 
        # for demonstration, but they could be hardcoded as constants too.
        # Here we pass them if the engine expects them.
        return self.engine.predict(input_data)
