import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
import dask

# Force synchronous Dask for memory stability on e2.micro
dask.config.set(scheduler='synchronous')

# Ensure the core module is discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
try:
    import weathergraph_backend
except ImportError:
    # If not in the package, try local build dir (for dev)
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
    import weathergraph_backend

class WeatherGraphModel:
    def __init__(self, model_path, weights_dir):
        """
        Initialize the high-performance C++ engine.
        """
        self.engine = weathergraph_backend.WeatherGraphEngine(model_path)
        
        # Load normalization constants (used if not in-graph)
        self.means = np.load(os.path.join(weights_dir, "means.npy")).astype(np.float32)
        self.stds = np.load(os.path.join(weights_dir, "stds.npy")).astype(np.float32)
        
        # ERA5 Variable ordering contract (Total 78 channels)
        # Order: z, q, t, u, v, w at each of the 13 levels
        self.level_vars = ['z', 'q', 't', 'u', 'v', 'w']
        self.levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    def _prepare_input(self, ds):
        """
        Extract variables in strict scientific order and flatten to [1, Nodes, 78].
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
        input_tensor = np.stack(data_slices, axis=-1).astype(np.float32)
        
        # Return with batch dim [1, Nodes, 78]
        return input_tensor[np.newaxis, ...]

    def forecast(self, initial_ds, steps=12):
        """
        Perform a 6-hour auto-regressive rollout.
        Returns a list of numpy arrays representing the atmospheric state at each step.
        """
        results = []
        
        # Initial preparation
        input_buffer = self._prepare_input(initial_ds)
        
        for i in range(steps):
            # ZERO-COPY call to C++ engine
            # The engine already has normalization/denormalization in-graph
            output_buffer = self.engine.predict(input_buffer)
            
            # The output of the GNN is the predicted change (delta) or state at T+6h
            # In Keisler 2022, it predicts the next state directly.
            # We reuse the output as the next input.
            input_buffer = output_buffer
            
            results.append(output_buffer)
            
        return results

    def forecast_export(self, initial_ds, steps=40, output_path="forecast",
                        fmt="netcdf4", t0=None):
        """
        Run an autoregressive rollout and export results in a structured
        scientific format, split by variable and pressure level.

        Parameters
        ----------
        initial_ds  : xarray.Dataset  — ERA5 initial state.
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

        Output layout (npz)
        ───────────────────
        <output_path>.npz
          trajectory  — float32[steps+1, 71042, 78]   raw model output
          lat         — float32[181]
          lon         — float32[360]
          levels      — int32[13]
          variables   — str[6]
        """
        import warnings

        trajectory = self.forecast(initial_ds, steps=steps)
        # Shape: (steps, 1, 71042, 78) — ERA5 1° grid is first 181×360=65160 nodes
        arr = np.stack([t[0] for t in trajectory], axis=0)  # (steps, 71042, 78)

        # ── NPZ — raw export, no reshape ──────────────────────────────────────
        if fmt == "npz":
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

        # ── Grid reshape: ERA5 1° nodes → (181, 360) ─────────────────────────
        # Nodes 0..65159 correspond to the ERA5 1° grid (lat outer, lon inner).
        era5_arr = arr[:, :65160, :]          # (steps, 65160, 78)
        era5_arr = era5_arr.reshape(steps, 181, 360, 78)

        lat  = np.linspace(90, -90, 181, dtype=np.float64)
        lon  = np.linspace(0, 359, 360, dtype=np.float64)
        times = pd.date_range(
            start=t0 if t0 else "2000-01-01",
            periods=steps,
            freq="6h",
        )

        n_vars   = len(self.level_vars)   # 6
        n_levels = len(self.levels)       # 13
        os.makedirs(output_path, exist_ok=True)

        for li, level in enumerate(self.levels):
            for vi, var in enumerate(self.level_vars):
                ch = li * n_vars + vi          # channel index in 78-dim axis
                data = era5_arr[:, :, :, ch]   # (steps, 181, 360)

                ds_out = xr.Dataset(
                    {var: xr.DataArray(
                        data.astype(np.float32),
                        dims=["time", "lat", "lon"],
                        coords={"time": times, "lat": lat, "lon": lon},
                        attrs={"level_hPa": int(level)},
                    )},
                    attrs={"WeatherGraph": "forecast", "steps": steps, "level_hPa": int(level)},
                )

                filename = f"{var}_{level}hPa"
                if fmt == "zarr":
                    out = os.path.join(output_path, f"{filename}.zarr")
                    ds_out.to_zarr(out, mode="w")
                else:  # netcdf4
                    out = os.path.join(output_path, f"{filename}.nc")
                    ds_out.to_netcdf(out)

        print(f"[export] {fmt} → {output_path}/  ({n_levels * n_vars} files, {steps} steps)")

    def predict_one_step(self, ds):
        """Single step prediction."""
        input_data = self._prepare_input(ds)
        # Note: In our current specialized ONNX graph, we pass means/stds as separate inputs 
        # for demonstration, but they could be hardcoded as constants too.
        # Here we pass them if the engine expects them.
        return self.engine.predict(input_data)
