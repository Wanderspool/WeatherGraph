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
    import keisler_cpp_backend
except ImportError:
    # If not in the package, try local build dir (for dev)
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))
    import keisler_cpp_backend

class GraphWeatherModel:
    def __init__(self, model_path, weights_dir):
        """
        Initialize the high-performance C++ engine.
        """
        self.engine = keisler_cpp_backend.KeislerEngine(model_path)
        
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

    def predict_one_step(self, ds):
        """Single step prediction."""
        input_data = self._prepare_input(ds)
        # Note: In our current specialized ONNX graph, we pass means/stds as separate inputs 
        # for demonstration, but they could be hardcoded as constants too.
        # Here we pass them if the engine expects them.
        return self.engine.predict(input_data)
