import pickle
import jax
import jax.numpy as jnp
import haiku as hk
import jraph
import jax2onnx
import onnx
# Upstream reference-model package name remains keisler_2022; keep these imports
# for compatibility while presenting WeatherGraph as the product surface.
from keisler_2022.runner import Runner
from keisler_2022.config import Config

import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert Keisler 2022 to ONNX")
    parser.add_argument("--resolution", type=float, default=1.0, help="Grid resolution in degrees")
    parser.add_argument("--output", type=str, default="weather_gnn.onnx", help="Output ONNX file path")
    args = parser.parse_args()

    print(f"Initializing reference-model runner for resolution {args.resolution}...")
    config = Config()
    runner = Runner(verbose=True, config=config)
    
    # Calculate grid shape based on resolution
    lat_steps = round(180.0 / args.resolution)
    lon_steps = round(360.0 / args.resolution)

    n_node_era5 = (int(lat_steps) + 1) * int(lon_steps)
    n_channels = 78
    
    print(f"Grid nodes: {n_node_era5} (lat: {lat_steps + 1}, lon: {lon_steps})")

    # Create dummy initial data
    dummy_data = jnp.zeros((n_node_era5, n_channels))
    
    # We need solar and doy for the steps
    # Just use zeros for initialization
    n_steps = 1
    dummy_solar = jnp.zeros((n_node_era5, n_steps))
    dummy_doy = jnp.zeros((n_node_era5, n_steps))
    
    # Setup graphs
    graphs = {
        "e": runner.static_graphs["e"].jraph(),
        "p": runner.static_graphs["p"].jraph(),
        "d": runner.static_graphs["d"].jraph(),
    }
    
    graphs["e"].nodes["data"] = runner.init_set(dummy_data)
    graphs["e"].nodes["all_solar"] = runner.init_set(dummy_solar)
    graphs["e"].nodes["all_doy"] = runner.init_set(dummy_doy)
    
    print("Loading parameters...")
    # Load model parameters
    with open(runner.config.resolve_artifact(runner.config.data.weights_file), "rb") as fp:
        params = pickle.load(fp)
    params = hk.data_structures.to_immutable_dict(params)
    
    # The function we want to export
    # We'll export a single step
    def model_fn(graphs, i_time):
        return runner.transformed.apply(params, graphs, i_time)

    print("Converting to ONNX...")
    # jax2onnx requires the function and example inputs
    # Note: Jraph GraphsTuple are named tuples, which jax2onnx might not handle directly
    # We might need to flatten them
    
    # For now, let's try a direct conversion
    # We might need to use dynamic_shapes if nodes/edges count varies (though here it's static)
    try:
        onnx_model = jax2onnx.to_onnx(
            model_fn,
            (graphs, 0),
            # dynamic_shapes=... 
        )
        
        print(f"Saving ONNX model to {args.output}...")
        onnx.save(onnx_model, args.output)
        print(f"Success! Saved to {args.output}")
    except Exception as e:
        print(f"Failed to convert: {e}")

if __name__ == "__main__":
    main()
