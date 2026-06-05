import argparse
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

def convert_to_onnx(weights_file: str, output_file: str):
    print("Initializing reference-model runner...")
    config = Config()
    runner = Runner(verbose=True, config=config)
    
    # We need dummy input data to initialize the graphs
    # 1.0 degree grid: 181 latitudes, 360 longitudes
    # 6 variables, 13 levels = 78 channels
    n_node_era5 = 181 * 360
    n_channels = 78
    
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
    
    print(f"Loading parameters from {weights_file}...")
    # Load model parameters
    with open(weights_file, "rb") as fp:
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
        onnx_model = jax2onnx.from_jax(
            model_fn,
            (graphs, 0),
            # dynamic_shapes=... 
        )
        
        print(f"Saving ONNX model to {output_file}...")
        onnx.save(onnx_model, output_file)
        print("Success!")
    except Exception as e:
        print(f"Failed to convert: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert WeatherGraph Pickle weights to ONNX format.")
    parser.add_argument("--weights-file", required=True, help="Path to input .pkl weights file.")
    parser.add_argument("--output-file", required=True, help="Path to output .onnx model.")
    args = parser.parse_args()
    
    convert_to_onnx(args.weights_file, args.output_file)

if __name__ == "__main__":
    main()
