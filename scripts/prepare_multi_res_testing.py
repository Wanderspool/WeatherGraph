import os
import numpy as np
import json
import sys
from pathlib import Path

# Add project root to path to import weathergraph
sys.path.append(os.getcwd())
from weathergraph.tile_bundle import build_tile_bundle, grid_shape_from_resolution

def prepare_res(resolution, base_dir):
    res_dir = Path(base_dir) / f"res_{resolution}"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    grid_shape = grid_shape_from_resolution(resolution)
    node_count = grid_shape[0] * grid_shape[1]
    
    print(f"Preparing resolution {resolution}: {grid_shape} grid, {node_count} nodes")
    
    # 1. Create dummy weights
    weights_dir = res_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    np.save(weights_dir / "means.npy", np.zeros(78, dtype=np.float32))
    np.save(weights_dir / "stds.npy", np.ones(78, dtype=np.float32))
    
    # 2. Create dummy graph data (minimal)
    graph_dir = res_dir / "graph"
    graph_dir.mkdir(exist_ok=True)
    # Identity model doesn't need real edges, but tile builder does
    senders = np.array([0], dtype=np.int64)
    receivers = np.array([0], dtype=np.int64)
    np.save(graph_dir / "senders.npy", senders)
    np.save(graph_dir / "receivers.npy", receivers)
    
    # 3. Create dummy model(s)
    models_dir = res_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # For 0.25 and 0.1 deg, we MUST use tiling to fit in 1GB RAM
    if resolution <= 0.25:
        # Create a small tile model
        # Base it loosely on the grid
        tile_lat = grid_shape[0] // 4 + 1
        tile_lon = grid_shape[1] // 4
        if tile_lon == 0: tile_lon = grid_shape[1]
        tile_node_count = tile_lat * tile_lon
        tile_model_path = models_dir / "tile.onnx"
        create_identity_onnx(tile_node_count, tile_model_path)

        # Build tile bundle
        # We need more "fake" edges for the tile builder to work if we want halo
        # But for Identity, halo_hops=0 is fine
        # Re-generate senders/receivers for the full node count to satisfy tile builder
        # Just self-loops for all nodes
        full_senders = np.arange(node_count, dtype=np.int64)
        full_receivers = np.arange(node_count, dtype=np.int64)
        np.save(graph_dir / "full_senders.npy", full_senders)
        np.save(graph_dir / "full_receivers.npy", full_receivers)

        bundle_dir = res_dir / "bundle"
        build_tile_bundle(
            output_dir=bundle_dir,
            senders_path=graph_dir / "full_senders.npy",
            receivers_path=graph_dir / "full_receivers.npy",
            tile_model_dir=models_dir,
            tile_model_template="tile.onnx",
            reference_grid_resolution_degrees=resolution,
            tile_grid_shape=(tile_lat, tile_lon),
            halo_hops=0
        )
        print(f"Created tile bundle at {bundle_dir}")
    else:
        # Simple single model
        model_path = models_dir / "model.onnx"
        create_identity_onnx(node_count, model_path)
        print(f"Created single model at {model_path}")

def create_identity_onnx(node_count, output_path):
    import onnx
    import onnx.helper as helper
    from onnx import TensorProto
    
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, node_count, 78])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, node_count, 78])
    node_def = helper.make_node('Identity', ['input'], ['output'])
    graph_def = helper.make_graph([node_def], 'dummy', [input_tensor], [output_tensor])
    model_def = helper.make_model(graph_def, producer_name='dummy', opset_imports=[helper.make_opsetid("ai.onnx", 14)])
    model_def.ir_version = 8
    onnx.save(model_def, str(output_path))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--base-dir", type=str, default="test_artifacts")
    args = parser.parse_args()
    prepare_res(args.resolution, args.base_dir)
