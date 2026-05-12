import numpy as np
import onnx
import onnxscript
from onnxscript import opset18 as op
import os
import argparse

# --- Concrete GNN Ops for the Engine ---

def get_weight_const(weights_dir, layer_name, param_name):
    path = os.path.join(weights_dir, layer_name, f"{param_name}.npy")
    return np.load(path)

def build_full_graph(weights_dir, graph_dir, out_path):
    print(f"Building full WeatherGraph-compatible ONNX graph at {out_path}...")
    
    # 1. Load static graph topology for constants
    enc_senders = np.load(os.path.join(graph_dir, "senders_receivers_encoder/senders.npy")).astype(np.int64)
    enc_receivers = np.load(os.path.join(graph_dir, "senders_receivers_encoder/receivers.npy")).astype(np.int64)
    
    # Weights for Encoder Edge MLP (linear, linear_1, linear_2)
    w0 = get_weight_const(weights_dir, "linear", "w")
    b0 = get_weight_const(weights_dir, "linear", "b")
    w1 = get_weight_const(weights_dir, "linear_1", "w")
    b1 = get_weight_const(weights_dir, "linear_1", "b")
    w2 = get_weight_const(weights_dir, "linear_2", "w")
    b2 = get_weight_const(weights_dir, "linear_2", "b")
    
    # Weights for Encoder Node MLP (linear_3, linear_4, linear_5)
    w3 = get_weight_const(weights_dir, "linear_3", "w")
    b3 = get_weight_const(weights_dir, "linear_3", "b")
    
    # Weights for LayerNorms
    ln0_s = get_weight_const(weights_dir, "layer_norm", "scale")
    ln0_o = get_weight_const(weights_dir, "layer_norm", "offset")

    # Load normalization constants
    means_val = np.load("data/means.npy").astype(np.float32)[:78]
    stds_val = np.load("data/stds.npy").astype(np.float32)[:78]

    @onnxscript.script(default_opset=op)
    def weathergraph_engine(
        input_data: onnxscript.FLOAT[1, 71042, 78]
    ):
        # --- Pre-processing ---
        # Normalize (In-graph constants)
        means = op.Constant(value=means_val)
        stds = op.Constant(value=stds_val)
        x = (input_data - means) / stds
        
        # --- Encoder ---
        # 1. Gather sender/receiver features
        # Note: input_data is [1, 71042, 78]. We need [71042, 78]
        # ONNX Gather requires axis.
        nodes = op.Squeeze(x, axes=[0])
        
        # Gather sender nodes
        senders_feat = op.Gather(nodes, op.Constant(value=enc_senders), axis=0)
        # Gather receiver nodes (static coords or latent features)
        # For simplicity in this prototype, we'll focus on the ERA5 node updates
        
        # 2. Edge MLP
        edge_h = op.MatMul(senders_feat, op.Constant(value=w0)) + op.Constant(value=b0)
        edge_h = op.Relu(edge_h)
        edge_h = op.MatMul(edge_h, op.Constant(value=w1)) + op.Constant(value=b1)
        edge_h = op.Relu(edge_h)
        edge_h = op.MatMul(edge_h, op.Constant(value=w2)) + op.Constant(value=b2)
        
        # 3. Scatter Add (Edge -> Node)
        # Aggregate back to H3 grid (5882 nodes)
        # We need a zero base for scatter
        h3_base = op.Constant(value=np.zeros((5882, 256), dtype=np.float32))
        
        # ScatterND indices must be [N, 1] for 1D node indexing
        scatter_idx = op.Reshape(op.Constant(value=enc_receivers), op.Constant(value=np.array([-1, 1], dtype=np.int64)))
        
        latent_nodes = op.ScatterND(h3_base, scatter_idx, edge_h, reduction="add")
        
        # 4. Node MLP
        latent_nodes = op.MatMul(latent_nodes, op.Constant(value=w3)) + op.Constant(value=b3)
        latent_nodes = op.LayerNormalization(latent_nodes, op.Constant(value=ln0_s), op.Constant(value=ln0_o))

        # --- Processor & Decoder ---
        # (Omitted for prototype brevity, but following the same Gather/MLP/Scatter pattern)
        
        # --- Post-processing ---
        # Map back to ERA5 and denormalize
        # For this prototype, we return the latent nodes as a proof of concept
        return latent_nodes

    model_proto = weathergraph_engine.to_model_proto()
    onnx.save(model_proto, out_path)
    print("Export successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build WeatherGraph ONNX from extracted data artifacts.")
    parser.add_argument("--weights-dir", default="data/weights", help="Path to extracted weights directory")
    parser.add_argument("--graph-dir",   default="data/graph_data", help="Path to extracted graph data directory")
    parser.add_argument("--output",      default="models/weather_gnn.onnx", help="Output .onnx file path")
    args = parser.parse_args()
    build_full_graph(args.weights_dir, args.graph_dir, args.output)
