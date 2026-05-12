import argparse
import os

import numpy as np

def extract_npz(path, out_dir):
    print(f"Extracting {path}...")
    try:
        data = np.load(path, allow_pickle=True)
        for key in data.files:
            val = data[key]
            print(f"  {key}: {val.shape}")
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"{key}.npy"), val)
    except Exception as e:
        print(f"Failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract GNN graph topology and features from the reference model .npz.gz archives."
    )
    parser.add_argument(
        "--source",
        default="../reference-model",
        help="Path to the reference-model source directory (default: ../reference-model)",
    )
    parser.add_argument(
        "--output",
        default="data/graph_data",
        help="Output directory for extracted graph .npy files (default: data/graph_data)",
    )
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source)
    if not os.path.isdir(source_dir):
        print(f"[ERROR] Source directory not found: {source_dir}")
        print("  Place the original reference-model repo next to this project, or pass --source <path>")
        raise SystemExit(1)

    # Upstream reference-model layout still stores artifacts under src/keisler_2022/data.
    data_dir = os.path.join(source_dir, "src", "keisler_2022", "data")
    out_dir = os.path.abspath(args.output)
    
    files = [
        "senders_receivers_encoder.npz.gz",
        "senders_receivers_processor.npz.gz",
        "senders_receivers_decoder.npz.gz",
        "node_features_n71042_e112246_s-8416688801745003395_r-6736346125390000850.npz.gz",
        "edge_features_n71042_e112246_s-8416688801745003395_r-6736346125390000850.npz.gz",
        "node_features_n5882_e41162_s-1135048384487896564_r7866883539119236492.npz.gz",
        "edge_features_n5882_e41162_s-1135048384487896564_r7866883539119236492.npz.gz",
        "node_features_n71042_e112246_s-6736346125390000850_r-8416688801745003395.npz.gz",
        "edge_features_n71042_e112246_s-6736346125390000850_r-8416688801745003395.npz.gz",
        "orography_landsea.npz.gz"
    ]
    
    for f in files:
        extract_npz(os.path.join(data_dir, f), os.path.join(out_dir, f.replace(".npz.gz", "")))
    print(f"[+] Graphs extracted to: {out_dir}")

if __name__ == "__main__":
    main()
