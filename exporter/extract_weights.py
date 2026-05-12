import argparse
import os
import pickle

import numpy as np

class ObjectDescriptor:
    def __init__(self, module, name, args=None, kwargs=None):
        self.module = module
        self.name = name
        self.args = args or []
        self.kwargs = kwargs or {}
    def __repr__(self):
        return f"Obj({self.module}.{self.name})"

class DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if any(x in module for x in ['jax', 'haiku', 'jraph']):
            return lambda *args, **kwargs: ObjectDescriptor(module, name, args, kwargs)
        return super().find_class(module, name)

def main():
    parser = argparse.ArgumentParser(
        description="Extract model weights from a Keisler-2022 pickle checkpoint."
    )
    parser.add_argument(
        "--source",
        default="../keisler-2022",
        help="Path to the keisler-2022 source directory (default: ../keisler-2022)",
    )
    parser.add_argument(
        "--output",
        default="data/weights",
        help="Output directory for extracted .npy weight files (default: data/weights)",
    )
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source)
    if not os.path.isdir(source_dir):
        print(f"[ERROR] Source directory not found: {source_dir}")
        print("  Place the original keisler-2022 repo next to this project, or pass --source <path>")
        raise SystemExit(1)

    data_dir = os.path.join(source_dir, "src", "keisler_2022", "data")
    pkl_candidates = [
        f for f in os.listdir(data_dir)
        if f.endswith(".pkl") and "good_era5_forecast" in f
    ] if os.path.isdir(data_dir) else []

    if not pkl_candidates:
        print(f"[ERROR] No checkpoint .pkl found in {data_dir}")
        raise SystemExit(1)

    pkl_path = os.path.join(data_dir, sorted(pkl_candidates)[-1])
    print(f"[+] Loading checkpoint: {pkl_path}")
    out_dir = os.path.abspath(args.output)

    with open(pkl_path, "rb") as f:
        data = DataUnpickler(f).load()

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, os.path.join(path, str(k)))
        elif isinstance(obj, ObjectDescriptor):
            # If it's a Haiku FlatMapping, the data is usually in args[0]
            if obj.name == 'FlatMapping':
                walk(obj.args[0], path)
            else:
                # Other objects
                for arg in obj.args:
                    walk(arg, path)
                for k, v in obj.kwargs.items():
                    walk(v, os.path.join(path, str(k)))
        elif hasattr(obj, 'shape'):
            print(f"Saving {path} {obj.shape}")
            out_file = os.path.join(out_dir, f"{path}.npy")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            np.save(out_file, obj)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, os.path.join(path, str(i)))

    walk(data)
    print(f"[+] Weights extracted to: {out_dir}")

if __name__ == "__main__":
    main()
