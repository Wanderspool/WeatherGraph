import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json

def generate_report(prediction_path, truth_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(prediction_path):
        print(f"Prediction file {prediction_path} missing. Skipping report.")
        return

    pred = xr.open_dataset(prediction_path)
    truth = xr.open_dataset(truth_path)
    
    # Ensure variables match
    var_name = list(pred.data_vars)[0]
    if var_name not in truth.data_vars:
         # Fallback for dummy truth
         truth_data = truth[list(truth.data_vars)[0]]
    else:
         truth_data = truth[var_name]

    # Simple comparison
    diff = pred[var_name].isel(time=-1) - truth_data.isel(level=0) # simplified for smoke test
    
    plt.figure(figsize=(12, 6))
    diff.plot(cmap='RdBu_r')
    plt.title(f"Forecast Error: {var_name}")
    plt.savefig(os.path.join(output_dir, f"error_map_{var_name}.png"))
    plt.close()
    
    summary = {"status": "success", "variable": var_name}
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output-dir", default="report")
    args = parser.parse_args()
    generate_report(args.prediction, args.truth, args.output_dir)
