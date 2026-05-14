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
    
    # Align times if necessary
    common_vars = set(pred.data_vars) & set(truth.data_vars)
    
    for var in common_vars:
        diff = pred[var] - truth[var]
        
        # Plot mean error map
        plt.figure(figsize=(12, 6))
        diff.mean(dim='time').plot(cmap='RdBu_r')
        plt.title(f"Mean Error Map: {var}")
        plt.savefig(os.path.join(output_dir, f"error_map_{var}.png"))
        plt.close()
        
        # Plot RMSE profile over time
        rmse = np.sqrt((diff**2).mean(dim=['lat', 'lon']))
        plt.figure(figsize=(10, 5))
        rmse.plot()
        plt.title(f"Global RMSE over Time: {var}")
        plt.ylabel("RMSE")
        plt.savefig(os.path.join(output_dir, f"rmse_time_{var}.png"))
        plt.close()

    # Save summary stats to JSON
    summary = {
        "variables": list(common_vars),
        "rmse": {var: float(np.sqrt((pred[var] - truth[var])**2).mean()) for var in common_vars},
        "inference_time_seconds": float(os.environ.get("WEATHERGRAPH_INFERENCE_TIME", 0))
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output-dir", default="report")
    args = parser.parse_args()
    generate_report(args.prediction, args.truth, args.output_dir)
