import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def generate_report(prediction_path, truth_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
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
    import json
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Generate HTML index
    html_content = """<!DOCTYPE html>
<html>
<head>
<title>WeatherGraph Validation Report</title>
<style>
body { font-family: sans-serif; margin: 40px; }
.images { display: flex; flex-wrap: wrap; gap: 20px; }
.image-card { border: 1px solid #ccc; padding: 10px; border-radius: 5px; }
img { max-width: 600px; }
</style>
</head>
<body>
<h1>WeatherGraph Validation Report</h1>
<h2>Summary Statistics</h2>
<pre id="summary"></pre>
<h2>Visualizations</h2>
<div class="images">
"""
    for var in common_vars:
        html_content += f"""
        <div class="image-card">
            <h3>{var} Mean Error Map</h3>
            <img src="error_map_{var}.png" alt="Error Map for {var}" />
        </div>
        <div class="image-card">
            <h3>{var} RMSE over Time</h3>
            <img src="rmse_time_{var}.png" alt="RMSE Time Series for {var}" />
        </div>
        """
    html_content += """
</div>
<script>
fetch('summary.json')
  .then(response => response.json())
  .then(data => document.getElementById('summary').textContent = JSON.stringify(data, null, 2));
</script>
</body>
</html>
"""
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output-dir", default="report")
    args = parser.parse_args()
    generate_report(args.prediction, args.truth, args.output_dir)
