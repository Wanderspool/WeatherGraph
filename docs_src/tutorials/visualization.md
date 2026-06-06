# Visualizing Predictions

WeatherGraph provides a built-in visualization module (`weathergraph.vis`) that helps researchers and operational meteorologists quickly inspect forecast trajectories. It supports:
1. **Interactive geographic maps** using Leaflet (via `folium`).
2. **Smooth time-series animations** (MP4 or GIF) using `matplotlib` and `imageio`.

---

## Prerequisites

Visualization tools require the `[vis]` extra dependencies. Install them using:

```bash
pip install 'weathergraph[vis]'
```

---

## 1. Creating Interactive Geographic Maps

The `create_interactive_map` function takes a forecast dataset and returns a Leaflet map. By default, it overlays the 2D field of a variable at the lowest pressure level (typically 1000 hPa) with a dynamic color scale.

### Python Example
Here is how to load a forecast and save an interactive HTML map:

```python
import weathergraph as wg
import xarray as xr
from weathergraph.vis import create_interactive_map

# 1. Load the forecast dataset
ds = xr.open_dataset("output/first_forecast.nc")

# 2. Generate the interactive map for temperature ('t') at step index 4 (24 hours)
leaflet_map = create_interactive_map(
    ds=ds,
    variable="t",
    time_index=4,
    cmap_name="coolwarm"  # Supports any standard matplotlib colormap
)

# 3. Save the map as a standalone HTML file
leaflet_map.save("output/temperature_map_24h.html")
print("Leaflet map saved to output/temperature_map_24h.html")
```

When opened in any web browser, the map allows you to zoom, pan, and read data ranges via the dynamic color scale.

---

## 2. Generating MP4 and GIF Animations

If your forecast covers multiple days, you can animate the rollouts over the `time` dimension to see the movement of weather fronts, winds, or pressure fields.

### Python Example
Here is how to generate an MP4 video of temperature changes:

```python
import weathergraph as wg
import xarray as xr
from weathergraph.vis import create_animation

# 1. Load the multi-step dataset
ds = xr.open_dataset("output/first_forecast.nc")

# 2. Render the animation for temperature
create_animation(
    ds=ds,
    variable="t",
    output_path="output/temperature_evolution.mp4",
    format="mp4",          # Can also be 'gif'
    cmap_name="viridis",
    fps=5,                 # Frames per second
    resolution="medium"    # Options: 'low' (72 dpi), 'medium' (150 dpi), 'high' (300 dpi)
)
print("MP4 animation saved to output/temperature_evolution.mp4")
```

---

## 3. Visualizing via the CLI

You can perform the same visualization operations from the command line using the `visualize` subcommand.

### Exporting a Map (HTML)
To generate an interactive Leaflet map of geopotential height (`z`) at step index 0:

```bash
weathergraph visualize \
  --input output/forecast_run_01/z_1000hPa.nc \
  --variable z \
  --format html \
  --time-index 0 \
  --output output/geopotential_map.html \
  --cmap jet
```

### Exporting an Animation (MP4)
To generate a high-resolution MP4 animation of specific humidity (`q`) across the entire forecast trajectory:

```bash
weathergraph visualize \
  --input output/forecast_run_01/q_850hPa.nc \
  --variable q \
  --format mp4 \
  --output output/humidity_forecast.mp4 \
  --cmap YlGnBu \
  --resolution high
```