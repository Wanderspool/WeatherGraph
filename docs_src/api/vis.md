# weathergraph.vis

The `weathergraph.vis` module provides plotting utilities to generate interactive maps and animations from weather forecast outputs.

---

## Technical Overview & Visualizations

WeatherGraph supports two primary visualization styles for gridded forecast variables:

### 1. Leaflet Interactive Maps (HTML)
Generated using `create_interactive_map` which utilizes `folium` to compile a lightweight Leaflet map. This map:
*   Extracts a specific forecast variable at a defined time index and pressure level.
*   Projects the data array coordinates onto a global geographic tile set.
*   Embeds dynamic color overlays, coordinate markers, and mouse-hover value inspectors.
*   Saves the resulting interface to a self-contained `.html` file.

### 2. Video Animations (MP4 / GIF)
Generated using `create_animation` which uses `matplotlib` to render successive forecast frames and compiles them into a video using `imageio` (configured with the `ffmpeg` plugin). This animation:
*   Sequentially loops through the time steps of a rollout.
*   Applies a uniform colormap and scales limits to prevent visual flashing between frames.
*   Outputs standard `.mp4` or `.gif` video streams with adjustable framerates (FPS) and resolution bounds.

---

## Technical Example

The following script loads a NetCDF prediction output and generates both an interactive Leaflet HTML map and a high-resolution MP4 video animation:

```python
import xarray as xr
import weathergraph.vis as vis

# 1. Load forecast dataset
ds = xr.open_dataset("output/forecast.nc")

# 2. Render a Folium interactive map for temperature (t) at the 500hPa level
html_map = vis.create_interactive_map(
    ds=ds,
    variable="t",
    time_index=0,
    cmap_name="thermal"
)
html_map.save("plots/temperature_init.html")

# 3. Compile a 10 FPS MP4 animation showing wind component evolution
vis.create_animation(
    ds=ds,
    variable="u",
    output_path="plots/zonal_wind_rollout.mp4",
    format="mp4",
    cmap_name="coolwarm",
    fps=10,
    resolution="high"
)
```

---

## Functions

::: weathergraph.vis.create_interactive_map
    options:
      show_source: true

---

::: weathergraph.vis.create_animation
    options:
      show_source: true
