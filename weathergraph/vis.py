import warnings
import numpy as np

try:
    import folium
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import imageio.v3 as iio
    from folium.raster_layers import ImageOverlay
    import io
    VIS_AVAILABLE = True
except ImportError:
    VIS_AVAILABLE = False


def _check_vis_available():
    if not VIS_AVAILABLE:
        raise ImportError(
            "Visualization dependencies are missing. "
            "Please install them using: pip install 'weathergraph[vis]'"
        )


def _get_colored_image(data_2d, cmap_name="viridis", vmin=None, vmax=None):
    """
    Converts a 2D numpy array to an RGBA image using matplotlib.
    """
    if vmin is None:
        vmin = np.nanmin(data_2d)
    if vmax is None:
        vmax = np.nanmax(data_2d)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    colored_data = cmap(norm(data_2d))
    return colored_data, vmin, vmax


def create_interactive_map(ds, variable, time_index=0, cmap_name="viridis"):
    """
    Creates an interactive Leaflet map for a specific variable and time step.

    Parameters
    ----------
    ds : xarray.Dataset
        The forecast dataset containing the variable.
    variable : str
        The variable name to visualize (e.g., 't', 'z', 'u', 'v').
    time_index : int
        The integer index along the time dimension.
    cmap_name : str
        Matplotlib colormap name.

    Returns
    -------
    folium.Map
        An interactive folium map object.
    """
    _check_vis_available()

    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in dataset.")

    if 'time' in ds.dims:
        data_slice = ds[variable].isel(time=time_index)
    else:
        data_slice = ds[variable]

    # Select the lowest pressure level if 'level' exists, or mean over levels.
    if 'level' in data_slice.dims:
        data_slice = data_slice.isel(level=-1) # Usually 1000hPa is at the end, or we can just pick 0

    lats = data_slice['lat'].values
    lons = data_slice['lon'].values
    data_2d = data_slice.values

    # Handle nan values
    data_2d = np.nan_to_num(data_2d, nan=np.nanmean(data_2d))

    colored_image, vmin, vmax = _get_colored_image(data_2d, cmap_name)

    # Folium expects bounds in [[lat_min, lon_min], [lat_max, lon_max]]
    # Note: ERA5 lats are typically descending (90 to -90). ImageOverlay expects the image
    # to align with the bounds. If lats are descending, the top of the image is lat_max.
    lat_min, lat_max = np.min(lats), np.max(lats)
    lon_min, lon_max = np.min(lons), np.max(lons)
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]

    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=2)

    ImageOverlay(
        image=colored_image,
        bounds=bounds,
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    # Add a color scale
    cmap = plt.get_cmap(cmap_name)
    colormap = folium.LinearColormap(
        colors=[cmap(i) for i in range(cmap.N)],
        vmin=vmin,
        vmax=vmax,
        caption=f"{variable} values"
    )
    m.add_child(colormap)

    return m


def create_animation(ds, variable, output_path, format="mp4", cmap_name="viridis", fps=5, resolution="medium"):
    """
    Creates an MP4 or GIF animation over the time dimension of the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The forecast dataset containing the variable.
    variable : str
        The variable name to visualize.
    output_path : str
        The path to save the generated animation.
    format : str
        'mp4' or 'gif'.
    cmap_name : str
        Matplotlib colormap name.
    fps : int
        Frames per second.
    resolution : str
        'low' (72 dpi), 'medium' (150 dpi), or 'high' (300 dpi).
    """
    _check_vis_available()

    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in dataset.")

    if 'time' not in ds.dims:
        raise ValueError("Dataset does not have a 'time' dimension for animation.")

    num_frames = len(ds['time'])
    if num_frames == 0:
        raise ValueError("Time dimension is empty.")

    # Map resolution to DPI
    dpi_map = {"low": 72, "medium": 150, "high": 300}
    dpi = dpi_map.get(resolution.lower(), 150)

    # Select the lowest pressure level if 'level' exists
    data_var = ds[variable]
    if 'level' in data_var.dims:
        data_var = data_var.isel(level=-1)

    lats = data_var['lat'].values
    lons = data_var['lon'].values

    # Determine global min/max for consistent colormapping
    vmin = float(data_var.min(skipna=True).values)
    vmax = float(data_var.max(skipna=True).values)

    frames = []
    
    # We use a single matplotlib figure and update it to be fast
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    ax.set_title(f"{variable} animation")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Initial plot
    initial_data = data_var.isel(time=0).values
    mesh = ax.pcolormesh(lons, lats, initial_data, cmap=cmap_name, vmin=vmin, vmax=vmax, shading='auto')
    fig.colorbar(mesh, ax=ax, label=variable)

    for i in range(num_frames):
        data_slice = data_var.isel(time=i).values
        mesh.set_array(data_slice.ravel())
        ax.set_title(f"{variable} - Step {i}")
        
        # Draw the canvas and get the RGB buffer
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        image = np.asarray(rgba_buffer)[:, :, :3]  # Drop alpha channel
        frames.append(image)

    plt.close(fig)

    if format.lower() == "mp4":
        iio.imwrite(output_path, frames, extension=".mp4", fps=fps)
    elif format.lower() == "gif":
        iio.imwrite(output_path, frames, extension=".gif", duration=1000/fps, loop=0)
    else:
        raise ValueError(f"Unsupported format '{format}'. Use 'mp4' or 'gif'.")
