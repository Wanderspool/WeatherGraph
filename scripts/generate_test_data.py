import xarray as xr
import numpy as np
import os

def generate_dummy_era5(output_path, lat_count, lon_count):
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    vars = ['z', 'q', 't', 'u', 'v', 'w']
    
    lat = np.linspace(90, -90, lat_count)
    lon = np.linspace(0, 360, lon_count, endpoint=False)
    
    coords = {"level": levels, "lat": lat, "lon": lon}
    # Add a time dimension for truth files
    if "_t10" in output_path:
        coords["time"] = [0]
        data_vars = {var: (["time", "level", "lat", "lon"], np.random.rand(1, len(levels), lat_count, lon_count).astype(np.float32)) for var in vars}
    else:
        data_vars = {var: (["level", "lat", "lon"], np.random.rand(len(levels), lat_count, lon_count).astype(np.float32)) for var in vars}
        
    ds = xr.Dataset(data_vars, coords=coords)
    ds.to_netcdf(output_path)
    print(f"Generated dummy ERA5 at {output_path}")

if __name__ == "__main__":
    import sys
    generate_dummy_era5(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
