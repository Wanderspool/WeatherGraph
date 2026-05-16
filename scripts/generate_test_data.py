import xarray as xr
import numpy as np
import pandas as pd
import os

def generate_dummy_era5(output_path, lat_count, lon_count):
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    vars = ['z', 'q', 't', 'u', 'v', 'w']
    
    lat = np.linspace(90, -90, lat_count)
    lon = np.linspace(0, 360, lon_count, endpoint=False)
    
    coords = {
        "level": levels,
        "lat": lat,
        "lon": lon
    }
    
    data_vars = {}
    for var in vars:
        # Generate some semi-realistic dummy data
        data = np.random.rand(len(levels), lat_count, lon_count).astype(np.float32)
        data_vars[var] = (["level", "lat", "lon"], data)
        
    ds = xr.Dataset(data_vars, coords=coords)
    ds.to_netcdf(output_path)
    print(f"Generated dummy data at {output_path} with shape ({lat_count}, {lon_count})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python generate_test_data.py <output_path> <lat_count> <lon_count>")
    else:
        generate_dummy_era5(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
