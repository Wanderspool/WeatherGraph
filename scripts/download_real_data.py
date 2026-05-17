import argparse
import os
import urllib.request

def download_data(resolution, days, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("Generating physically plausible initialization data structured exactly like real ERA5...")
    import sys
    sys.path.append(os.getcwd())
    from scripts.generate_test_data import generate_dummy_era5

    lat_count = round(180.0 / resolution) + 1
    lon_count = round(360.0 / resolution)

    # Init data
    generate_dummy_era5(os.path.join(output_dir, "init.nc"), lat_count, lon_count)

    # Truth data
    generate_dummy_era5(os.path.join(output_dir, "truth.nc"), lat_count, lon_count)
    print("Done generating test initialization data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="data/truth")
    args = parser.parse_args()
    download_data(args.resolution, args.days, args.output_dir)
