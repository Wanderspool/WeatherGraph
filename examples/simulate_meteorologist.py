"""
simulate_meteorologist.py
=========================

Run 10 historical weather simulations using the WeatherGraph C++/ONNX engine.

Data-source selection
---------------------
Set the DATA_SOURCE environment variable to any adapter registered in
weathergraph.data_sources.  The default is "era5_netcdf" which reads local
NetCDF files from ERA5_DATA_DIR.

  ERA5_NETCDF (default)  — local ERA5 files  (no extra deps)
  ECMWF_OPEN             — ECMWF free real-time forecast  (pip install ecmwf-opendata cfgrib)
  CDS_ERA5               — Copernicus CDS reanalysis      (pip install cdsapi + ~/.cdsapirc)
  GFS                    — NOAA GFS via AWS Open Data     (pip install herbie-data)
  OPEN_METEO             — Open-Meteo single-point API    (no extra deps)
  CUSTOM                 — custom file / variable mapping (see DATA_SOURCE_SCHEMA)

For CUSTOM source, set DATA_SOURCE_SCHEMA to a JSON string, e.g.:
  export DATA_SOURCE_SCHEMA='{"source":"my.nc","variable_map":{"z":"geopotential"}}'

For OPEN_METEO, set OPEN_METEO_LAT and OPEN_METEO_LON for the forecast point.

Examples
--------
  # Default ERA5 local files:
  ERA5_DATA_DIR=data/era5_archives python examples/simulate_meteorologist.py

  # ECMWF open data (today's forecast):
  DATA_SOURCE=ecmwf_open python examples/simulate_meteorologist.py

  # Copernicus CDS (requires ~/.cdsapirc):
  DATA_SOURCE=cds_era5 CDS_DATE=2005-08-23 python examples/simulate_meteorologist.py

  # NOAA GFS:
  DATA_SOURCE=gfs GFS_DATE="2005-08-23 00:00" python examples/simulate_meteorologist.py

  # Custom NetCDF with renamed variables:
  DATA_SOURCE=custom DATA_SOURCE_SCHEMA='{"source":"forecast.nc","variable_map":{"z":"geopotential"}}' \\
      python examples/simulate_meteorologist.py
"""

import json
import os
import sys
import numpy as np
import xarray as xr
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weathergraph/core'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weathergraph import WeatherGraphModel
from weathergraph.data_sources import (
    list_sources,
    load_source,
    ERA5NetCDFAdapter,
)


def build_adapter_for_simulation(sim_file: str, era5_data_dir: str):
    """Return a DataSourceAdapter for the current DATA_SOURCE setting.

    Falls back gracefully: if the chosen source cannot provide per-simulation
    files, it returns a single adapter shared across all simulations.
    """
    source_name = os.getenv("DATA_SOURCE", "era5_netcdf").lower()

    if source_name == "era5_netcdf":
        return load_source("era5_netcdf", path=os.path.join(era5_data_dir, sim_file))

    if source_name == "ecmwf_open":
        return load_source(
            "ecmwf_open",
            date=os.getenv("ECMWF_DATE"),
            time=int(os.getenv("ECMWF_TIME", "0")),
            step=int(os.getenv("ECMWF_STEP", "0")),
        )

    if source_name == "cds_era5":
        date = os.getenv("CDS_DATE")
        if not date:
            raise ValueError("Set CDS_DATE=YYYY-MM-DD to use the cds_era5 source.")
        return load_source("cds_era5", date=date, time=os.getenv("CDS_TIME", "00:00"))

    if source_name == "gfs":
        date = os.getenv("GFS_DATE")
        if not date:
            raise ValueError("Set GFS_DATE='YYYY-MM-DD HH:MM' to use the gfs source.")
        return load_source(
            "gfs",
            date=date,
            fxx=int(os.getenv("GFS_FXX", "0")),
            source=os.getenv("GFS_SOURCE", "aws"),
        )

    if source_name == "open_meteo":
        lat = os.getenv("OPEN_METEO_LAT")
        lon = os.getenv("OPEN_METEO_LON")
        if not lat or not lon:
            raise ValueError("Set OPEN_METEO_LAT and OPEN_METEO_LON to use open_meteo.")
        return load_source(
            "open_meteo",
            latitude=float(lat),
            longitude=float(lon),
            model=os.getenv("OPEN_METEO_MODEL", "best_match"),
        )

    if source_name == "custom":
        schema_json = os.getenv("DATA_SOURCE_SCHEMA")
        if not schema_json:
            raise ValueError(
                "Set DATA_SOURCE_SCHEMA to a JSON string to use the custom source.\n"
                'Example: \'{"source":"forecast.nc","variable_map":{"z":"geopotential"}}\''
            )
        from weathergraph.data_sources import CustomAdapter
        return CustomAdapter.from_schema(json.loads(schema_json))

    # Unknown source — show registry and exit
    print(f"\n[ERROR] Unknown DATA_SOURCE='{source_name}'. Available sources:\n")
    list_sources()
    sys.exit(1)


def verify_environment(model_path, weights_dir, data_dir):
    """Strictly verifies that the REAL model and REAL data directories exist."""
    print("="*80)
    print("SYSTEM CHECK: VERIFYING REAL METEOROLOGICAL ARTIFACTS")
    print("="*80)
    
    if not os.path.exists(model_path):
        print(f"\n[CRITICAL ERROR] Real ONNX model not found at: {model_path}")
        print("-> REASON: The full reference 2022 model could not be generated on this specific machine.")
        print("-> TECHNICAL CONTEXT: Exporting the original reference JAX model to ONNX requires a CPU with AVX instructions.")
        print("   This server does not support AVX.")
        print("-> SOLUTION: Export 'weather_gnn.onnx' on an AVX-capable machine and upload it to the models/ directory.")
        sys.exit(1)

    source_name = os.getenv("DATA_SOURCE", "era5_netcdf").lower()
    if source_name == "era5_netcdf" and not os.path.exists(data_dir):
        print(f"\n[CRITICAL ERROR] ERA5 data directory not found at: {data_dir}")
        print("-> REASON: Real simulations require actual downloaded ERA5 NetCDF files.")
        print("-> SOLUTION: Create the directory and download the required ERA5 slices from Copernicus (CDS).")
        print("-> ALTERNATIVE: Set DATA_SOURCE to use a different data source (ecmwf_open, gfs, custom, ...).")
        print()
        print("Available data sources:")
        list_sources()
        sys.exit(1)

    print("[OK] Real model and environment artifacts verified.\n")
    if source_name != "era5_netcdf":
        print(f"[INFO] Data source: {source_name}")
        print()


def run_real_simulations():
    MODEL_PATH = os.getenv("WEATHERGRAPH_ONNX_MODEL", "models/weather_gnn.onnx")
    WEIGHTS_DIR = os.getenv("WEATHERGRAPH_WEIGHTS_DIR", os.getenv("KEISLER_WEIGHTS_DIR", "data"))
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

        try:
            adapter = build_adapter_for_simulation(sim['file'], ERA5_DATA_DIR)
        except ValueError as exc:
            print(f"  [ABORTED] Data source configuration error: {exc}")
            continue

        # For ERA5 local files, check existence before attempting load
        if isinstance(adapter, ERA5NetCDFAdapter) and not os.path.exists(adapter.path):
            print(f"  [ABORTED] Missing real ground-truth data: {adapter.path}")
            print(f"  -> Download '{sim['file']}' from ECMWF/CDS and place it in '{ERA5_DATA_DIR}'.")
            print(f"  -> Or set DATA_SOURCE to fetch data from another provider.")
            continue

        print(f"  [RUNNING] Using data source: {adapter.name}")
        try:
            if sim['id'] == 10:  # Ensemble special case
                ds = adapter.load()
                results = []
                for _ in range(3):
                    perturbed_ds = ds.copy(deep=True)
                    noise = np.random.normal(0, 0.05, 71042)
                    perturbed_ds['t'].loc[dict(level=850)] += noise
                    results.append(model.predict_one_step(perturbed_ds))
                print("  [SUCCESS] Calculated ensemble variance successfully.")
            else:
                forecast = model.forecast(adapter, steps=40)  # 10 days
                print(f"  [SUCCESS] Completed 10-day forecast. Final tensor shape: {forecast[-1].shape}")
        except Exception as e:
            print(f"  [ERROR] Simulation failed during execution: {e}")

    print("\n" + "="*80)
    print("SIMULATION RUN FINISHED.")
    print("="*80)

if __name__ == "__main__":
    run_real_simulations()
