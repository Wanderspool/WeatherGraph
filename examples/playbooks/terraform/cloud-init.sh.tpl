#!/usr/bin/env bash
# ==============================================================================
# WeatherGraph — Cloud-Init Bootstrap Script
# Rendered by Terraform from cloud-init.sh.tpl
# Runs once on first boot; self-destructs the instance when done.
# ==============================================================================
set -euo pipefail
exec > /var/log/weathergraph.log 2>&1

WG_DIR="/opt/weathergraph"
WG_VENV="$WG_DIR/.venv"
MODEL_PATH="$WG_DIR/models/weather_gnn.onnx"

echo "[wg] === WeatherGraph bootstrap started at $(date -u) ==="

# ── 1. System dependencies ────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y --no-install-recommends git cmake g++ curl python3 python3-pip python3-venv

# ── 2. Clone engine ───────────────────────────────────────────────────────────
git clone --depth=1 https://github.com/Wanderspool/WeatherGraph "$WG_DIR"
cd "$WG_DIR"

# ── 3. Python environment ─────────────────────────────────────────────────────
python3 -m venv "$WG_VENV"
"$WG_VENV/bin/pip" install --quiet --upgrade pip \
  "pybind11>=2.13.0" numpy xarray pandas dask netCDF4 zarr onnx

# ── 4. ONNX Runtime ───────────────────────────────────────────────────────────
make onnxruntime

# ── 5. Build C++ backend ──────────────────────────────────────────────────────
PYBIND11_CMAKE=$("$WG_VENV/bin/python" -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$PYBIND11_CMAKE"
cmake --build build --parallel "$(nproc)"
mkdir -p weathergraph/core
cp build/weathergraph_backend.so weathergraph/core/
cp onnxruntime-sdk/lib/libonnxruntime.so* weathergraph/core/

# ── 6. Acquire model ──────────────────────────────────────────────────────────
mkdir -p "$(dirname "$MODEL_PATH")"

%{ if model_source == "url" }
echo "[wg] Downloading model from ${model_url}"
curl -fSL "${model_url}" --output "$MODEL_PATH"
%{ endif }

%{ if model_source == "script" }
%{ if model_script_pip != "" }
"$WG_VENV/bin/pip" install --quiet ${model_script_pip}
%{ endif }
echo "[wg] Running conversion script: ${model_script}"
"$WG_VENV/bin/python" "$WG_DIR/${model_script}" ${model_script_args} --output "$MODEL_PATH"
%{ endif }

"$WG_VENV/bin/python" -c "
import onnx; m = onnx.load('$MODEL_PATH');
onnx.checker.check_model(m);
print('[validate] OK opset', m.opset_import[0].version)"

# ── 7. Run forecast ───────────────────────────────────────────────────────────
%{ if input_data != "" }
echo "[wg] Running forecast: steps=${steps}, fmt=${output_fmt}"

# Copy/download input data if it's a cloud URI
INPUT_LOCAL="/tmp/wg_input.nc"
if [[ "${input_data}" == gs://* ]]; then
  gsutil cp "${input_data}" "$INPUT_LOCAL"
elif [[ "${input_data}" == s3://* ]]; then
  aws s3 cp "${input_data}" "$INPUT_LOCAL"
else
  INPUT_LOCAL="${input_data}"
fi

export LD_LIBRARY_PATH="$WG_DIR/weathergraph/core:$LD_LIBRARY_PATH"
export PYTHONPATH="$WG_DIR"

"$WG_VENV/bin/python" - <<'PYEOF'
import json
from weathergraph import WeatherGraphModel
from weathergraph.data_sources import load_source

model = WeatherGraphModel(
    model_path="$MODEL_PATH",
    weights_dir="$WG_DIR/data",
)
source_name = "${data_source}"
params_raw = "${data_source_params}"
if source_name == "era5_netcdf":
  source = load_source("era5_netcdf", path="$INPUT_LOCAL")
elif params_raw:
  source = load_source(source_name, **json.loads(params_raw))
else:
  source = load_source(source_name)
model.forecast_export(
  source,
    steps=${steps},
    output_path="${output_dir}",
    fmt="${output_fmt}",
%{ if t0 != "" }    t0="${t0}",
%{ endif }
)
PYEOF

# Upload results if a bucket is given
%{ if output_bucket != "" }
if [[ "${output_bucket}" == gs://* ]]; then
  gsutil -m cp -r "${output_dir}/" "${output_bucket}/"
elif [[ "${output_bucket}" == s3://* ]]; then
  aws s3 sync "${output_dir}/" "${output_bucket}/"
fi
echo "[wg] Results uploaded to ${output_bucket}"
%{ endif }
%{ endif }

echo "[wg] === Done at $(date -u) ==="
