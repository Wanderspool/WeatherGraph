# ==============================================================================
# WeatherGraph — Build Pipeline
#
# Prerequisites:
#   - Python 3.10+ with pip
#   - CMake >= 3.15
#   - curl
#   - Original reference-model directory at $(WEATHERGRAPH_SOURCE_DIR)
#
# Typical first-time setup:
#   make all WEATHERGRAPH_SOURCE_DIR=/path/to/reference-model
#
# If the source model is already in the default location (../reference-model):
#   make all
# ==============================================================================

PYTHON        ?= python3
WEATHERGRAPH_SOURCE_DIR ?= $(if $(KEISLER_SOURCE_DIR),$(KEISLER_SOURCE_DIR),../reference-model)
KEISLER_SOURCE_DIR ?= $(WEATHERGRAPH_SOURCE_DIR)

# ONNX Runtime release to download
ONNXRUNTIME_VERSION ?= 1.18.0
ONNXRUNTIME_PLATFORM ?= linux-x64
ONNXRUNTIME_TARBALL  = onnxruntime-$(ONNXRUNTIME_PLATFORM)-$(ONNXRUNTIME_VERSION).tgz
ONNXRUNTIME_URL      = https://github.com/microsoft/onnxruntime/releases/download/v$(ONNXRUNTIME_VERSION)/$(ONNXRUNTIME_TARBALL)

BUILD_DIR     = build
MODEL_OUT     = models/weather_gnn.onnx
SO_OUT        = weathergraph/core/weathergraph_backend.so

.PHONY: all onnxruntime extract convert build test clean help

# ------------------------------------------------------------------------------
all: extract convert build  ## Full pipeline (extract → convert → build)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
onnxruntime:  ## Download & unpack ONNX Runtime shared library into onnxruntime-sdk/lib/
# ------------------------------------------------------------------------------
	@echo "[onnxruntime] Downloading ONNX Runtime v$(ONNXRUNTIME_VERSION) for $(ONNXRUNTIME_PLATFORM)..."
	@mkdir -p onnxruntime-sdk/lib
	@curl -fsSL "$(ONNXRUNTIME_URL)" | tar -xz --strip-components=1 \
	    -C onnxruntime-sdk \
	    --wildcards "*/lib/libonnxruntime*"
	@echo "[onnxruntime] Done. Libraries in onnxruntime-sdk/lib/"

# ------------------------------------------------------------------------------
extract:  ## Extract weights & graph topology from WEATHERGRAPH_SOURCE_DIR into data/
# ------------------------------------------------------------------------------
	@if [ ! -d "$(WEATHERGRAPH_SOURCE_DIR)" ]; then \
	    echo ""; \
	    echo "[ERROR] Source model directory not found: $(WEATHERGRAPH_SOURCE_DIR)"; \
	    echo ""; \
	    echo "  The original reference-model JAX source must be present to extract weights."; \
	    echo "  Clone it next to this project, or pass the path explicitly:"; \
	    echo ""; \
	    echo "    make extract WEATHERGRAPH_SOURCE_DIR=/path/to/reference-model"; \
	    echo "    # Backward-compatible alias: KEISLER_SOURCE_DIR=/path/to/reference-model"; \
	    echo ""; \
	    exit 1; \
	fi
	@echo "[extract] Extracting weights from $(WEATHERGRAPH_SOURCE_DIR)..."
	$(PYTHON) exporter/extract_weights.py --source "$(WEATHERGRAPH_SOURCE_DIR)" --output data/weights
	@echo "[extract] Extracting graph topology from $(WEATHERGRAPH_SOURCE_DIR)..."
	$(PYTHON) exporter/extract_graphs.py --source "$(WEATHERGRAPH_SOURCE_DIR)" --output data/graph_data
	@echo "[extract] Done."

# ------------------------------------------------------------------------------
convert: data/weights data/graph_data  ## Build ONNX graph → models/weather_gnn.onnx
# ------------------------------------------------------------------------------
	@echo "[convert] Building ONNX graph..."
	$(PYTHON) exporter/build_gnn_graph.py
	@echo "[convert] Model written to $(MODEL_OUT)"

# ------------------------------------------------------------------------------
build: onnxruntime-sdk/lib/libonnxruntime.so $(MODEL_OUT)  ## Compile C++ pybind11 backend
# ------------------------------------------------------------------------------
	@echo "[build] Configuring CMake..."
	@mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	@echo "[build] Compiling..."
	cmake --build $(BUILD_DIR) --parallel
	@echo "[build] Copying .so into weathergraph/core/"
	@cp $(BUILD_DIR)/weathergraph_backend.so $(SO_OUT)
	@cp onnxruntime-sdk/lib/libonnxruntime.so* weathergraph/core/
	@echo "[build] Done. Backend: $(SO_OUT)"

# ------------------------------------------------------------------------------
test:  ## Run the test suite
# ------------------------------------------------------------------------------
	$(PYTHON) -m pytest tests/ -v

# ------------------------------------------------------------------------------
clean:  ## Remove all generated files (data artifacts, build dir, .so, .onnx)
# ------------------------------------------------------------------------------
	@echo "[clean] Removing generated artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf data/weights data/graph_data data/means.npy data/stds.npy
	rm -f  $(MODEL_OUT)
	rm -f  $(SO_OUT) weathergraph/core/libonnxruntime*
	@echo "[clean] Done."

# ------------------------------------------------------------------------------
help:  ## Show this help message
# ------------------------------------------------------------------------------
	@echo ""
	@echo "WeatherGraph — Makefile targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables:"
	@echo "  WEATHERGRAPH_SOURCE_DIR Path to the reference JAX model (default: ../reference-model)"
	@echo "  KEISLER_SOURCE_DIR      Backward-compatible alias for WEATHERGRAPH_SOURCE_DIR"
	@echo "  PYTHON              Python interpreter to use          (default: python3)"
	@echo "  ONNXRUNTIME_VERSION ONNX Runtime version to download   (default: $(ONNXRUNTIME_VERSION))"
	@echo ""
