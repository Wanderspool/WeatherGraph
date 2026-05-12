# =============================================================================
# WeatherGraph — Multi-stage Docker Build
#
# Stages:
#   builder    — Downloads ONNX Runtime, compiles the C++ pybind11 backend.
#   test       — Runs the full unit test suite against the built backend.
#   production — Minimal runtime image suitable for deployment.
#
# Build examples:
#   # Run tests only:
#   docker build --target test -t weathergraph:test .
#
#   # Build production image:
#   docker build --target production -t weathergraph:latest .
# =============================================================================

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ARG ONNXRUNTIME_VERSION=1.18.0
ARG ONNXRUNTIME_PLATFORM=linux-x64
ARG ONNXRUNTIME_URL=

# Build tools + curl for ONNX Runtime download
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    curl \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy build manifests first to allow layer caching of dependency downloads
COPY Makefile CMakeLists.txt pyproject.toml ./
COPY onnxruntime-sdk/include/ onnxruntime-sdk/include/
COPY src/ src/
COPY weathergraph/ weathergraph/

# Download the ONNX Runtime shared libraries (matches `make onnxruntime`)
RUN if [ -n "$ONNXRUNTIME_URL" ]; then \
            make onnxruntime \
                ONNXRUNTIME_VERSION="$ONNXRUNTIME_VERSION" \
                ONNXRUNTIME_PLATFORM="$ONNXRUNTIME_PLATFORM" \
                ONNXRUNTIME_URL="$ONNXRUNTIME_URL"; \
        else \
            make onnxruntime \
                ONNXRUNTIME_VERSION="$ONNXRUNTIME_VERSION" \
                ONNXRUNTIME_PLATFORM="$ONNXRUNTIME_PLATFORM"; \
        fi

# CMakeLists.txt expects a pybind11-enabled virtual environment during build.
RUN python3.11 -m venv /root/weathergraph/venv \
    && /root/weathergraph/venv/bin/pip install --quiet --upgrade pip pybind11 \
    && python3.11 -m venv /app/weathergraph/venv \
    && /app/weathergraph/venv/bin/pip install --quiet --upgrade pip pybind11

# Compile the C++ pybind11 backend
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel "$(nproc)" \
    && cp build/weathergraph_backend.so weathergraph/core/ \
    && cp onnxruntime-sdk/lib/libonnxruntime*.so* weathergraph/core/

# Install Python runtime dependencies
RUN pip install --no-cache-dir \
    numpy \
    xarray \
    pandas \
    dask \
    netCDF4 \
    zarr

# ─── Stage 2: Test ────────────────────────────────────────────────────────────
FROM builder AS test

COPY tests/ tests/

RUN pip install --no-cache-dir \
    pytest \
    psutil \
    memory-profiler \
    onnx

# Validate the build by running the unit tests at image build time.
# This ensures the test image is always green before it can be pushed.
ENV PYTHONPATH=/app
RUN pytest tests/test_cpp_backend.py tests/test_memory_leak.py -v --tb=short

# ─── Stage 3: Production ──────────────────────────────────────────────────────
FROM python:3.11-slim AS production

# libgomp is required by ONNX Runtime for multi-threaded inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the compiled backend and Python package from the builder
COPY --from=builder /app/weathergraph/ ./weathergraph/

# Reinstall Python runtime dependencies (no build tools in this stage)
RUN pip install --no-cache-dir \
    numpy \
    xarray \
    pandas \
    dask \
    netCDF4 \
    zarr

# The ONNX model is large — mount it at runtime from a GCS bucket or volume.
# Example: docker run -v /path/to/models:/app/models weathergraph:latest
# Accelerator-backed deployments also need an ONNX Runtime bundle with the
# matching execution-provider libraries plus the corresponding vendor runtime.
VOLUME ["/app/models"]

ENV PYTHONPATH=/app

# Default to the supported researcher-facing CLI. This still imports the
# backend on container start, so it remains a lightweight smoke check.
CMD ["python3", "-m", "weathergraph.cli", "--help"]
