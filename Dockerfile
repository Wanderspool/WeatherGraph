# =============================================================================
# Keisler Weather Engine — Multi-stage Docker Build
#
# Stages:
#   builder    — Downloads ONNX Runtime, compiles the C++ pybind11 backend.
#   test       — Runs the full unit test suite against the built backend.
#   production — Minimal runtime image suitable for deployment.
#
# Build examples:
#   # Run tests only:
#   docker build --target test -t keisler-engine:test .
#
#   # Build production image:
#   docker build --target production -t keisler-engine:latest .
# =============================================================================

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

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
COPY keisler_engine/ keisler_engine/

# Download the ONNX Runtime shared library (matches `make onnxruntime`)
RUN make onnxruntime

# CMakeLists.txt hardcodes pybind11 at /root/keisler_engine/venv and uses
# keisler_engine/venv (relative) for Python include detection.
RUN python3.11 -m venv /root/keisler_engine/venv \
    && /root/keisler_engine/venv/bin/pip install --quiet --upgrade pip pybind11 \
    && python3.11 -m venv /app/keisler_engine/venv \
    && /app/keisler_engine/venv/bin/pip install --quiet --upgrade pip pybind11

# Compile the C++ pybind11 backend
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel "$(nproc)" \
    && cp build/keisler_cpp_backend.so keisler_engine/core/ \
    && cp onnxruntime-sdk/lib/libonnxruntime.so* keisler_engine/core/

# Install Python runtime dependencies
RUN pip install --no-cache-dir \
    numpy \
    xarray \
    pandas \
    dask \
    netCDF4

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
COPY --from=builder /app/keisler_engine/ ./keisler_engine/

# Reinstall Python runtime dependencies (no build tools in this stage)
RUN pip install --no-cache-dir \
    numpy \
    xarray \
    pandas \
    dask \
    netCDF4

# The ONNX model is large — mount it at runtime from a GCS bucket or volume.
# Example: docker run -v /path/to/models:/app/models keisler-engine:latest
VOLUME ["/app/models"]

ENV PYTHONPATH=/app

# Smoke-test: verify the engine is importable on container start
CMD ["python3", "-c", \
     "from keisler_engine import GraphWeatherModel; print('Keisler engine ready.')"]
