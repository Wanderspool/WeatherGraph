# The WeatherGraph Engine Architecture

WeatherGraph is built using a **hybrid split architecture** designed to maximize computational efficiency while preserving developer flexibility. This page details how the C++ data plane and the Python control plane coordinate to run weather forecasts.

---

## 1. High-Level Design: Data Plane vs. Control Plane

The codebase is split into two distinct layers:

```mermaid
graph LR
    subgraph Python Control Plane
        A[Data Ingestion / Adapters] -->|xarray.Dataset| B[Tensor Preparation]
        B -->|NumPy float32| C[Autoregressive Orchestration]
        E[Output Formatting / CF-Export] <--|NumPy float32| C
    end
    subgraph C++ Data Plane
        C -->|pybind11 Buffer Protocol| D[ONNX Runtime Engine]
        D -->|Zero-Copy Inference| D
        D -->|pybind11 Buffer Protocol| C
    end
```

### The C++ Data Plane
Exposed to Python via pybind11 as `weathergraph_backend.WeatherGraphEngine`.
*   **Responsibilities**: Manages the ONNX Runtime sessions, allocates execution-provider arenas, handles GPU execution provider interfaces (CUDA/TensorRT), and executes raw model inference (`session->Run(...)`).
*   **Rationale**: Performance-critical inference loop paths must run with absolute minimum memory allocation and CPU scheduling latency.

### The Python Control Plane
The user-facing package `weathergraph`.
*   **Responsibilities**: Handles GRIB/NetCDF/Zarr ingestion via xarray/Dask, converts spatial grids to flat GNN node layouts, manages autoregressive loop iteration steps, maps spatial coordinates, and exports CF-compliant NetCDF/Zarr files.
*   **Rationale**: Scientists and developers expect a high-level, expressive interface that integrates with standard climate tools like MetPy, Dask, and Jupiter notebooks.

---

## 2. The 78-Channel Scientific Contract

WeatherGraph GNN models expect inputs in a flat, 2D node format with a shape of `[1, nodes, 78]`. The 78-channel dimension represents a fixed sequence of **6 atmospheric variables** across **13 pressure levels**:

### Core Variables
1.  `z`: Geopotential ($m^2/s^2$)
2.  `q`: Specific Humidity ($kg/kg$)
3.  `t`: Air Temperature ($K$)
4.  `u`: Eastward Wind ($m/s$)
5.  `v`: Northward Wind ($m/s$)
6.  `w`: Vertical Velocity ($Pa/s$)

### Vertical Levels (hPa)
50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa.

### Flat Tensor Layout
The 78 channels are organized as $13 \text{ levels} \times 6 \text{ variables}$ in the following order:

$$\text{Channels} = [\underbrace{z_{50}, q_{50}, t_{50}, u_{50}, v_{50}, w_{50}}_{\text{Level 50 hPa}}, \quad \dots \quad , \quad \underbrace{z_{1000}, q_{1000}, t_{1000}, u_{1000}, v_{1000}, w_{1000}}_{\text{Level 1000 hPa}}]$$

---

## 3. Zero-Copy Buffer Sharing

Memory allocations during inference can quickly trigger CPU/GPU bottle-necks. To eliminate this, WeatherGraph uses the **pybind11 Buffer Protocol** to share memory between Python and C++ without copying:

1.  **Preparation**: Python instantiates the input array as a contiguous `float32` NumPy array.
2.  **Mapping**: When passing the array to `engine.predict(input_data)`, the C++ layer retrieves the buffer's memory pointer.
3.  **Wrapping**: ONNX Runtime wraps the raw pointer directly in an `Ort::Value` tensor.
4.  **Inference**: ONNX Runtime executes the session, writing outputs directly into another pre-allocated C++ memory buffer.
5.  **Return**: The output buffer is returned back to Python wrapped as a NumPy array referencing the same C++ memory block.

This guarantees zero data copying during the autoregressive rollout loop, ensuring that performance is limited only by ONNX Runtime execution speed.

---

## 4. Dask Scheduling Guardrails

To prevent memory spikes when preparing inputs from massive remote datasets, WeatherGraph forces **synchronous Dask scheduling** inside `weathergraph/model.py`:

```python
import dask
dask.config.set(scheduler='synchronous')
```

### Why this is necessary
Standard Dask execution providers use threaded or distributed schedulers. When handling large-scale meteorology grids, these schedulers spawn parallel worker tasks that can load duplicate data chunks into RAM, causing immediate Out-of-Memory (OOM) failures on standard workstations. Forcing a synchronous scheduler guarantees that only one data chunk is processed at a time, keeping RAM consumption under strict, predictable control.