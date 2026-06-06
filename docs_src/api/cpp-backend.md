# C++ Backend API Concepts

While Python orchestrates data fetching, coordinate scaling, and folder exports, the performance-critical inference engine runs entirely in C++. This page details the pybind11 bindings that expose the C++ classes to the Python control plane.

---

## 1. Zero-Copy FFI Protocols

Performance-critical model runs require passing multidimensional arrays across the boundary between Python (NumPy) and C++ (ONNX Runtime) without copying data. WeatherGraph achieves this via the **pybind11 Buffer Protocol**:
*   The Python runtime instantiates float32 contiguous arrays using standard memory allocations.
*   When passing array references to C++ (e.g. via `engine.predict(input)`), C++ receives the buffer metadata containing dimensions, strides, and a raw pointer `float* data = static_cast<float*>(info.ptr)`.
*   ONNX Runtime wraps this raw data pointer inside an `Ort::Value` tensor object. This tensor directly references the NumPy memory allocation, ensuring that the GPU upload path is the only overhead.
*   Once model computation terminates, output pointers are returned back to Python wrapped as a new NumPy array that directly references the C++ memory block.

---

## 2. Exception Mapping & Safety

To prevent raw segmentation faults or C++ library crashes from crashing Python workflows, the C++ backend (`src/cpp/main.cpp`) registers translator callbacks mapping standard exceptions:

| C++ Exception | Translated Python Exception | Description |
| :--- | :--- | :--- |
| `Ort::Exception` | `RuntimeError("ONNX Runtime Error: ...")` | Failures during graph loading, operator mapping, or model execution. |
| `std::bad_alloc` | `RuntimeError("Memory Allocation Error (OOM): ...")` | Out-Of-Memory events on CPU host memory. |
| `std::out_of_range`| `IndexError("Out of Range Error: ...")` | Mismatch between shape definitions and buffer index slicing. |

---

## 3. `WeatherGraphEngine`

The main inference engine class, defined in C++ as `class WeatherGraphModel` and exposed to Python as `WeatherGraphEngine`.

### Constructor Definition
```cpp
WeatherGraphEngine(
    const std::string& model_path,
    int intra_op_threads = 1,
    bool disable_cpu_mem_arena = false,
    bool disable_mem_pattern = false,
    const std::string& execution_provider = "cpu",
    int execution_device_id = 0,
    uint64_t execution_memory_limit = 0,
    bool disable_cpu_ep_fallback = false,
    const ProviderOptions& execution_provider_options = ProviderOptions{},
    const std::string& constraints_model_path = ""
)
```

### Methods
*   **`output_shape()`** $\rightarrow$ `list[int]`: Returns the model's output shape (typically `[1, nodes, 78]`).
*   **`execution_provider()`** $\rightarrow$ `str`: Returns the name of the active ONNX Runtime execution provider (e.g. `"cuda"`).
*   **`cpu_mem_arena_enabled()`** $\rightarrow$ `bool`: Checks if the CPU arena allocator is active.
*   **`mem_pattern_enabled()`** $\rightarrow$ `bool`: Checks if operator execution path pre-allocation is enabled.
*   **`cpu_ep_fallback_enabled()`** $\rightarrow$ `bool`: Checks if the engine will fall back to CPU when an operator is unsupported on GPUs.
*   **`predict(input_data)`** $\rightarrow$ `numpy.ndarray`: Runs a single-step inference. The input must be a C-contiguous `float32` array with shape `[1, nodes, 78]`.
*   **`predict_ensemble(...)`** $\rightarrow$ `EnsembleResult`: Runs multi-member perturbed forecasts and aggregates statistics inside a single C++ call.

---

## 4. `EnsembleResult`

A data container class that stores real-time statistics computed by Welford's Algorithm inside the C++ layer.

### Attributes (Read-Only)
*   **`mean`** $\rightarrow$ `numpy.ndarray`: The accumulated ensemble mean array (shape: `[agg_steps, nodes, 78]`).
*   **`std_dev`** $\rightarrow$ `numpy.ndarray`: The accumulated ensemble standard deviation array (shape: `[agg_steps, nodes, 78]`).
*   **`probabilities`** $\rightarrow$ `dict[str, numpy.ndarray]`: A dictionary mapping named threshold rules to probability maps (shape: `[agg_steps, nodes]`).
*   **`total_members`** $\rightarrow$ `int`: Total number of perturbed ensemble members executed.
*   **`total_steps`** $\rightarrow$ `int`: Total number of forecast rollout steps.
*   **`aggregated_step_indices`** $\rightarrow$ `list[int]`: The specific step indices at which statistics were recorded.