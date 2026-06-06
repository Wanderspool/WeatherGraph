# Execution Providers (GPUs)

WeatherGraph delegates computation to ONNX Runtime backends using **Execution Providers (EPs)**. This architecture allows the same Python code to run on a variety of hardware accelerators—from consumer NVIDIA cards to AMD clusters, Intel rigs, or standard CPU workstations.

---

## 1. Supported Accelerators

The framework supports five execution providers:

| EP Name | Target Hardware | Compiler Extras |
| :--- | :--- | :--- |
| **`cpu`** | Standard x86/ARM processors | Pre-installed (Default) |
| **`cuda`** | NVIDIA GPUs (via CUDA and cuDNN) | `CMAKE_ARGS="-DONNXRUNTIME_CUDA=ON"` |
| **`tensorrt`**| NVIDIA TensorRT engine optimizer | `CMAKE_ARGS="-DONNXRUNTIME_TENSORRT=ON"` |
| **`rocm`** | AMD GPUs (via ROCm software stack) | `CMAKE_ARGS="-DONNXRUNTIME_ROCM=ON"` |
| **`openvino`**| Intel CPUs, Integrated GPUs, and VPUs | `CMAKE_ARGS="-DONNXRUNTIME_OPENVINO=ON"` |

---

## 2. Basic Configuration

To run a forecast on a GPU, set the `execution_provider` parameter during model initialization:

```python
import weathergraph as wg

model = wg.WeatherGraphModel(
    model_path="models/weather_gnn.onnx",
    weights_dir="data",
    execution_provider="cuda",  # Place computation on GPU
    execution_device_id=0       # Target the first GPU card
)
```

The C++ backend will automatically normalize common alias names:
*   `"nvidia"` $\rightarrow$ `"cuda"`
*   `"trt"` $\rightarrow$ `"tensorrt"`
*   `"amd"` $\rightarrow$ `"rocm"`
*   `"intel"` $\rightarrow$ `"openvino"`

---

## 3. Advanced Provider Options

You can tune execution providers by passing a dictionary or JSON string to `execution_provider_options`. These parameters are forwarded directly to ONNX Runtime:

```python
# Configure CUDA execution provider options
cuda_options = {
    "device_id": 0,
    "arena_extend_strategy": "kSameAsRequested",
    "gpu_mem_limit": 8 * 1024 * 1024 * 1024,  # Cap at 8 GB VRAM
    "cudnn_conv_algo_search": "DEFAULT",
    "do_copy_in_default_stream": True
}

model = wg.WeatherGraphModel(
    model_path="models/weather_gnn.onnx",
    weights_dir="data",
    execution_provider="cuda",
    execution_provider_options=cuda_options
)
```

---

## 4. Preventing CPU Fallbacks

By default, if ONNX Runtime encounters a GNN node operator that is not supported by the selected GPU provider, it silently falls back to executing that node on the host CPU. While this prevents crashes, copying tensors back and forth between CPU and GPU memory causes severe performance bottlenecks.

To enforce strict hardware placement, enable the CPU fallback guardrail:

```python
model = wg.WeatherGraphModel(
    model_path="models/weather_gnn.onnx",
    weights_dir="data",
    execution_provider="cuda",
    disable_cpu_ep_fallback=True  # Crash if any node falls back to CPU
)
```

If fallback occurs, session initialization will immediately raise a `ValueError` detailing which nodes failed to place on the GPU, allowing you to troubleshoot driver compatibilities or ONNX operator support.

---

## 5. Memory Capping

To manage shared resources in cloud environments or cluster setups, you can cap the maximum memory allocated by the accelerator using the `execution_memory_limit` parameter:

```python
model = wg.WeatherGraphModel(
    model_path="models/weather_gnn.onnx",
    weights_dir="data",
    execution_provider="cuda",
    execution_memory_limit=4 * 1024 * 1024 * 1024  # Limit to 4 GB VRAM
)
```

*   **CUDA/ROCm**: Sets the memory limit of the provider's allocator arena.
*   **TensorRT**: Maps directly to the TensorRT workspace limit, controlling the size of intermediate GPU layer buffers.