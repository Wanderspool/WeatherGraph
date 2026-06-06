# Model Conversion & Export

To execute weather forecasts, WeatherGraph requires an optimized, compiled ONNX graph. This guide describes the structural contract the ONNX model must satisfy and how to compile weights files into compliant ONNX graphs.

---

## 1. Graph Contract Constraints

The WeatherGraph engine enforces strict structural constraints on the ONNX graph at session load time. Any model that does not conform to this contract will fail validation during initialization:

```text
               ┌──────────────────────────────┐
               │    Input: "input_tensor"     │
               │      Shape: [1, nodes, 78]   │
               └──────────────┬───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │    GNN Model      │
                    │   Computation     │
                    └─────────┬─────────┘
                              │
               ┌──────────────▼───────────────┐
               │   Output: "output_tensor"    │
               │      Shape: [1, nodes, 78]   │
               └──────────────────────────────┘
```

### Specifications
*   **Batch Size**: Must be exactly `1`. Dynamic batching is not supported on the main autoregressive rollout path.
*   **Variable Dimension**: The final tensor dimension must contain exactly `78` channels, representing the fixed vertical profiles of the 6 core atmospheric variables.
*   **Node Dimension**: The number of nodes along the second dimension must be static or mapped dynamically. For spatial tiling, the tile node count must match the node count specified in the model partition file.

---

## 2. In-Graph vs. External Weights

Weather GNN models frequently contain hundreds of millions of parameters. 
*   **In-Graph Weights**: Models under 2 GB can store parameter weights directly inside the primary `.onnx` binary file.
*   **External Weights**: The ONNX standard has a 2 GB file size limit. If your GNN exceeds this limit, weights must be exported into external data files (e.g. `model.onnx.data`). These external data files must reside in the same directory as the `.onnx` model so that the C++ loader can locate them during session creation.

---

## 3. Compiling Weights via the CLI

If you have trained a weather model and serialized the weights as a pickle archive (`.pkl`), you can convert it to a compliant ONNX graph using the `compile-model` subcommand:

```bash
weathergraph compile-model \
  --weights-file ~/.cache/weathergraph/model.pkl \
  --output-file models/weather_gnn.onnx
```

### Under the Hood
1.  **Parsing**: The converter script loads the PyTorch state dictionary from the `.pkl` file.
2.  **Trimming**: Removes training-specific layers (e.g., optimizers, dropout rates, learning schedulers).
3.  **Trace**: Traces the forward execution loop using PyTorch's native `torch.onnx.export` utility.
4.  **Optimization**: Simplifies operator trees and executes constant-folding optimizations using `onnx` and `onnxruntime` tools.