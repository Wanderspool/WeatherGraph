# Spatial Tiling & Large Grids

Global weather prediction models at high horizontal resolution (e.g. 0.1°) generate massive graph structures. A global grid at 0.1° resolution contains 6.48 million grid cells. Instantiating a GNN over this grid requires tens of gigabytes of memory, exceeding the capacity of standard consumer and workstation GPUs.

WeatherGraph solves this problem using **exact spatial tiling**. This guide explains the underlying partition mathematics, bundle manifests, and how to execute tiled forecasts.

---

## 1. What is Exact Tiling?

Unlike grid-based image cropping which introduces approximations at patch borders, WeatherGraph's C++ engine implements **graph-aware exact tiling**. 

For each spatial partition:
1.  **Owned Output Nodes**: The set of nodes for which this tile is responsible for generating final forecasts.
2.  **Receptive Field (Input Halo)**: The set of surrounding nodes required to compute the forecast for the owned nodes. The halo size is determined by the number of message-passing hops in the GNN model.

```text
    ┌───────────────────────────┐
    │ Input Halo (Receptive)    │
    │   ┌───────────────────┐   │
    │   │ Owned Outputs     │   │
    │   │                   │   │
    │   └───────────────────┘   │
    └───────────────────────────┘
```

By loading only the input halo nodes into memory, executing the prediction, and scattering the results to the owned output coordinates, WeatherGraph guarantees mathematically identical results to a global run—using only a fraction of the RAM.

---

## 2. Tested Configuration Example

To enable tiled inference, set `spatial_tiling=True` and point the engine to your tile bundle:

```python
--8<-- "tests/doc_examples/test_tiling.py:tiling_config"
```

---

## 3. The Tile Bundle Manifest (`manifest.json`)

A tile bundle is packaged as a directory containing a `manifest.json` and binary coordinate index files (`.npy` files).

### Manifest Structure
Below is a sample `manifest.json` layout:

```json
{
  "global_input_shape": [1, 64800, 78],
  "global_output_shape": [1, 64800, 78],
  "reference_grid_shape": [181, 360],
  "reference_grid_resolution_degrees": 1.0,
  "tiles": [
    {
      "id": "tile_000",
      "model_path": "onnx/tile_000.onnx",
      "input_indices_path": "indices/tile_000_input.npy",
      "output_indices_path": "indices/tile_000_output.npy",
      "output_weights_path": "indices/tile_000_weights.npy"
    },
    {
      "id": "tile_001",
      "model_path": "onnx/tile_001.onnx",
      "input_indices_path": "indices/tile_001_input.npy",
      "output_indices_path": "indices/tile_001_output.npy"
    }
  ]
}
```

### Manifest Fields
*   `global_output_shape`: The total size of the global tensor.
*   `tiles`: List of partition specifications:
    *   `model_path`: Relative path to the ONNX graphcompiled for this specific tile's partition shape.
    *   `input_indices_path`: Path to a 1D `int64` NumPy array file containing the global indices of all nodes in this tile's receptive field (owned + halo).
    *   `output_indices_path`: Path to a 1D `int64` NumPy array file containing the global indices of the output nodes owned by this tile.
    *   `output_weights_path` (Optional): Path to a 1D `float32` array containing weighting coefficients for stitching boundaries (used during [Halo Exchange](halo-exchange.md)).

---

## 4. Operational Pipeline Flow

When executing a forecast step, the tiled engine:
1.  Loads the global input state.
2.  Loops through the partitions sequentially:
    *   Slices the global input tensor using the tile's `input_indices` to extract the active partition.
    *   Loads/caches the ONNX engine for the tile and runs `predict(tile_input)`.
    *   Scatters the outputs into the global output tensor using the tile's `output_indices`.
3.  Replaces the global input tensor with the assembled global output tensor and repeats for the next autoregressive step.

This sequence guarantees that only one partition's active working set is loaded in CPU/GPU memory at any given moment, scaling forecasts to arbitrary grid sizes.