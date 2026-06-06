# weathergraph.tile_bundle

The `weathergraph.tile_bundle` module provides graph-partitioning utilities to split global meshes into smaller tiles for memory-efficient spatial tiled inference.

---

## Technical Overview: Exact Spatial Tiling

For high-resolution weather models, loading the entire global GNN mesh into workstation memory is highly demanding. WeatherGraph solves this via **Exact Spatial Tiling**:
1.  **Decomposition**: The global mesh nodes (representing coordinate cells) are partitioned into $N \times M$ non-overlapping output regions.
2.  **Halo Expansion**: To compute predictions for the boundaries of a partition, the model requires input context from surrounding cells. The module expands each tile's input indices by executing a breadth-first search (BFS) on the adjacency graph matching the GNN's receptive field:
    $$\mathcal{H}^{(k)}(v) = \{ u \in \mathcal{V} \mid d(u, v) \le k \}$$
    where $k$ represents `halo-hops` (message-passing layers) and $d(u,v)$ is the shortest path between nodes.
3.  **ONNX Model Splitting**: The global model weights are split into per-tile weights containing corresponding input/output shape signatures.
4.  **Manifest Serialization**: Metadata is compiled into a single `manifest.json` pointing to binary `.npy` arrays containing the index maps.

---

## Manifest JSON Schema

A generated tile bundle directory contains a `manifest.json` file structured as follows:

```json
{
  "reference_grid_shape": [181, 360],
  "reference_grid_resolution_degrees": 1.0,
  "global_input_shape": [1, 65160, 78],
  "global_output_shape": [1, 65160, 78],
  "tiles": [
    {
      "id": "tile_000",
      "model_path": "tiles/tile_000.onnx",
      "input_indices_path": "indices/tile_000_input.npy",
      "output_indices_path": "indices/tile_000_output.npy",
      "output_weights_path": "indices/tile_000_weights.npy"
    }
  ]
}
```

---

## Technical Example

The following script demonstrates how to generate a $2 \times 2$ tile bundle from a model's senders/receivers adjacency vectors:

```python
import weathergraph.tile_bundle as tb

# Build the tile bundle structure
summary = tb.build_tile_bundle(
    output_dir="data/tiled_bundle_2x2",
    senders_path="data/graph/senders.npy",
    receivers_path="data/graph/receivers.npy",
    tile_model_dir="models/tiled_onnx",
    tile_model_template="model_tile_{index}.onnx",
    reference_grid_shape=(181, 360),
    tile_grid_shape="2x2",
    halo_hops=2
)

print(f"Generated {summary['num_tiles']} tiles successfully!")
```

---

## Primary Builders

::: weathergraph.tile_bundle.build_tile_bundle
    options:
      show_source: true

---

## Partitioning Functions

::: weathergraph.tile_bundle.partition_regular_grid
::: weathergraph.tile_bundle.partition_contiguous_nodes

---

## Graph Geometry Helpers

::: weathergraph.tile_bundle.expand_input_halo
::: weathergraph.tile_bundle.build_receptive_field_adjacency
::: weathergraph.tile_bundle.load_edge_index

---

## Coordinate Resolution Utilities

::: weathergraph.tile_bundle.normalize_grid_shape
::: weathergraph.tile_bundle.grid_shape_from_resolution
::: weathergraph.tile_bundle.derive_reference_grid_resolution
::: weathergraph.tile_bundle.infer_global_node_count
