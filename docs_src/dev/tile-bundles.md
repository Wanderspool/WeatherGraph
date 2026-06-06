# Generating Tile Bundles

To run forecasts over high-resolution grids without running out of memory, the global mesh must be partitioned into a **Tile Bundle**. This guide explains how to generate tile bundles using WeatherGraph's graph-partitioning tools.

---

## 1. Graph Mesh and Topology Inputs

Generating a tile bundle requires the adjacency list of the GNN's grid mesh. The GNN connectivity is represented by two 1D NumPy arrays stored on disk:
*   `senders.npy`: 1D array of sender node indices.
*   `receivers.npy`: 1D array of receiver node indices.

These arrays define the edges of the global graph. WeatherGraph uses this adjacency list to calculate which surrounding nodes (halo) must be included in each tile's receptive field to guarantee mathematically exact calculations.

---

## 2. Partitioning Strategies

WeatherGraph supports two strategies for dividing the global node mesh:

### Geographic Grid Partitioning (`--tile-grid-shape`)
Recommended for regular global lat/lon grids. You specify the number of partitions along the latitude and longitude axes:

```mermaid
grid
  ├─ Tile 0,0 ─┬─ Tile 0,1 ─┐
  ├─ Tile 1,0 ─┼─ Tile 1,1 ─┤
```

For example, a `--tile-grid-shape 2x2` command splits the world into four quadrants.

### Contiguous Node Clustering (`--tile-node-count`)
Recommended for irregular or unstructured meshes (e.g. icosahedral grids like ICON). The partitioning tool groups nodes based on graph connectivity metrics to minimize the boundary surface area, keeping the halo region as small as possible.

---

## 3. CLI Command Example

To build a tile bundle, use the `build-tile-bundle` subcommand. This parses your graph topology, computes partition boundaries, and generates the manifest and coordinate arrays:

```bash
weathergraph build-tile-bundle \
  --senders-path data/graph/senders.npy \
  --receivers-path data/graph/receivers.npy \
  --tile-model-dir models/tile_onnx/ \
  --tile-model-template "tile_{index}.onnx" \
  --reference-grid-shape 1801x3600 \
  --tile-grid-shape 4x8 \
  --halo-hops 2 \
  --halo-direction incoming \
  --output-dir tile_bundle/
```

### Key Parameter Definitions
*   `--halo-hops 2`: Include nodes up to 2 graph edges away from the partition boundary. The hop count must match the message-passing depth of your GNN.
*   `--halo-direction incoming`: Expand the halo based on incoming message vectors, ensuring that nodes contributing features to border outputs are included.
*   `--tile-model-template`: Maps each tile to its compiled ONNX graph file containing the correct input/output shape for that partition.

---

## 4. Generated Output Structure

The command writes a complete tile bundle directory at the specified `--output-dir` path:

```text
tile_bundle/
  ├── manifest.json              # Main bundle configuration
  └── indices/
        ├── tile_000_input.npy   # Input coordinates list (owned + halo)
        ├── tile_000_output.npy  # Output coordinates list (owned)
        ├── tile_001_input.npy
        └── ...
```

This bundle is fully prepared to be passed to `WeatherGraphModel` using `tile_bundle_path="tile_bundle/"`.