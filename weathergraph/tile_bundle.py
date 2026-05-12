from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def normalize_grid_shape(reference_grid_shape: str | tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    if reference_grid_shape is None:
        return None
    if isinstance(reference_grid_shape, str):
        normalized = reference_grid_shape.strip().lower().replace("x", ",")
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
    else:
        parts = list(reference_grid_shape)
    if len(parts) != 2:
        raise ValueError("reference_grid_shape must contain exactly two dimensions: (lat_count, lon_count).")
    lat_count = int(parts[0])
    lon_count = int(parts[1])
    if lat_count <= 0 or lon_count <= 0:
        raise ValueError("reference_grid_shape dimensions must be > 0.")
    return (lat_count, lon_count)


def grid_shape_from_resolution(reference_grid_resolution_degrees: float | str | None) -> tuple[int, int] | None:
    if reference_grid_resolution_degrees is None:
        return None
    resolution = float(reference_grid_resolution_degrees)
    if resolution <= 0.0:
        raise ValueError("reference_grid_resolution_degrees must be > 0.")
    lat_steps = round(180.0 / resolution)
    lon_steps = round(360.0 / resolution)
    if not np.isclose(lat_steps * resolution, 180.0, atol=1e-6):
        raise ValueError("reference_grid_resolution_degrees must divide 180 degrees exactly.")
    if not np.isclose(lon_steps * resolution, 360.0, atol=1e-6):
        raise ValueError("reference_grid_resolution_degrees must divide 360 degrees exactly.")
    return (int(lat_steps) + 1, int(lon_steps))


def derive_reference_grid_resolution(reference_grid_shape: tuple[int, int] | None) -> float | None:
    if reference_grid_shape is None:
        return None
    lat_count, lon_count = reference_grid_shape
    if lat_count <= 1 or lon_count <= 0:
        return None
    lat_resolution = 180.0 / float(lat_count - 1)
    lon_resolution = 360.0 / float(lon_count)
    if np.isclose(lat_resolution, lon_resolution, atol=1e-9):
        return float(lat_resolution)
    return None


def infer_global_node_count(
    num_nodes: int | None,
    reference_grid_shape: tuple[int, int] | None,
) -> int:
    if num_nodes is not None:
        inferred = int(num_nodes)
    elif reference_grid_shape is not None:
        inferred = int(reference_grid_shape[0] * reference_grid_shape[1])
    else:
        raise ValueError("Either num_nodes or reference_grid_shape/reference_grid_resolution_degrees is required.")
    if inferred <= 0:
        raise ValueError("num_nodes must be > 0.")
    return inferred


def load_edge_index(senders_path: str | Path, receivers_path: str | Path, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    senders = np.asarray(np.load(senders_path), dtype=np.int64)
    receivers = np.asarray(np.load(receivers_path), dtype=np.int64)
    if senders.ndim != 1 or receivers.ndim != 1:
        raise ValueError("senders and receivers must be 1D arrays.")
    if senders.shape[0] != receivers.shape[0]:
        raise ValueError("senders and receivers must have the same length.")
    if np.any(senders < 0) or np.any(senders >= num_nodes):
        raise ValueError("senders indices must lie within the global node range.")
    if np.any(receivers < 0) or np.any(receivers >= num_nodes):
        raise ValueError("receivers indices must lie within the global node range.")
    return senders, receivers


def build_receptive_field_adjacency(
    num_nodes: int,
    senders: np.ndarray,
    receivers: np.ndarray,
    halo_direction: str = "incoming",
) -> list[set[int]]:
    adjacency = [set() for _ in range(num_nodes)]
    for sender, receiver in zip(senders.tolist(), receivers.tolist()):
        adjacency[receiver].add(sender)
        if halo_direction == "undirected":
            adjacency[sender].add(receiver)
    return adjacency


def partition_regular_grid(reference_grid_shape: tuple[int, int], tile_grid_shape: tuple[int, int]) -> list[np.ndarray]:
    lat_count, lon_count = reference_grid_shape
    tile_lat_count, tile_lon_count = tile_grid_shape
    if tile_lat_count <= 0 or tile_lon_count <= 0:
        raise ValueError("tile_grid_shape dimensions must be > 0.")

    global_indices = np.arange(lat_count * lon_count, dtype=np.int64).reshape(lat_count, lon_count)
    partitions = []
    for lat_start in range(0, lat_count, tile_lat_count):
        for lon_start in range(0, lon_count, tile_lon_count):
            block = global_indices[
                lat_start:min(lat_start + tile_lat_count, lat_count),
                lon_start:min(lon_start + tile_lon_count, lon_count),
            ]
            partitions.append(np.ascontiguousarray(block.reshape(-1), dtype=np.int64))
    return partitions


def partition_contiguous_nodes(
    num_nodes: int,
    tile_node_count: int | None = None,
    tile_count: int | None = None,
) -> list[np.ndarray]:
    if tile_node_count is None and tile_count is None:
        raise ValueError("Either tile_node_count or tile_count is required for contiguous partitioning.")
    if tile_node_count is not None and tile_count is not None:
        raise ValueError("Specify either tile_node_count or tile_count, not both.")

    if tile_count is not None:
        if int(tile_count) <= 0:
            raise ValueError("tile_count must be > 0.")
        tile_node_count = int(math.ceil(num_nodes / float(tile_count)))

    tile_node_count = int(tile_node_count)
    if tile_node_count <= 0:
        raise ValueError("tile_node_count must be > 0.")

    partitions = []
    for start in range(0, num_nodes, tile_node_count):
        stop = min(start + tile_node_count, num_nodes)
        partitions.append(np.arange(start, stop, dtype=np.int64))
    return partitions


def expand_input_halo(output_indices: np.ndarray, adjacency: list[set[int]], halo_hops: int) -> np.ndarray:
    if halo_hops < 0:
        raise ValueError("halo_hops must be >= 0.")
    visited = set(int(index) for index in output_indices.tolist())
    frontier = set(visited)
    for _ in range(halo_hops):
        next_frontier: set[int] = set()
        for node_index in frontier:
            next_frontier.update(adjacency[node_index])
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return np.asarray(sorted(visited), dtype=np.int64)


def resolve_tile_model_paths(
    tile_count: int,
    tile_model_dir: str | Path,
    tile_model_template: str,
) -> list[dict[str, str]]:
    tile_model_dir = Path(tile_model_dir)
    resolved = []
    for index in range(tile_count):
        tile_id = f"tile_{index:03d}"
        model_name = tile_model_template.format(index=index, tile_id=tile_id)
        model_path = tile_model_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Tile model not found for {tile_id}: {model_path}")
        resolved.append(
            {
                "id": tile_id,
                "absolute_model_path": str(model_path.resolve()),
            }
        )
    return resolved


def os_path_relpath(path: Path, start: Path) -> str:
    return os.path.relpath(str(Path(path).resolve()), start=str(Path(start).resolve()))


def build_tile_bundle(
    *,
    output_dir: str | Path,
    senders_path: str | Path,
    receivers_path: str | Path,
    tile_model_dir: str | Path,
    tile_model_template: str = "{tile_id}.onnx",
    num_nodes: int | None = None,
    reference_grid_shape: str | tuple[int, int] | list[int] | None = None,
    reference_grid_resolution_degrees: float | str | None = None,
    tile_grid_shape: str | tuple[int, int] | list[int] | None = None,
    tile_node_count: int | None = None,
    tile_count: int | None = None,
    halo_hops: int = 1,
    halo_direction: str = "incoming",
) -> dict[str, Any]:
    if halo_direction not in {"incoming", "undirected"}:
        raise ValueError("halo_direction must be 'incoming' or 'undirected'.")

    normalized_reference_grid_shape = normalize_grid_shape(reference_grid_shape)
    derived_shape = grid_shape_from_resolution(reference_grid_resolution_degrees)
    if derived_shape is not None:
        if normalized_reference_grid_shape is not None and derived_shape != normalized_reference_grid_shape:
            raise ValueError(
                "reference_grid_shape and reference_grid_resolution_degrees must describe the same grid."
            )
        normalized_reference_grid_shape = derived_shape

    num_nodes = infer_global_node_count(num_nodes, normalized_reference_grid_shape)
    senders, receivers = load_edge_index(senders_path, receivers_path, num_nodes)
    adjacency = build_receptive_field_adjacency(num_nodes, senders, receivers, halo_direction=halo_direction)

    normalized_tile_grid_shape = normalize_grid_shape(tile_grid_shape)
    if normalized_tile_grid_shape is not None:
        if normalized_reference_grid_shape is None:
            raise ValueError("tile_grid_shape requires reference_grid_shape or reference_grid_resolution_degrees.")
        output_partitions = partition_regular_grid(normalized_reference_grid_shape, normalized_tile_grid_shape)
    else:
        output_partitions = partition_contiguous_nodes(num_nodes, tile_node_count=tile_node_count, tile_count=tile_count)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_models = resolve_tile_model_paths(
        len(output_partitions),
        tile_model_dir=tile_model_dir,
        tile_model_template=tile_model_template,
    )

    manifest_tiles = []
    max_tile_input_nodes = 0
    max_tile_output_nodes = 0
    max_tile_working_set_bytes = 0
    for index, output_indices in enumerate(output_partitions):
        tile_id = tile_models[index]["id"]
        output_indices = np.asarray(sorted(np.unique(output_indices).tolist()), dtype=np.int64)
        input_indices = expand_input_halo(output_indices, adjacency, halo_hops=halo_hops)

        input_indices_path = output_dir / f"{tile_id}_input.npy"
        output_indices_path = output_dir / f"{tile_id}_output.npy"
        np.save(input_indices_path, input_indices)
        np.save(output_indices_path, output_indices)

        input_state_bytes = int(input_indices.shape[0] * 78 * 4)
        output_state_bytes = int(output_indices.shape[0] * 78 * 4)
        max_tile_input_nodes = max(max_tile_input_nodes, int(input_indices.shape[0]))
        max_tile_output_nodes = max(max_tile_output_nodes, int(output_indices.shape[0]))
        max_tile_working_set_bytes = max(max_tile_working_set_bytes, input_state_bytes + output_state_bytes)
        manifest_tiles.append(
            {
                "id": tile_id,
                "model_path": os_path_relpath(Path(tile_models[index]["absolute_model_path"]), output_dir),
                "input_indices_path": input_indices_path.name,
                "output_indices_path": output_indices_path.name,
                "input_node_count": int(input_indices.shape[0]),
                "output_node_count": int(output_indices.shape[0]),
                "input_state_bytes": input_state_bytes,
                "output_state_bytes": output_state_bytes,
            }
        )

    reference_grid_resolution = (
        float(reference_grid_resolution_degrees)
        if reference_grid_resolution_degrees is not None
        else derive_reference_grid_resolution(normalized_reference_grid_shape)
    )
    manifest = {
        "global_input_shape": [1, int(num_nodes), 78],
        "global_output_shape": [1, int(num_nodes), 78],
        "reference_grid_shape": list(normalized_reference_grid_shape) if normalized_reference_grid_shape is not None else None,
        "reference_grid_resolution_degrees": reference_grid_resolution,
        "tile_partitioning": {
            "tile_grid_shape": list(normalized_tile_grid_shape) if normalized_tile_grid_shape is not None else None,
            "tile_node_count": int(tile_node_count) if tile_node_count is not None else None,
            "tile_count": int(tile_count) if tile_count is not None else len(manifest_tiles),
            "halo_hops": int(halo_hops),
            "halo_direction": halo_direction,
        },
        "tiles": manifest_tiles,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "manifest_path": str(manifest_path),
        "tile_count": len(manifest_tiles),
        "global_node_count": int(num_nodes),
        "reference_grid_shape": list(normalized_reference_grid_shape) if normalized_reference_grid_shape is not None else None,
        "reference_grid_resolution_degrees": reference_grid_resolution,
        "halo_hops": int(halo_hops),
        "halo_direction": halo_direction,
        "max_tile_input_nodes": max_tile_input_nodes,
        "max_tile_output_nodes": max_tile_output_nodes,
        "max_tile_working_set_bytes": max_tile_working_set_bytes,
    }