import json
import os
import tempfile

import numpy as np

import weathergraph.cli as cli
from weathergraph import WeatherGraphModel
from weathergraph.tile_bundle import build_tile_bundle


def _save_bidirectional_grid_edges(base_dir, rows, cols):
    senders = []
    receivers = []
    for row_index in range(rows):
        for col_index in range(cols):
            node_index = row_index * cols + col_index
            if col_index + 1 < cols:
                right_index = node_index + 1
                senders.extend([node_index, right_index])
                receivers.extend([right_index, node_index])
            if row_index + 1 < rows:
                down_index = node_index + cols
                senders.extend([node_index, down_index])
                receivers.extend([down_index, node_index])
    senders_path = os.path.join(base_dir, "senders.npy")
    receivers_path = os.path.join(base_dir, "receivers.npy")
    np.save(senders_path, np.asarray(senders, dtype=np.int64))
    np.save(receivers_path, np.asarray(receivers, dtype=np.int64))
    return senders_path, receivers_path


def test_build_tile_bundle_regular_grid_outputs_manifest_and_halo_indices():
    with tempfile.TemporaryDirectory() as tmpdir:
        senders_path, receivers_path = _save_bidirectional_grid_edges(tmpdir, rows=2, cols=2)
        model_dir = os.path.join(tmpdir, "tile_models")
        os.makedirs(model_dir, exist_ok=True)
        for tile_name in ["tile_000.onnx", "tile_001.onnx", "tile_002.onnx", "tile_003.onnx"]:
            with open(os.path.join(model_dir, tile_name), "wb") as handle:
                handle.write(b"placeholder")

        bundle_dir = os.path.join(tmpdir, "bundle")
        report = build_tile_bundle(
            output_dir=bundle_dir,
            senders_path=senders_path,
            receivers_path=receivers_path,
            tile_model_dir=model_dir,
            reference_grid_shape=(2, 2),
            tile_grid_shape=(1, 1),
            halo_hops=1,
        )

        with open(os.path.join(bundle_dir, "manifest.json"), "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        assert report["tile_count"] == 4
        assert report["global_node_count"] == 4
        assert report["reference_grid_shape"] == [2, 2]
        assert manifest["global_input_shape"] == [1, 4, 78]
        assert manifest["global_output_shape"] == [1, 4, 78]
        assert manifest["tile_partitioning"]["tile_grid_shape"] == [1, 1]

        tile_000 = manifest["tiles"][0]
        tile_000_output = np.load(os.path.join(bundle_dir, tile_000["output_indices_path"]))
        tile_000_input = np.load(os.path.join(bundle_dir, tile_000["input_indices_path"]))

        np.testing.assert_array_equal(tile_000_output, np.array([0], dtype=np.int64))
        np.testing.assert_array_equal(tile_000_input, np.array([0, 1, 2], dtype=np.int64))
        assert report["max_tile_input_nodes"] == 3
        assert report["max_tile_output_nodes"] == 1


def test_cli_build_tile_bundle_emits_summary_json(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        senders_path, receivers_path = _save_bidirectional_grid_edges(tmpdir, rows=2, cols=2)
        model_dir = os.path.join(tmpdir, "tile_models")
        os.makedirs(model_dir, exist_ok=True)
        for tile_name in ["tile_000.onnx", "tile_001.onnx", "tile_002.onnx", "tile_003.onnx"]:
            with open(os.path.join(model_dir, tile_name), "wb") as handle:
                handle.write(b"placeholder")

        bundle_dir = os.path.join(tmpdir, "bundle")
        exit_code = cli.main(
            [
                "build-tile-bundle",
                "--output-dir",
                bundle_dir,
                "--senders-path",
                senders_path,
                "--receivers-path",
                receivers_path,
                "--tile-model-dir",
                model_dir,
                "--reference-grid-shape",
                "2x2",
                "--tile-grid-shape",
                "1x1",
                "--json",
            ]
        )

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["tile_count"] == 4
        assert payload["global_node_count"] == 4
        assert payload["reference_grid_shape"] == [2, 2]
        assert os.path.exists(payload["manifest_path"])


def test_generated_tile_bundle_is_accepted_by_weathergraph_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        senders_path, receivers_path = _save_bidirectional_grid_edges(tmpdir, rows=2, cols=2)
        model_dir = os.path.join(tmpdir, "tile_models")
        os.makedirs(model_dir, exist_ok=True)
        for tile_name in ["tile_000.onnx", "tile_001.onnx", "tile_002.onnx", "tile_003.onnx"]:
            with open(os.path.join(model_dir, tile_name), "wb") as handle:
                handle.write(b"placeholder")

        np.save(os.path.join(tmpdir, "means.npy"), np.zeros(78, dtype=np.float32))
        np.save(os.path.join(tmpdir, "stds.npy"), np.ones(78, dtype=np.float32))

        bundle_dir = os.path.join(tmpdir, "bundle")
        build_tile_bundle(
            output_dir=bundle_dir,
            senders_path=senders_path,
            receivers_path=receivers_path,
            tile_model_dir=model_dir,
            reference_grid_shape=(2, 2),
            tile_grid_shape=(1, 1),
            halo_hops=1,
        )

        model = WeatherGraphModel(
            "unused_global_model.onnx",
            weights_dir=tmpdir,
            spatial_tiling=True,
            tile_bundle_path=bundle_dir,
        )

        assert model.output_shape == (1, 4, 78)
        assert model.reference_grid_shape == (2, 2)
        assert model.tile_bundle["tiles"][0]["input_indices"].shape[0] == 3