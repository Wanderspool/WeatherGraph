import json
import os
import tempfile
import types
import numpy as np
import pytest
import psutil
import onnx
import onnx.helper as helper
import xarray as xr
from onnx import TensorProto
from weathergraph import WeatherGraphModel
import weathergraph_backend

def create_dummy_onnx(path, output_shape=(1, 71042, 78)):
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 71042, 78])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, list(output_shape))
    if output_shape == (1, 71042, 78):
        nodes = [helper.make_node('Identity', ['input'], ['output'])]
        initializers = []
    else:
        assert output_shape[0] == 1
        assert output_shape[2] == 78
        starts = helper.make_tensor('starts', TensorProto.INT64, [3], [0, 0, 0])
        ends = helper.make_tensor('ends', TensorProto.INT64, [3], [1, output_shape[1], 78])
        axes = helper.make_tensor('axes', TensorProto.INT64, [3], [0, 1, 2])
        steps = helper.make_tensor('steps', TensorProto.INT64, [3], [1, 1, 1])
        nodes = [helper.make_node('Slice', ['input', 'starts', 'ends', 'axes', 'steps'], ['output'])]
        initializers = [starts, ends, axes, steps]
    op = helper.make_opsetid("ai.onnx", 14)
    graph_def = helper.make_graph(nodes, 'dummy', [input_tensor], [output_tensor], initializer=initializers)
    model_def = helper.make_model(graph_def, producer_name='dummy', opset_imports=[op])
    model_def.ir_version = 8
    onnx.save(model_def, path)

@pytest.fixture
def mock_engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "dummy_model.onnx")
        create_dummy_onnx(model_path)
        yield weathergraph_backend.WeatherGraphEngine(model_path)

def test_engine_inference_zero_copy(mock_engine):
    """Verifies that the engine performs inference correctly and returns expected shapes."""
    input_data = np.random.randn(1, 71042, 78).astype(np.float32)
    output_data = mock_engine.predict(input_data)
    
    assert output_data.shape == (1, 71042, 78)
    np.testing.assert_allclose(input_data, output_data)

def test_engine_uses_model_output_shape():
    """Verifies the backend allocates output buffers from ONNX metadata rather than mirroring the input shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "dummy_model.onnx")
        create_dummy_onnx(model_path, output_shape=(1, 70000, 78))
        engine = weathergraph_backend.WeatherGraphEngine(model_path)

        input_data = np.random.randn(1, 71042, 78).astype(np.float32)
        output_data = engine.predict(input_data)

        assert output_data.shape == (1, 70000, 78)
        np.testing.assert_allclose(output_data, input_data[:, :70000, :])

def test_engine_accepts_thread_configuration():
    """Verifies the backend constructor accepts an explicit intra-op thread count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "dummy_model.onnx")
        create_dummy_onnx(model_path)
        engine = weathergraph_backend.WeatherGraphEngine(model_path, intra_op_threads=2)

        input_data = np.random.randn(1, 71042, 78).astype(np.float32)
        output_data = engine.predict(input_data)

        assert output_data.shape == input_data.shape

def test_engine_defaults_keep_ort_allocators_enabled():
    """Verifies low-memory ORT flags stay off by default so existing behavior is preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "dummy_model.onnx")
        create_dummy_onnx(model_path)
        engine = weathergraph_backend.WeatherGraphEngine(model_path)

        assert engine.cpu_mem_arena_enabled() is True
        assert engine.mem_pattern_enabled() is True

def test_engine_can_disable_ort_arena_and_mem_pattern():
    """Verifies the backend exposes optional low-memory ORT session knobs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "dummy_model.onnx")
        create_dummy_onnx(model_path)
        engine = weathergraph_backend.WeatherGraphEngine(
            model_path,
            disable_cpu_mem_arena=True,
            disable_mem_pattern=True,
        )

        input_data = np.random.randn(1, 71042, 78).astype(np.float32)
        output_data = engine.predict(input_data)

        assert engine.cpu_mem_arena_enabled() is False
        assert engine.mem_pattern_enabled() is False
        assert output_data.shape == input_data.shape

def test_engine_reports_output_shape():
    """Verifies the backend exposes ONNX output metadata to the Python wrapper."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "dummy_model.onnx")
        create_dummy_onnx(model_path, output_shape=(1, 70000, 78))
        engine = weathergraph_backend.WeatherGraphEngine(model_path)

        assert tuple(engine.output_shape()) == (1, 70000, 78)

def test_memory_safety_non_contiguous(mock_engine):
    """Verifies the engine rejects non-contiguous arrays, preventing C++ segfaults."""
    input_data = np.random.randn(1, 71042, 78).astype(np.float32)
    non_contiguous = input_data[:, ::-1, :]
    
    assert not non_contiguous.flags.c_contiguous
    
    with pytest.raises(RuntimeError, match="C-contiguous"):
        mock_engine.predict(non_contiguous)

def test_robustness_nan_inf(mock_engine):
    """Verifies the engine processes NaN/Inf without throwing C++ Floating Point Exceptions."""
    input_data = np.random.randn(1, 71042, 78).astype(np.float32)
    input_data[0, 0, 0] = np.nan
    input_data[0, 1, 0] = np.inf
    
    output_data = mock_engine.predict(input_data)
    
    assert np.isnan(output_data[0, 0, 0])
    assert np.isinf(output_data[0, 1, 0])

def test_model_prepare_input_is_contiguous_and_forwards_threads(monkeypatch):
    """Verifies the Python wrapper prepares contiguous float32 input and forwards thread configuration."""
    import weathergraph.model as model_module

    class RecordingEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False):
            self.model_path = model_path
            self.intra_op_threads = intra_op_threads
            self.disable_cpu_mem_arena = disable_cpu_mem_arena
            self.disable_mem_pattern = disable_mem_pattern
            self._output_shape = (1, 4, 78)

        def output_shape(self):
            return self._output_shape

        def cpu_mem_arena_enabled(self):
            return not self.disable_cpu_mem_arena

        def mem_pattern_enabled(self):
            return not self.disable_mem_pattern

        def predict(self, input_data):
            return input_data

    monkeypatch.setattr(
        model_module,
        "weathergraph_backend",
        types.SimpleNamespace(WeatherGraphEngine=RecordingEngine),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "means.npy"), np.zeros(78, dtype=np.float32))
        np.save(os.path.join(tmpdir, "stds.npy"), np.ones(78, dtype=np.float32))

        ds = xr.Dataset(
            {
                var: xr.DataArray(
                    (np.arange(13 * 4, dtype=np.float32).reshape(13, 2, 2) + idx),
                    dims=["level", "latitude", "longitude"],
                    coords={
                        "level": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
                        "latitude": [10.0, 20.0],
                        "longitude": [30.0, 40.0],
                    },
                )
                for idx, var in enumerate(["z", "q", "t", "u", "v", "w"])
            }
        )

        model = WeatherGraphModel(
            "dummy.onnx",
            weights_dir=tmpdir,
            intra_op_threads=3,
            disable_cpu_mem_arena=True,
            disable_mem_pattern=True,
        )
        prepared = model._prepare_input(ds)

        assert model.engine.intra_op_threads == 3
        assert model.engine.disable_cpu_mem_arena is True
        assert model.engine.disable_mem_pattern is True
        assert model.cpu_mem_arena_enabled is False
        assert model.mem_pattern_enabled is False
        assert prepared.shape == (1, 4, 78)
        assert prepared.dtype == np.float32
        assert prepared.flags.c_contiguous

def test_model_iter_forecast_streams_steps(monkeypatch):
    """Verifies iter_forecast yields each step without requiring full trajectory materialization."""
    import weathergraph.model as model_module

    class IncrementEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False):
            self.model_path = model_path
            self.intra_op_threads = intra_op_threads
            self.disable_cpu_mem_arena = disable_cpu_mem_arena
            self.disable_mem_pattern = disable_mem_pattern

        def output_shape(self):
            return (1, 4, 78)

        def cpu_mem_arena_enabled(self):
            return not self.disable_cpu_mem_arena

        def mem_pattern_enabled(self):
            return not self.disable_mem_pattern

        def predict(self, input_data):
            return np.ascontiguousarray(input_data + 1.0)

    monkeypatch.setattr(
        model_module,
        "weathergraph_backend",
        types.SimpleNamespace(WeatherGraphEngine=IncrementEngine),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "means.npy"), np.zeros(78, dtype=np.float32))
        np.save(os.path.join(tmpdir, "stds.npy"), np.ones(78, dtype=np.float32))

        ds = xr.Dataset(
            {
                var: xr.DataArray(
                    np.zeros((13, 2, 2), dtype=np.float32),
                    dims=["level", "latitude", "longitude"],
                    coords={
                        "level": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
                        "latitude": [10.0, 20.0],
                        "longitude": [30.0, 40.0],
                    },
                )
                for var in ["z", "q", "t", "u", "v", "w"]
            }
        )

        model = WeatherGraphModel("dummy.onnx", weights_dir=tmpdir)
        steps = list(model.iter_forecast(ds, steps=3))

        assert len(steps) == 3
        np.testing.assert_allclose(steps[0], 1.0)
        np.testing.assert_allclose(steps[1], 2.0)
        np.testing.assert_allclose(steps[2], 3.0)

def test_model_requires_tile_bundle_for_spatial_tiling():
    """Verifies exact tiled inference is opt-in and requires explicit bundle metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "means.npy"), np.zeros(78, dtype=np.float32))
        np.save(os.path.join(tmpdir, "stds.npy"), np.ones(78, dtype=np.float32))

        with pytest.raises(ValueError, match="tile_bundle_path"):
            WeatherGraphModel("dummy.onnx", weights_dir=tmpdir, spatial_tiling=True)

def test_model_runs_exact_tiled_prediction_from_bundle(monkeypatch):
    """Verifies tiled inference stitches exact outputs from tile-local ONNX models."""
    import weathergraph.model as model_module

    created_paths = []

    class TileEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False):
            self.model_path = model_path
            self.intra_op_threads = intra_op_threads
            self.disable_cpu_mem_arena = disable_cpu_mem_arena
            self.disable_mem_pattern = disable_mem_pattern
            created_paths.append(os.path.basename(model_path))

        def output_shape(self):
            return (1, 2, 78)

        def cpu_mem_arena_enabled(self):
            return not self.disable_cpu_mem_arena

        def mem_pattern_enabled(self):
            return not self.disable_mem_pattern

        def predict(self, input_data):
            if os.path.basename(self.model_path) == "tile_0.onnx":
                return np.ascontiguousarray(input_data[:, :2, :])
            return np.ascontiguousarray(input_data[:, 1:, :])

    monkeypatch.setattr(
        model_module,
        "weathergraph_backend",
        types.SimpleNamespace(WeatherGraphEngine=TileEngine),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "means.npy"), np.zeros(78, dtype=np.float32))
        np.save(os.path.join(tmpdir, "stds.npy"), np.ones(78, dtype=np.float32))

        bundle_dir = os.path.join(tmpdir, "tile_bundle")
        os.makedirs(bundle_dir, exist_ok=True)
        np.save(os.path.join(bundle_dir, "tile_0_input.npy"), np.array([0, 1, 2], dtype=np.int64))
        np.save(os.path.join(bundle_dir, "tile_0_output.npy"), np.array([0, 1], dtype=np.int64))
        np.save(os.path.join(bundle_dir, "tile_1_input.npy"), np.array([1, 2, 3], dtype=np.int64))
        np.save(os.path.join(bundle_dir, "tile_1_output.npy"), np.array([2, 3], dtype=np.int64))
        with open(os.path.join(bundle_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "global_input_shape": [1, 4, 78],
                    "global_output_shape": [1, 4, 78],
                    "tiles": [
                        {
                            "id": "tile_0",
                            "model_path": "tile_0.onnx",
                            "input_indices_path": "tile_0_input.npy",
                            "output_indices_path": "tile_0_output.npy",
                        },
                        {
                            "id": "tile_1",
                            "model_path": "tile_1.onnx",
                            "input_indices_path": "tile_1_input.npy",
                            "output_indices_path": "tile_1_output.npy",
                        },
                    ],
                },
                handle,
            )

        model = WeatherGraphModel(
            "unused_global_model.onnx",
            weights_dir=tmpdir,
            disable_cpu_mem_arena=True,
            disable_mem_pattern=True,
            spatial_tiling=True,
            tile_bundle_path=bundle_dir,
        )
        input_data = np.arange(1 * 4 * 78, dtype=np.float32).reshape(1, 4, 78)

        assert created_paths == []

        output_data = model.engine.predict(input_data)
        second_output = model.engine.predict(input_data)

        np.testing.assert_allclose(output_data, input_data)
        np.testing.assert_allclose(second_output, input_data)
        assert created_paths == ["tile_0.onnx", "tile_1.onnx"]
        assert model.cpu_mem_arena_enabled is False
        assert model.mem_pattern_enabled is False

def test_forecast_export_uses_streaming_path(monkeypatch):
    """Verifies NetCDF export dispatches through the streaming path instead of materializing forecast()."""
    import weathergraph.model as model_module

    class StubEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False):
            self.model_path = model_path
            self.intra_op_threads = intra_op_threads
            self.disable_cpu_mem_arena = disable_cpu_mem_arena
            self.disable_mem_pattern = disable_mem_pattern

        def output_shape(self):
            return (1, 65160, 78)

        def cpu_mem_arena_enabled(self):
            return not self.disable_cpu_mem_arena

        def mem_pattern_enabled(self):
            return not self.disable_mem_pattern

        def predict(self, input_data):
            return input_data

    monkeypatch.setattr(
        model_module,
        "weathergraph_backend",
        types.SimpleNamespace(WeatherGraphEngine=StubEngine),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "means.npy"), np.zeros(78, dtype=np.float32))
        np.save(os.path.join(tmpdir, "stds.npy"), np.ones(78, dtype=np.float32))

        model = WeatherGraphModel("dummy.onnx", weights_dir=tmpdir)

        def fail_forecast(*args, **kwargs):
            raise AssertionError("forecast() should not be used by streaming export")

        stream_calls = {}

        def fake_iter_reference_grid_forecast(initial_ds, steps):
            for step_index in range(steps):
                yield step_index, np.zeros((181, 360, 78), dtype=np.float32) + step_index

        def fake_stream_netcdf_export(output_path, times, lat, lon, steps, step_iterator):
            stream_calls["output_path"] = output_path
            stream_calls["steps"] = steps
            stream_calls["seen"] = [step_index for step_index, _ in step_iterator]

        monkeypatch.setattr(model, "forecast", fail_forecast)
        monkeypatch.setattr(model, "_resolve_dataset", lambda source: source)
        monkeypatch.setattr(model, "_iter_reference_grid_forecast", fake_iter_reference_grid_forecast)
        monkeypatch.setattr(model, "_stream_netcdf_export", fake_stream_netcdf_export)

        model.forecast_export(initial_ds="ignored", steps=2, output_path=os.path.join(tmpdir, "forecast"), fmt="netcdf4")

        assert stream_calls["steps"] == 2
        assert stream_calls["seen"] == [0, 1]

def test_model_rejects_latent_output_artifacts(monkeypatch):
    """Verifies the Python wrapper fails fast on latent-output prototype artifacts."""
    import weathergraph.model as model_module

    class LatentEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False):
            self.model_path = model_path
            self.intra_op_threads = intra_op_threads
            self.disable_cpu_mem_arena = disable_cpu_mem_arena
            self.disable_mem_pattern = disable_mem_pattern

        def output_shape(self):
            return (5882, 256)

        def cpu_mem_arena_enabled(self):
            return not self.disable_cpu_mem_arena

        def mem_pattern_enabled(self):
            return not self.disable_mem_pattern

    monkeypatch.setattr(
        model_module,
        "weathergraph_backend",
        types.SimpleNamespace(WeatherGraphEngine=LatentEngine),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "means.npy"), np.zeros(78, dtype=np.float32))
        np.save(os.path.join(tmpdir, "stds.npy"), np.ones(78, dtype=np.float32))

        with pytest.raises(ValueError, match="autoregressive ONNX artifact"):
            WeatherGraphModel("latent.onnx", weights_dir=tmpdir)
