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
    op = helper.make_opsetid("", 14)
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
        assert engine.execution_provider() == "cpu"
        assert engine.cpu_ep_fallback_enabled() is True

def test_engine_rejects_unknown_execution_provider():
    """Verifies the backend rejects unsupported execution-provider names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "dummy_model.onnx")
        create_dummy_onnx(model_path)

        with pytest.raises(ValueError, match="execution_provider must be one of"):
            weathergraph_backend.WeatherGraphEngine(
                model_path,
                execution_provider="metal",
            )

def test_engine_rejects_cpu_fallback_disable_without_accelerator():
    """Verifies the backend validates accelerator-only fallback controls before session creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "dummy_model.onnx")
        create_dummy_onnx(model_path)

        with pytest.raises(ValueError, match="non-CPU execution_provider"):
            weathergraph_backend.WeatherGraphEngine(
                model_path,
                disable_cpu_ep_fallback=True,
            )

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
    
    assert output_data[0, 0, 0] == 0.0
    assert output_data[0, 1, 0] == 0.0

def test_model_prepare_input_is_contiguous_and_forwards_threads(monkeypatch):
    """Verifies the Python wrapper prepares contiguous float32 input and forwards thread configuration."""
    import weathergraph.model as model_module

    class RecordingEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False,
                     execution_provider="cpu",
                     execution_device_id=0,
                     execution_memory_limit=0,
                     disable_cpu_ep_fallback=False,
                     execution_provider_options=None,
                     **_kwargs):
            self.model_path = model_path
            self.intra_op_threads = intra_op_threads
            self.disable_cpu_mem_arena = disable_cpu_mem_arena
            self.disable_mem_pattern = disable_mem_pattern
            self.execution_provider_name = execution_provider
            self.execution_device_id = execution_device_id
            self.execution_memory_limit = execution_memory_limit
            self.disable_cpu_ep_fallback = disable_cpu_ep_fallback
            self.execution_provider_options = execution_provider_options or {}
            self._output_shape = (1, 4, 78)

        def output_shape(self):
            return self._output_shape

        def cpu_mem_arena_enabled(self):
            return not self.disable_cpu_mem_arena

        def mem_pattern_enabled(self):
            return not self.disable_mem_pattern

        def execution_provider(self):
            return self.execution_provider_name

        def cpu_ep_fallback_enabled(self):
            return not self.disable_cpu_ep_fallback

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

def test_model_forwards_multi_provider_runtime_configuration(monkeypatch):
    """Verifies the Python wrapper forwards multi-provider execution configuration to the backend."""
    import weathergraph.model as model_module

    class ProviderRecordingEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False,
                     execution_provider="cpu",
                     execution_device_id=0,
                     execution_memory_limit=0,
                     disable_cpu_ep_fallback=False,
                     execution_provider_options=None,
                     **_kwargs):
            self.model_path = model_path
            self.intra_op_threads = intra_op_threads
            self.disable_cpu_mem_arena = disable_cpu_mem_arena
            self.disable_mem_pattern = disable_mem_pattern
            self.execution_provider_name = execution_provider
            self.execution_device_id = execution_device_id
            self.execution_memory_limit = execution_memory_limit
            self.disable_cpu_ep_fallback = disable_cpu_ep_fallback
            self.execution_provider_options = execution_provider_options or {}

        def output_shape(self):
            return (1, 4, 78)

        def cpu_mem_arena_enabled(self):
            return not self.disable_cpu_mem_arena

        def mem_pattern_enabled(self):
            return not self.disable_mem_pattern

        def execution_provider(self):
            return self.execution_provider_name

        def cpu_ep_fallback_enabled(self):
            return not self.disable_cpu_ep_fallback

        def predict(self, input_data):
            return input_data

    monkeypatch.setattr(
        model_module,
        "weathergraph_backend",
        types.SimpleNamespace(WeatherGraphEngine=ProviderRecordingEngine),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "means.npy"), np.zeros(78, dtype=np.float32))
        np.save(os.path.join(tmpdir, "stds.npy"), np.ones(78, dtype=np.float32))

        model = WeatherGraphModel(
            "dummy.onnx",
            weights_dir=tmpdir,
            execution_provider="amd",
            execution_device_id=2,
            execution_memory_limit=1024,
            execution_provider_options={"tunable_op_enable": True, "arena_extend_strategy": 1},
            disable_cpu_ep_fallback=True,
        )

        assert model.engine.execution_provider_name == "rocm"
        assert model.engine.execution_device_id == 2
        assert model.engine.execution_memory_limit == 1024
        assert model.engine.disable_cpu_ep_fallback is True
        assert model.engine.execution_provider_options == {
            "tunable_op_enable": "true",
            "arena_extend_strategy": "1",
        }
        assert model.execution_provider == "rocm"
        assert model.cpu_ep_fallback_enabled is False
        assert model.runtime_options["execution_provider"] == "rocm"
        assert model.runtime_options["execution_device_id"] == 2
        assert model.runtime_options["execution_memory_limit"] == 1024
        assert model.runtime_options["execution_provider_options"] == {
            "tunable_op_enable": "true",
            "arena_extend_strategy": "1",
        }
        assert model.runtime_options["disable_cpu_ep_fallback"] is True
        assert model.runtime_options["cpu_ep_fallback_enabled"] is False

def test_model_accepts_legacy_cuda_runtime_aliases(monkeypatch):
    """Verifies wrapper-level aliases keep existing CUDA-oriented keyword names working."""
    import weathergraph.model as model_module

    class AliasRecordingEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False,
                     execution_provider="cpu",
                     execution_device_id=0,
                     execution_memory_limit=0,
                     disable_cpu_ep_fallback=False,
                     execution_provider_options=None,
                     **_kwargs):
            self.execution_provider_name = execution_provider
            self.execution_device_id = execution_device_id
            self.execution_memory_limit = execution_memory_limit
            self.execution_provider_options = execution_provider_options or {}

        def output_shape(self):
            return (1, 4, 78)

        def cpu_mem_arena_enabled(self):
            return True

        def mem_pattern_enabled(self):
            return True

        def execution_provider(self):
            return self.execution_provider_name

        def cpu_ep_fallback_enabled(self):
            return True

        def predict(self, input_data):
            return input_data

    monkeypatch.setattr(
        model_module,
        "weathergraph_backend",
        types.SimpleNamespace(WeatherGraphEngine=AliasRecordingEngine),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "means.npy"), np.zeros(78, dtype=np.float32))
        np.save(os.path.join(tmpdir, "stds.npy"), np.ones(78, dtype=np.float32))

        model = WeatherGraphModel(
            "dummy.onnx",
            weights_dir=tmpdir,
            execution_provider="nvidia",
            cuda_device_id=3,
            cuda_gpu_mem_limit=2048,
        )

        assert model.execution_provider == "cuda"
        assert model.engine.execution_device_id == 3
        assert model.engine.execution_memory_limit == 2048

def test_model_rejects_cpu_fallback_disable_without_accelerator():
    """Verifies wrapper-level validation rejects impossible CPU-fallback settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "means.npy"), np.zeros(78, dtype=np.float32))
        np.save(os.path.join(tmpdir, "stds.npy"), np.ones(78, dtype=np.float32))

        with pytest.raises(ValueError, match="non-CPU execution_provider"):
            WeatherGraphModel(
                "dummy.onnx",
                weights_dir=tmpdir,
                disable_cpu_ep_fallback=True,
            )

def test_model_rejects_provider_options_for_cpu():
    """Verifies wrapper-level validation rejects provider-specific options on the CPU path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "means.npy"), np.zeros(78, dtype=np.float32))
        np.save(os.path.join(tmpdir, "stds.npy"), np.ones(78, dtype=np.float32))

        with pytest.raises(ValueError, match="execution_provider_options require a non-CPU"):
            WeatherGraphModel(
                "dummy.onnx",
                weights_dir=tmpdir,
                execution_provider_options={"device_id": 1},
            )

def test_model_iter_forecast_streams_steps(monkeypatch):
    """Verifies iter_forecast yields each step without requiring full trajectory materialization."""
    import weathergraph.model as model_module

    class IncrementEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False,
                     execution_provider="cpu",
                     execution_device_id=0,
                     execution_memory_limit=0,
                     disable_cpu_ep_fallback=False,
                     execution_provider_options=None,
                     **_kwargs):
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
                     disable_mem_pattern=False,
                     execution_provider="cpu",
                     execution_device_id=0,
                     execution_memory_limit=0,
                     disable_cpu_ep_fallback=False,
                     execution_provider_options=None,
                     **_kwargs):
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

def test_model_supports_memmap_tiled_state_and_budget_report(monkeypatch):
    """Verifies tiled runs can materialize large global states via memmap and expose tile budget estimates."""
    import weathergraph.model as model_module

    class TileEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False,
                     execution_provider="cpu",
                     execution_device_id=0,
                     execution_memory_limit=0,
                     disable_cpu_ep_fallback=False,
                     execution_provider_options=None,
                     **_kwargs):
            self.model_path = model_path

        def output_shape(self):
            return (1, 2, 78)

        def cpu_mem_arena_enabled(self):
            return True

        def mem_pattern_enabled(self):
            return True

        def execution_provider(self):
            return "cpu"

        def cpu_ep_fallback_enabled(self):
            return True

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
                    "reference_grid_shape": [2, 2],
                    "reference_grid_resolution_degrees": 90.0,
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

        state_dir = os.path.join(tmpdir, "tile_state")
        model = WeatherGraphModel(
            "unused_global_model.onnx",
            weights_dir=tmpdir,
            spatial_tiling=True,
            tile_bundle_path=bundle_dir,
            tile_state_backend="memmap",
            tile_state_dir=state_dir,
        )

        input_data = np.arange(1 * 4 * 78, dtype=np.float32).reshape(1, 4, 78)
        output_data = model.engine.predict(input_data)
        report = model.estimate_tiled_memory_report()

        assert isinstance(output_data, np.memmap)
        np.testing.assert_allclose(output_data, input_data)
        assert report["reference_grid_shape"] == (2, 2)
        assert report["reference_grid_node_count"] == 4
        assert report["global_state_bytes"] == 1 * 4 * 78 * 4
        assert report["max_tile_input_nodes"] == 3
        assert report["max_tile_output_nodes"] == 2
        assert report["max_tile_working_set_bytes"] == (3 * 78 * 4) + (2 * 78 * 4)
        assert any(name.endswith(".dat") for name in os.listdir(state_dir))

def test_model_reference_grid_resolution_prepares_higher_resolution_exports(monkeypatch):
    """Verifies export/reference-grid metadata can be configured independently of the old 1° hardcoded shape."""
    import weathergraph.model as model_module

    class StubEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False,
                     execution_provider="cpu",
                     execution_device_id=0,
                     execution_memory_limit=0,
                     disable_cpu_ep_fallback=False,
                     execution_provider_options=None,
                     **_kwargs):
            self.model_path = model_path

        def output_shape(self):
            return (1, 6483600, 78)

        def cpu_mem_arena_enabled(self):
            return True

        def mem_pattern_enabled(self):
            return True

        def execution_provider(self):
            return "cpu"

        def cpu_ep_fallback_enabled(self):
            return True

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

        model = WeatherGraphModel(
            "dummy.onnx",
            weights_dir=tmpdir,
            reference_grid_resolution_degrees=0.1,
        )

        lat, lon = model._reference_grid_coordinates(dtype=np.float32)

        assert model.reference_grid_shape == (1801, 3600)
        assert model.reference_grid_node_count == 1801 * 3600
        assert model.reference_grid_resolution_degrees == pytest.approx(0.1)
        assert lat.shape == (1801,)
        assert lon.shape == (3600,)
        assert lon[-1] == pytest.approx(359.9)

def test_reference_grid_forecast_uses_configured_shape(monkeypatch):
    """Verifies the reference-grid reshape path honors configurable export geometry."""
    import weathergraph.model as model_module

    class StubEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False,
                     execution_provider="cpu",
                     execution_device_id=0,
                     execution_memory_limit=0,
                     disable_cpu_ep_fallback=False,
                     execution_provider_options=None,
                     **_kwargs):
            self.model_path = model_path

        def output_shape(self):
            return (1, 12, 78)

        def cpu_mem_arena_enabled(self):
            return True

        def mem_pattern_enabled(self):
            return True

        def execution_provider(self):
            return "cpu"

        def cpu_ep_fallback_enabled(self):
            return True

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

        model = WeatherGraphModel(
            "dummy.onnx",
            weights_dir=tmpdir,
            reference_grid_shape=(3, 4),
        )

        def fake_iter_forecast(initial_ds, steps):
            yield np.arange(1 * 12 * 78, dtype=np.float32).reshape(1, 12, 78)

        monkeypatch.setattr(model, "iter_forecast", fake_iter_forecast)

        step_index, era5_step = next(model._iter_reference_grid_forecast(initial_ds="ignored", steps=1))

        assert step_index == 0
        assert era5_step.shape == (3, 4, 78)

def test_forecast_export_uses_streaming_path(monkeypatch):
    """Verifies NetCDF export dispatches through the streaming path instead of materializing forecast()."""
    import weathergraph.model as model_module

    class StubEngine:
        def __init__(self,
                     model_path,
                     intra_op_threads=1,
                     disable_cpu_mem_arena=False,
                     disable_mem_pattern=False,
                     execution_provider="cpu",
                     execution_device_id=0,
                     execution_memory_limit=0,
                     disable_cpu_ep_fallback=False,
                     execution_provider_options=None,
                     **_kwargs):
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
                     disable_mem_pattern=False,
                     execution_provider="cpu",
                     execution_device_id=0,
                     execution_memory_limit=0,
                     disable_cpu_ep_fallback=False,
                     execution_provider_options=None,
                     **_kwargs):
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

def test_engine_supports_constraints_model():
    """Verifies the engine loads an optional constraints model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "dummy_model.onnx")
        create_dummy_onnx(model_path)
        constraints_path = os.path.join(tmpdir, "dummy_constraints.onnx")
        create_dummy_onnx(constraints_path)
        
        # Test constructor accepts constraints_model_path
        engine = weathergraph_backend.WeatherGraphEngine(
            model_path,
            constraints_model_path=constraints_path
        )
        
        input_data = np.random.randn(1, 71042, 78).astype(np.float32)
        output_data = engine.predict(input_data)
        assert output_data.shape == input_data.shape

def test_model_tiled_inference_with_halo_exchange(monkeypatch):
    """Verifies tiled inference supports output weights and accumulates overlapping margins correctly."""
    import weathergraph.model as model_module

    class TileEngine:
        def __init__(self, model_path, **kwargs):
            self.model_path = model_path
        def output_shape(self): return (1, 2, 78)
        def predict(self, input_data):
            # Output 2s to test accumulation
            return np.ones((1, 2, 78), dtype=np.float32) * 2.0
            
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
        # Tile 0: nodes 0, 1 (overlapping on 1)
        np.save(os.path.join(bundle_dir, "tile_0_input.npy"), np.array([0, 1], dtype=np.int64))
        np.save(os.path.join(bundle_dir, "tile_0_output.npy"), np.array([0, 1], dtype=np.int64))
        np.save(os.path.join(bundle_dir, "tile_0_weights.npy"), np.array([1.0, 0.5], dtype=np.float32))
        
        # Tile 1: nodes 1, 2 (overlapping on 1)
        np.save(os.path.join(bundle_dir, "tile_1_input.npy"), np.array([1, 2], dtype=np.int64))
        np.save(os.path.join(bundle_dir, "tile_1_output.npy"), np.array([1, 2], dtype=np.int64))
        np.save(os.path.join(bundle_dir, "tile_1_weights.npy"), np.array([0.5, 1.0], dtype=np.float32))
        
        with open(os.path.join(bundle_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "global_input_shape": [1, 3, 78],
                "global_output_shape": [1, 3, 78],
                "tiles": [
                    {
                        "id": "tile_0",
                        "model_path": "tile_0.onnx",
                        "input_indices_path": "tile_0_input.npy",
                        "output_indices_path": "tile_0_output.npy",
                        "output_weights_path": "tile_0_weights.npy",
                    },
                    {
                        "id": "tile_1",
                        "model_path": "tile_1.onnx",
                        "input_indices_path": "tile_1_input.npy",
                        "output_indices_path": "tile_1_output.npy",
                        "output_weights_path": "tile_1_weights.npy",
                    },
                ],
            }, handle)

        model = model_module.WeatherGraphModel(
            "unused_global_model.onnx",
            weights_dir=tmpdir,
            spatial_tiling=True,
            tile_bundle_path=bundle_dir,
        )
        input_data = np.zeros((1, 3, 78), dtype=np.float32)
        output_data = model.engine.predict(input_data)

        # Tile 0 output = 2.0 * [1.0, 0.5] = [2.0, 1.0]
        # Tile 1 output = 2.0 * [0.5, 1.0] = [1.0, 2.0]
        # Weight sum: Node 0=1.0, Node 1=1.0, Node 2=1.0
        # Output sum: Node 0=2.0, Node 1=(1.0+1.0)=2.0, Node 2=2.0
        # Final output = sum / weight_sum = [2.0, 2.0, 2.0]
        np.testing.assert_allclose(output_data[0, :, 0], [2.0, 2.0, 2.0])

