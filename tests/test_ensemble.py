"""Tests for the Probabilistic Core: ensemble inference with Welford
aggregation, per-channel Gaussian perturbation, and threshold probability maps.

These tests use mock engines (via monkeypatch) to exercise the full
Python→C++ contract without requiring a real ONNX model.
"""

import json
import os
import tempfile
import types

import numpy as np
import pytest
import xarray as xr

from weathergraph import WeatherGraphModel, EnsembleStats
import weathergraph_backend


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_dummy_onnx(path, output_shape=(1, 71042, 78)):
    """Create a dummy Identity ONNX model with matching input/output shapes."""
    import onnx
    import onnx.helper as helper
    from onnx import TensorProto

    nodes_dim = output_shape[1]
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, nodes_dim, 78])
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, list(output_shape))
    nodes = [helper.make_node('Identity', ['input'], ['output'])]
    op = helper.make_opsetid("", 14)
    graph_def = helper.make_graph(
        nodes, 'dummy', [input_tensor], [output_tensor])
    model_def = helper.make_model(
        graph_def, producer_name='dummy', opset_imports=[op])
    model_def.ir_version = 8
    onnx.save(model_def, path)


def make_small_onnx(path, nodes=4):
    """Create a small Identity ONNX model with configurable node count."""
    create_dummy_onnx(path, output_shape=(1, nodes, 78))


def _make_stub_engine_class(output_nodes=4):
    """Create a recording mock engine class for monkeypatching."""

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
            self._output_shape = (1, output_nodes, 78)

        def output_shape(self):
            return self._output_shape

        def cpu_mem_arena_enabled(self):
            return True

        def mem_pattern_enabled(self):
            return True

        def execution_provider(self):
            return "cpu"

        def cpu_ep_fallback_enabled(self):
            return True

        def predict(self, input_data):
            return np.ascontiguousarray(input_data + 1.0)

        def predict_ensemble(self, initial_state, steps=40, members=50,
                             channel_scales=None, threshold_channels=None,
                             threshold_values=None, threshold_ops=None,
                             threshold_names=None, aggregate_steps=None,
                             seed=0):
            """Mock ensemble that returns predictable values."""
            num_nodes = initial_state.shape[1]
            num_channels = initial_state.shape[2]

            agg_steps = list(aggregate_steps) if aggregate_steps else list(range(steps))
            agg_count = len(agg_steps)

            # Mean = initial + step_index + 1  (deterministic increments)
            mean = np.zeros((agg_count, num_nodes, num_channels), dtype=np.float32)
            std_dev = np.zeros((agg_count, num_nodes, num_channels), dtype=np.float32)
            for ai, s in enumerate(agg_steps):
                mean[ai, :, :] = initial_state[0] + s + 1

            probs = {}
            if threshold_names:
                for name in threshold_names:
                    probs[name] = np.full(
                        (agg_count, num_nodes), 0.5, dtype=np.float32)

            result = types.SimpleNamespace(
                mean=mean,
                std_dev=std_dev,
                probabilities=probs,
                total_members=members,
                total_steps=steps,
                aggregated_step_indices=agg_steps,
            )
            return result

    return StubEngine


def _make_model(monkeypatch, output_nodes=4, **model_kwargs):
    """Build a WeatherGraphModel with a mock engine."""
    import weathergraph.model as model_module
    StubEngine = _make_stub_engine_class(output_nodes)
    monkeypatch.setattr(
        model_module,
        "weathergraph_backend",
        types.SimpleNamespace(WeatherGraphEngine=StubEngine),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "means.npy"), np.zeros(78, dtype=np.float32))
        np.save(os.path.join(tmpdir, "stds.npy"), np.ones(78, dtype=np.float32))
        model = WeatherGraphModel("dummy.onnx", weights_dir=tmpdir, **model_kwargs)
        yield model


def _make_test_dataset(lat_count=2, lon_count=2):
    """Create a small xr.Dataset suitable for _prepare_input."""
    return xr.Dataset(
        {
            var: xr.DataArray(
                np.zeros((13, lat_count, lon_count), dtype=np.float32),
                dims=["level", "latitude", "longitude"],
                coords={
                    "level": [50, 100, 150, 200, 250, 300, 400,
                              500, 600, 700, 850, 925, 1000],
                    "latitude": np.linspace(90, -90, lat_count),
                    "longitude": np.linspace(0, 360, lon_count, endpoint=False),
                },
            )
            for var in ["z", "q", "t", "u", "v", "w"]
        }
    )


# ── Phase 1: C++ Ensemble Core Tests (via real ONNX) ─────────────────────────

class TestCppEnsembleCore:
    """Tests that exercise the C++ predict_ensemble directly via a dummy ONNX."""

    @pytest.fixture
    def engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "dummy.onnx")
            create_dummy_onnx(model_path, output_shape=(1, 4, 78))
            yield weathergraph_backend.WeatherGraphEngine(model_path)

    def test_deterministic_at_zero_noise(self, engine):
        """With all channel_scales=0, mean equals deterministic output
        and std_dev is 0.
        """
        initial = np.ones((1, 4, 78), dtype=np.float32) * 5.0
        scales = np.zeros(78, dtype=np.float32)

        result = engine.predict_ensemble(
            initial, steps=2, members=3,
            channel_scales=scales, seed=42)

        mean = np.asarray(result.mean)
        std = np.asarray(result.std_dev)

        # Identity model: output = input for every step
        assert mean.shape == (2, 4, 78)
        assert std.shape == (2, 4, 78)
        np.testing.assert_allclose(mean, 5.0, atol=1e-5)
        np.testing.assert_allclose(std, 0.0, atol=1e-5)

    def test_seed_reproducibility(self, engine):
        """Two calls with the same seed produce identical results."""
        initial = np.random.randn(1, 4, 78).astype(np.float32)
        scales = np.full(78, 0.1, dtype=np.float32)

        r1 = engine.predict_ensemble(
            initial, steps=2, members=5,
            channel_scales=scales, seed=123)
        r2 = engine.predict_ensemble(
            initial, steps=2, members=5,
            channel_scales=scales, seed=123)

        np.testing.assert_array_equal(
            np.asarray(r1.mean), np.asarray(r2.mean))
        np.testing.assert_array_equal(
            np.asarray(r1.std_dev), np.asarray(r2.std_dev))

    def test_seed_zero_nondeterministic(self, engine):
        """seed=0 produces different results on each call (with high probability)."""
        initial = np.ones((1, 4, 78), dtype=np.float32)
        scales = np.full(78, 0.5, dtype=np.float32)

        r1 = engine.predict_ensemble(
            initial, steps=1, members=10,
            channel_scales=scales, seed=0)
        r2 = engine.predict_ensemble(
            initial, steps=1, members=10,
            channel_scales=scales, seed=0)

        # Extremely unlikely to match with random seeds
        assert not np.array_equal(
            np.asarray(r1.mean), np.asarray(r2.mean))

    def test_channel_mask_preserves_static(self, engine):
        """Channels with scale=0 (geopotential) are identical across members."""
        initial = np.ones((1, 4, 78), dtype=np.float32) * 10.0
        scales = np.zeros(78, dtype=np.float32)
        # Only perturb channels 1-5 on each level, leave 0 (z) untouched
        for level_idx in range(13):
            for var_idx in range(1, 6):
                scales[level_idx * 6 + var_idx] = 1.0

        result = engine.predict_ensemble(
            initial, steps=1, members=20,
            channel_scales=scales, seed=42)

        std = np.asarray(result.std_dev)
        # z channels (index 0, 6, 12, ...) must have zero std
        for level_idx in range(13):
            z_ch = level_idx * 6
            np.testing.assert_allclose(
                std[0, :, z_ch], 0.0, atol=1e-6,
                err_msg=f"z channel at level index {level_idx} has nonzero std")

    def test_aggregate_steps_reduces_output(self, engine):
        """aggregate_steps parameter controls which steps appear in output."""
        initial = np.ones((1, 4, 78), dtype=np.float32)
        scales = np.zeros(78, dtype=np.float32)

        result = engine.predict_ensemble(
            initial, steps=5, members=2,
            channel_scales=scales,
            aggregate_steps=[1, 3], seed=42)

        mean = np.asarray(result.mean)
        assert mean.shape[0] == 2  # only 2 aggregated steps
        assert list(result.aggregated_step_indices) == [1, 3]

    def test_ensemble_result_fields(self, engine):
        """EnsembleResult exposes all expected fields."""
        initial = np.ones((1, 4, 78), dtype=np.float32)
        scales = np.zeros(78, dtype=np.float32)

        result = engine.predict_ensemble(
            initial, steps=3, members=5,
            channel_scales=scales, seed=1)

        assert result.total_members == 5
        assert result.total_steps == 3
        assert list(result.aggregated_step_indices) == [0, 1, 2]
        assert isinstance(result.probabilities, dict)


class TestCppThresholds:
    """Tests for threshold probability counting in C++."""

    @pytest.fixture
    def engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "dummy.onnx")
            create_dummy_onnx(model_path, output_shape=(1, 4, 78))
            yield weathergraph_backend.WeatherGraphEngine(model_path)

    def test_threshold_all_exceed(self, engine):
        """If all values > threshold, probability = 1.0."""
        initial = np.ones((1, 4, 78), dtype=np.float32) * 100.0
        scales = np.zeros(78, dtype=np.float32)

        result = engine.predict_ensemble(
            initial, steps=1, members=10,
            channel_scales=scales,
            threshold_channels=[0],
            threshold_values=[0.0],
            threshold_ops=[">"],
            threshold_names=["test_all"],
            seed=42)

        probs = dict(result.probabilities)
        prob = np.asarray(probs["test_all"])
        np.testing.assert_allclose(prob, 1.0)

    def test_threshold_none_exceed(self, engine):
        """If all values < threshold, probability = 0.0."""
        initial = np.ones((1, 4, 78), dtype=np.float32) * -100.0
        scales = np.zeros(78, dtype=np.float32)

        result = engine.predict_ensemble(
            initial, steps=1, members=10,
            channel_scales=scales,
            threshold_channels=[0],
            threshold_values=[0.0],
            threshold_ops=[">"],
            threshold_names=["test_none"],
            seed=42)

        probs = dict(result.probabilities)
        prob = np.asarray(probs["test_none"])
        np.testing.assert_allclose(prob, 0.0)

    def test_threshold_less_than_operator(self, engine):
        """Operator '<' works correctly."""
        initial = np.ones((1, 4, 78), dtype=np.float32) * -50.0
        scales = np.zeros(78, dtype=np.float32)

        result = engine.predict_ensemble(
            initial, steps=1, members=5,
            channel_scales=scales,
            threshold_channels=[2],
            threshold_values=[0.0],
            threshold_ops=["<"],
            threshold_names=["below_zero"],
            seed=42)

        probs = dict(result.probabilities)
        prob = np.asarray(probs["below_zero"])
        np.testing.assert_allclose(prob, 1.0)

    def test_multiple_threshold_rules(self, engine):
        """Multiple rules are computed independently."""
        initial = np.ones((1, 4, 78), dtype=np.float32) * 50.0
        scales = np.zeros(78, dtype=np.float32)

        result = engine.predict_ensemble(
            initial, steps=1, members=3,
            channel_scales=scales,
            threshold_channels=[0, 1],
            threshold_values=[0.0, 999.0],
            threshold_ops=[">", ">"],
            threshold_names=["high", "impossible"],
            seed=42)

        probs = dict(result.probabilities)
        np.testing.assert_allclose(np.asarray(probs["high"]), 1.0)
        np.testing.assert_allclose(np.asarray(probs["impossible"]), 0.0)


# ── Phase 2: Perturbation Engine Tests ────────────────────────────────────────

class TestPerturbation:
    """Tests for per-step perturbation behaviour."""

    @pytest.fixture
    def engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "dummy.onnx")
            create_dummy_onnx(model_path, output_shape=(1, 4, 78))
            yield weathergraph_backend.WeatherGraphEngine(model_path)

    def test_nonzero_noise_produces_spread(self, engine):
        """With nonzero scales, std_dev should be > 0."""
        initial = np.ones((1, 4, 78), dtype=np.float32)
        scales = np.full(78, 0.5, dtype=np.float32)

        result = engine.predict_ensemble(
            initial, steps=1, members=50,
            channel_scales=scales, seed=42)

        std = np.asarray(result.std_dev)
        assert np.mean(std) > 0.01, "Expected nonzero spread with noise"

    def test_welford_numerical_stability(self, engine):
        """Welford is stable with extreme values (no NaN/Inf in output)."""
        initial = np.ones((1, 4, 78), dtype=np.float32) * 1e7
        scales = np.full(78, 0.001, dtype=np.float32)

        result = engine.predict_ensemble(
            initial, steps=1, members=20,
            channel_scales=scales, seed=42)

        mean = np.asarray(result.mean)
        std = np.asarray(result.std_dev)
        assert not np.any(np.isnan(mean)), "NaN in mean"
        assert not np.any(np.isinf(mean)), "Inf in mean"
        assert not np.any(np.isnan(std)), "NaN in std_dev"
        assert not np.any(np.isinf(std)), "Inf in std_dev"


# ── Phase 3: Python Wrapper Tests ────────────────────────────────────────────

class TestChannelMapping:
    """Tests for _channel_index, _build_channel_scales, _parse_threshold_expr."""

    @pytest.fixture
    def model(self, monkeypatch):
        yield from _make_model(monkeypatch)

    def test_channel_index_mapping(self, model):
        """t at 850 hPa → level_idx=10 × 6 + var_idx=2 = 62."""
        assert model._channel_index("t", 850) == 62

    def test_channel_index_z_first(self, model):
        """z at first level (50 hPa) → index 0."""
        assert model._channel_index("z", 50) == 0

    def test_channel_index_w_last_level(self, model):
        """w at 1000 hPa → level_idx=12 × 6 + var_idx=5 = 77."""
        assert model._channel_index("w", 1000) == 77

    def test_build_channel_scales_dict(self, model):
        """Per-variable dict expands correctly."""
        scales = model._build_channel_scales({"t": 0.5, "q": 0.001})
        # t channels
        for level in model.levels:
            assert scales[model._channel_index("t", level)] == pytest.approx(0.5)
        # q channels
        for level in model.levels:
            assert scales[model._channel_index("q", level)] == pytest.approx(0.001)
        # z, u, v, w channels: 0
        for level in model.levels:
            assert scales[model._channel_index("z", level)] == 0.0
            assert scales[model._channel_index("u", level)] == 0.0

    def test_build_channel_scales_scalar(self, model):
        """Scalar σ applies to all dynamic channels except z."""
        scales = model._build_channel_scales(0.1)
        for level in model.levels:
            assert scales[model._channel_index("z", level)] == 0.0
            assert scales[model._channel_index("t", level)] == pytest.approx(0.1)
            assert scales[model._channel_index("q", level)] == pytest.approx(0.1)

    def test_build_channel_scales_none(self, model):
        """None → all zeros (no perturbation)."""
        scales = model._build_channel_scales(None)
        np.testing.assert_array_equal(scales, 0.0)

    def test_build_channel_scales_rejects_unknown_var(self, model):
        with pytest.raises(ValueError, match="Unknown variable"):
            model._build_channel_scales({"precipitation": 0.5})

    def test_parse_threshold_specific_level(self, model):
        """'t@850 < 273.15' parses to single rule."""
        rules = model._parse_threshold_expr("frost", "t@850 < 273.15")
        assert len(rules) == 1
        name, ch_idx, op, val = rules[0]
        assert name == "frost"
        assert ch_idx == 62
        assert op == "<"
        assert val == pytest.approx(273.15)

    def test_parse_threshold_all_levels(self, model):
        """'t < 250' expands to 13 rules (one per level)."""
        rules = model._parse_threshold_expr("cold", "t < 250.0")
        assert len(rules) == 13
        for name, ch_idx, op, val in rules:
            assert "cold@" in name
            assert op == "<"
            assert val == pytest.approx(250.0)

    def test_parse_threshold_greater_than(self, model):
        """'q@1000 > 0.015' with '>' operator."""
        rules = model._parse_threshold_expr("rain", "q@1000 > 0.015")
        assert len(rules) == 1
        assert rules[0][2] == ">"

    def test_parse_threshold_rejects_invalid(self, model):
        with pytest.raises(ValueError, match="Invalid threshold"):
            model._parse_threshold_expr("bad", "temperature is hot")

    def test_parse_threshold_rejects_unknown_var(self, model):
        with pytest.raises(ValueError, match="Unknown variable"):
            model._parse_threshold_expr("x", "x@850 > 0")

    def test_parse_threshold_rejects_unknown_level(self, model):
        with pytest.raises(ValueError, match="Unknown pressure level"):
            model._parse_threshold_expr("bad", "t@999 > 0")


class TestPredictEnsemble:
    """Tests for the full predict_ensemble Python method."""

    @pytest.fixture
    def model(self, monkeypatch):
        yield from _make_model(monkeypatch)

    def test_returns_ensemble_stats(self, model):
        """predict_ensemble returns an EnsembleStats instance."""
        ds = _make_test_dataset()
        stats = model.predict_ensemble(
            ds, steps=2, members=3,
            perturbation_scale=None, as_dataset=False)

        assert isinstance(stats, EnsembleStats)
        assert stats.members == 3
        assert stats.steps == 2
        assert isinstance(stats.mean, np.ndarray)
        assert isinstance(stats.std_dev, np.ndarray)

    def test_aggregate_steps_filtering(self, model):
        """aggregate_steps limits output time dimension."""
        ds = _make_test_dataset()
        stats = model.predict_ensemble(
            ds, steps=5, members=2,
            aggregate_steps=[1, 3], as_dataset=False)

        assert stats.mean.shape[0] == 2
        assert stats.aggregated_steps == [1, 3]

    def test_as_dataset_with_reference_grid(self, monkeypatch):
        """as_dataset=True with reference_grid returns xr.Dataset."""
        # Use a model with reference_grid_shape matching the node count
        gen = _make_model(monkeypatch, output_nodes=4,
                          reference_grid_shape=(2, 2))
        model = next(gen)
        ds = _make_test_dataset(lat_count=2, lon_count=2)

        stats = model.predict_ensemble(
            ds, steps=2, members=2,
            as_dataset=True)

        assert isinstance(stats.mean, xr.Dataset)
        assert isinstance(stats.std_dev, xr.Dataset)
        assert "t" in stats.mean
        assert stats.mean["t"].dims == ("time", "level", "lat", "lon")

    def test_thresholds_produce_probability_maps(self, model):
        """Threshold rules produce named probability entries."""
        ds = _make_test_dataset()
        stats = model.predict_ensemble(
            ds, steps=2, members=3,
            thresholds={"frost": "t@850 < 273.15"},
            as_dataset=False)

        assert "frost" in stats.probabilities
        prob = stats.probabilities["frost"]
        assert prob.shape[0] == 2  # 2 steps
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0

    def test_validation_rejects_bad_members(self, model):
        ds = _make_test_dataset()
        with pytest.raises(ValueError, match="members must be"):
            model.predict_ensemble(ds, steps=1, members=0)

    def test_validation_rejects_bad_steps(self, model):
        ds = _make_test_dataset()
        with pytest.raises(ValueError, match="steps must be"):
            model.predict_ensemble(ds, steps=0, members=1)


# ── Phase 4: CLI Tests ───────────────────────────────────────────────────────

class TestCLIEnsemble:
    """Tests for the ensemble CLI subcommand argument parsing."""

    def test_ensemble_subcommand_exists(self):
        from weathergraph.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "ensemble",
            "--model-path", "test.onnx",
            "--weights-dir", "data",
            "--steps", "10",
            "--members", "5",
            "--perturbation-scale", '{"t": 0.5}',
            "--threshold", "frost=t@850<273.15",
            "--threshold", "rain=q@1000>0.015",
            "--aggregate-steps", "4,9",
            "--seed", "42",
        ])
        assert args.command == "ensemble"
        assert args.steps == 10
        assert args.members == 5
        assert args.perturbation_scale == '{"t": 0.5}'
        assert len(args.threshold) == 2
        assert args.aggregate_steps == "4,9"
        assert args.seed == 42

    def test_ensemble_defaults(self):
        from weathergraph.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "ensemble",
            "--model-path", "test.onnx",
            "--weights-dir", "data",
        ])
        assert args.steps == 40
        assert args.members == 50
        assert args.seed == 0
        assert args.perturbation_scale is None
        assert args.output_format == "none"
