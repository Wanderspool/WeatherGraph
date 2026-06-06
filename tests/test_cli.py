import json
import types

import numpy as np

import weathergraph.cli as cli


def test_cli_inspect_outputs_json_summary(monkeypatch, capsys):
    created = {}

    class StubModel:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.output_shape = (1, 4, 78)
            self.execution_provider = kwargs["execution_provider"]
            self.cpu_ep_fallback_enabled = kwargs["disable_cpu_ep_fallback"]
            self.cpu_mem_arena_enabled = not kwargs["disable_cpu_mem_arena"]
            self.mem_pattern_enabled = not kwargs["disable_mem_pattern"]
            self.reference_grid_shape = (2, 2)
            self.reference_grid_resolution_degrees = 90.0
            self.spatial_tiling = kwargs["spatial_tiling"]
            self.tile_state_backend = kwargs["tile_state_backend"]
            self.runtime_options = dict(kwargs)

        def estimate_state_bytes(self, node_count=None, buffers=1, dtype=np.float32):
            return 1248

        def estimate_tiled_memory_report(self):
            return {
                "reference_grid_shape": (2, 2),
                "reference_grid_resolution_degrees": 90.0,
                "reference_grid_node_count": 4,
                "global_state_bytes": 1248,
                "tile_state_backend": self.tile_state_backend,
                "max_tile_input_nodes": 3,
                "max_tile_output_nodes": 2,
                "max_tile_working_set_bytes": 1560,
                "tiles": [],
            }

    monkeypatch.setattr(cli, "WeatherGraphModel", StubModel)

    exit_code = cli.main(
        [
            "inspect",
            "--model-path",
            "model.onnx",
            "--weights-dir",
            "data",
            "--execution-provider",
            "cuda",
            "--execution-provider-options",
            '{"arena_extend_strategy":"kNextPowerOfTwo"}',
            "--spatial-tiling",
            "--tile-bundle-path",
            "tile_bundle/manifest.json",
            "--tile-state-backend",
            "memmap",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert created["execution_provider"] == "cuda"
    assert created["execution_provider_options"] == {"arena_extend_strategy": "kNextPowerOfTwo"}
    assert payload["output_shape"] == [1, 4, 78]
    assert payload["tile_memory_report"]["tile_state_backend"] == "memmap"
    assert payload["global_state_bytes"] == 1248


def test_cli_forecast_exports_with_source_arguments(monkeypatch, capsys):
    calls = {}

    class StubModel:
        def __init__(self, **kwargs):
            calls["model_kwargs"] = kwargs
            self.output_shape = (1, 4, 78)
            self.execution_provider = kwargs["execution_provider"]
            self.cpu_ep_fallback_enabled = kwargs["disable_cpu_ep_fallback"]
            self.cpu_mem_arena_enabled = True
            self.mem_pattern_enabled = True
            self.reference_grid_shape = None
            self.reference_grid_resolution_degrees = None
            self.spatial_tiling = kwargs["spatial_tiling"]
            self.tile_state_backend = kwargs["tile_state_backend"]
            self.runtime_options = dict(kwargs)

        def estimate_state_bytes(self, node_count=None, buffers=1, dtype=np.float32):
            return 1248

        def forecast_export(self, source, steps, output_path, fmt, t0=None):
            calls["export"] = {
                "source": source,
                "steps": steps,
                "output_path": output_path,
                "fmt": fmt,
                "t0": t0,
            }

    def fake_load_source(name, **kwargs):
        calls["source"] = {"name": name, "kwargs": kwargs}
        return types.SimpleNamespace(name=name)

    monkeypatch.setattr(cli, "WeatherGraphModel", StubModel)
    monkeypatch.setattr(cli, "load_source", fake_load_source)

    exit_code = cli.main(
        [
            "forecast",
            "--model-path",
            "model.onnx",
            "--weights-dir",
            "data",
            "--data-source",
            "gfs",
            "--source-arg",
            "date=2024-01-01 00:00",
            "--source-arg",
            "fxx=6",
            "--execution-provider-options",
            '{"trt_engine_cache_enable":true}',
            "--steps",
            "2",
            "--output-format",
            "npz",
            "--output-path",
            "forecast_out",
            "--start-time",
            "2024-01-01T00:00",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert calls["source"] == {"name": "gfs", "kwargs": {"date": "2024-01-01 00:00", "fxx": 6}}
    assert calls["model_kwargs"]["execution_provider_options"] == {"trt_engine_cache_enable": True}
    assert calls["export"]["steps"] == 2
    assert calls["export"]["fmt"] == "npz"
    assert payload["mode"] == "forecast_export"


def test_cli_forecast_one_step_without_export(monkeypatch, capsys):
    calls = {}

    class StubModel:
        def __init__(self, **kwargs):
            calls["model_kwargs"] = kwargs
            self.output_shape = (1, 4, 78)
            self.execution_provider = kwargs["execution_provider"]
            self.cpu_ep_fallback_enabled = kwargs["disable_cpu_ep_fallback"]
            self.cpu_mem_arena_enabled = True
            self.mem_pattern_enabled = True
            self.reference_grid_shape = None
            self.reference_grid_resolution_degrees = None
            self.spatial_tiling = kwargs["spatial_tiling"]
            self.tile_state_backend = kwargs["tile_state_backend"]
            self.runtime_options = dict(kwargs)

        def estimate_state_bytes(self, node_count=None, buffers=1, dtype=np.float32):
            return 1248

        def predict_one_step(self, source):
            calls["predict_source"] = source
            return np.zeros((1, 4, 78), dtype=np.float32)

    def fake_load_source(name, **kwargs):
        calls["source"] = {"name": name, "kwargs": kwargs}
        return types.SimpleNamespace(name=name)

    monkeypatch.setattr(cli, "WeatherGraphModel", StubModel)
    monkeypatch.setattr(cli, "load_source", fake_load_source)

    exit_code = cli.main(
        [
            "forecast",
            "--model-path",
            "model.onnx",
            "--weights-dir",
            "data",
            "--data-source",
            "era5_netcdf",
            "--input-path",
            "data/era5_archives/init.nc",
            "--steps",
            "1",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert calls["source"] == {"name": "era5_netcdf", "kwargs": {"path": "data/era5_archives/init.nc"}}
    assert payload["mode"] == "predict_one_step"
    assert payload["output_shape"] == [1, 4, 78]


def test_cli_visualize(monkeypatch, capsys, tmp_path):
    calls = {}

    def fake_create_interactive_map(ds, variable, time_index=0, cmap_name="viridis"):
        calls["html"] = {"variable": variable, "time_index": time_index, "cmap_name": cmap_name}
        import types
        return types.SimpleNamespace(save=lambda path: calls.update({"saved_html": path}))

    def fake_create_animation(ds, variable, output_path, format="mp4", cmap_name="viridis", fps=5, resolution="medium", **kwargs):
        calls["anim"] = {"variable": variable, "output_path": output_path, "format": format, "cmap_name": cmap_name, "fps": fps, "resolution": resolution}

    import xarray as xr
    def fake_open_dataset(path):
        return xr.Dataset()

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
    # The functions are imported directly in _cmd_visualize, so we need to patch them in the module where they are used.
    monkeypatch.setattr(cli, "create_interactive_map", fake_create_interactive_map, raising=False)
    monkeypatch.setattr(cli, "create_animation", fake_create_animation, raising=False)
    
    # We patch inside the sys.modules to catch the local import in _cmd_visualize
    import sys
    import types
    sys.modules["weathergraph.vis"] = types.SimpleNamespace(
        create_interactive_map=fake_create_interactive_map,
        create_animation=fake_create_animation
    )

    html_out = str(tmp_path / "out.html")
    exit_code_html = cli.main([
        "visualize",
        "--input", "dummy.nc",
        "--variable", "t",
        "--format", "html",
        "--output", html_out
    ])

    assert exit_code_html == 0
    assert calls["html"]["variable"] == "t"
    assert calls["saved_html"] == html_out

    mp4_out = str(tmp_path / "out.mp4")
    exit_code_mp4 = cli.main([
        "visualize",
        "--input", "dummy.nc",
        "--variable", "z",
        "--format", "mp4",
        "--output", mp4_out,
        "--fps", "10"
    ])

    assert exit_code_mp4 == 0
    assert calls["anim"]["variable"] == "z"
    assert calls["anim"]["format"] == "mp4"
    assert calls["anim"]["output_path"] == mp4_out
    assert calls["anim"]["fps"] == 10