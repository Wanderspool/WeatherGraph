from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import REGISTRY, WeatherGraphModel, list_sources, load_source
from .tile_bundle import build_tile_bundle
from .utils import get_default_cache_dir, download_file
from pathlib import Path


def _coerce_scalar(value: str) -> Any:
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    try:
        return int(normalized)
    except ValueError:
        pass

    try:
        return float(normalized)
    except ValueError:
        pass

    if normalized.startswith(("{", "[", '"')):
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            return normalized

    return normalized


def _parse_key_value_pairs(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"Invalid KEY=VALUE argument: {item}")
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError(f"Invalid KEY=VALUE argument: {item}")
        parsed[normalized_key] = _coerce_scalar(value)
    return parsed


def _parse_json_object(raw: str | None, label: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    normalized = raw.strip()
    if not normalized:
        return None
    parsed = json.loads(normalized)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to a JSON object.")
    return parsed


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default="models/weather_gnn.onnx", help="Path to the ONNX artifact.")
    parser.add_argument("--weights-dir", default="data", help="Directory containing means.npy and stds.npy.")
    parser.add_argument("--intra-op-threads", type=int, default=1, help="ONNX Runtime intra-op thread count.")
    parser.add_argument("--execution-provider", default="cpu", help="Preferred ONNX Runtime execution provider.")
    parser.add_argument("--execution-device-id", type=int, default=0, help="Accelerator device ordinal.")
    parser.add_argument("--execution-memory-limit", type=int, default=0, help="Provider memory cap in bytes.")
    parser.add_argument(
        "--execution-provider-options",
        default=None,
        help="JSON object string forwarded to the selected execution provider.",
    )
    parser.add_argument(
        "--disable-cpu-ep-fallback",
        action="store_true",
        help="Fail if any node would silently fall back to the CPU execution provider.",
    )
    parser.add_argument(
        "--disable-cpu-mem-arena",
        action="store_true",
        help="Disable ONNX Runtime's CPU arena allocator.",
    )
    parser.add_argument(
        "--disable-mem-pattern",
        action="store_true",
        help="Disable ONNX Runtime memory-pattern reuse.",
    )
    parser.add_argument(
        "--reference-grid-shape",
        default=None,
        help="Reference-grid shape as LATxLON or LAT,LON.",
    )
    parser.add_argument(
        "--reference-grid-resolution-degrees",
        type=float,
        default=None,
        help="Regular global reference-grid resolution in degrees, for example 0.25 or 0.1.",
    )
    parser.add_argument(
        "--spatial-tiling",
        action="store_true",
        help="Enable exact graph-aware tiled inference.",
    )
    parser.add_argument("--tile-bundle-path", default=None, help="Tile bundle manifest path or directory.")
    parser.add_argument(
        "--tile-state-backend",
        choices=["ram", "memmap"],
        default="ram",
        help="Backend used for global tiled-state buffers.",
    )
    parser.add_argument(
        "--tile-state-dir",
        default=None,
        help="Directory for memmap-backed tiled-state buffers.",
    )


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-source",
        default="era5_netcdf",
        choices=sorted(REGISTRY.keys()),
        help="Registered WeatherGraph data-source adapter.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Convenience alias for source arg path=... when using file-backed sources.",
    )
    parser.add_argument(
        "--source-kwargs",
        default=None,
        help="JSON object string forwarded to load_source().",
    )
    parser.add_argument(
        "--source-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional load_source() keyword arguments. May be repeated.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weathergraph",
        description="Researcher-facing CLI for WeatherGraph runtime inspection and forecasting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-sources", help="List registered data-source adapters.")
    list_parser.set_defaults(func=_cmd_list_sources)

    bundle_parser = subparsers.add_parser(
        "build-tile-bundle",
        help="Build a tile-bundle manifest plus input/output index arrays from graph topology.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bundle_parser.add_argument("--output-dir", required=True, help="Directory where the bundle manifest and index arrays are written.")
    bundle_parser.add_argument("--senders-path", required=True, help="Path to the global graph senders.npy array.")
    bundle_parser.add_argument("--receivers-path", required=True, help="Path to the global graph receivers.npy array.")
    bundle_parser.add_argument("--tile-model-dir", required=True, help="Directory containing per-tile ONNX artifacts.")
    bundle_parser.add_argument(
        "--tile-model-template",
        default="{tile_id}.onnx",
        help="Filename template inside tile-model-dir. Available placeholders: {index}, {tile_id}.",
    )
    bundle_parser.add_argument("--num-nodes", type=int, default=None, help="Global node count when no reference-grid shape is provided.")
    bundle_parser.add_argument("--reference-grid-shape", default=None, help="Global reference-grid shape as LATxLON or LAT,LON.")
    bundle_parser.add_argument(
        "--reference-grid-resolution-degrees",
        type=float,
        default=None,
        help="Regular global reference-grid resolution in degrees, for example 0.25 or 0.1.",
    )
    partition_group = bundle_parser.add_mutually_exclusive_group(required=True)
    partition_group.add_argument("--tile-grid-shape", default=None, help="Regular-grid tile shape as LATxLON or LAT,LON.")
    partition_group.add_argument("--tile-node-count", type=int, default=None, help="Contiguous node count per tile.")
    partition_group.add_argument("--tile-count", type=int, default=None, help="Contiguous tile count when no regular-grid tiling is requested.")
    bundle_parser.add_argument("--halo-hops", type=int, default=1, help="Number of graph hops included in each tile input halo.")
    bundle_parser.add_argument(
        "--halo-direction",
        choices=["incoming", "undirected"],
        default="incoming",
        help="How the receptive-field halo is expanded from each tile's owned output nodes.",
    )
    bundle_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    bundle_parser.set_defaults(func=_cmd_build_tile_bundle)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Construct the runtime and print a summary plus memory sizing information.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_runtime_arguments(inspect_parser)
    inspect_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    inspect_parser.set_defaults(func=_cmd_inspect)

    forecast_parser = subparsers.add_parser(
        "forecast",
        help="Run one-step inference, iterative rollout, or streaming export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_runtime_arguments(forecast_parser)
    _add_source_arguments(forecast_parser)
    forecast_parser.add_argument("--steps", type=int, default=40, help="Number of 6-hour autoregressive steps.")
    forecast_parser.add_argument(
        "--output-format",
        choices=["none", "netcdf4", "zarr", "npz"],
        default="none",
        help="Export format. Use 'none' to keep results in-process and print a summary only.",
    )
    forecast_parser.add_argument("--output-path", default=None, help="Export destination for netcdf4, zarr, or npz.")
    forecast_parser.add_argument("--start-time", default=None, help="Optional ISO-8601 start time for exported forecasts.")
    forecast_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    forecast_parser.set_defaults(func=_cmd_forecast)

    vis_parser = subparsers.add_parser(
        "visualize",
        help="Generate Leaflet HTML maps or MP4/GIF animations from a saved forecast.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    vis_parser.add_argument("--input", required=True, help="Path to the saved forecast NetCDF dataset.")
    vis_parser.add_argument("--variable", required=True, help="Variable to visualize (e.g., 't', 'z').")
    vis_parser.add_argument(
        "--format",
        choices=["html", "mp4", "gif"],
        default="mp4",
        help="Output artifact format.",
    )
    vis_parser.add_argument("--output", required=True, help="Path to save the generated artifact.")
    vis_parser.add_argument("--cmap", default="viridis", help="Colormap name.")
    vis_parser.add_argument("--time-index", type=int, default=0, help="Time index for HTML map generation.")
    vis_parser.add_argument("--resolution", choices=["high", "medium", "low"], default="medium", help="Output resolution for animations.")
    vis_parser.set_defaults(func=_cmd_visualize)

    # ── Ensemble subcommand ──
    ensemble_parser = subparsers.add_parser(
        "ensemble",
        help="Run O(1)-memory ensemble inference with Welford aggregation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_runtime_arguments(ensemble_parser)
    _add_source_arguments(ensemble_parser)
    ensemble_parser.add_argument("--steps", type=int, default=40, help="Number of 6-hour autoregressive steps.")
    ensemble_parser.add_argument("--members", type=int, default=50, help="Number of ensemble members.")
    ensemble_parser.add_argument(
        "--perturbation-scale",
        default=None,
        help='Per-variable σ as JSON (e.g. \'{"t": 0.5, "q": 0.001}\') or a scalar float.',
    )
    ensemble_parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="NAME=EXPR",
        help='Named threshold rule.  Example: frost=t@850<273.15.  May be repeated.',
    )
    ensemble_parser.add_argument(
        "--aggregate-steps",
        default=None,
        help="Comma-separated step indices to aggregate (e.g. 9,19,29,39). Default: all steps.",
    )
    ensemble_parser.add_argument("--seed", type=int, default=0, help="PRNG seed (0 = non-deterministic).")
    ensemble_parser.add_argument(
        "--output-format",
        choices=["none", "netcdf4", "zarr"],
        default="none",
        help="Export format for ensemble mean/std_dev.",
    )
    ensemble_parser.add_argument("--output-path", default=None, help="Export destination directory.")
    ensemble_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    ensemble_parser.set_defaults(func=_cmd_ensemble)

    # ── New subcommands ──
    dl_parser = subparsers.add_parser("download-model", help="Download a model from a URL.")
    dl_parser.add_argument("--model-url", required=True, help="URL to download.")
    dl_parser.add_argument("--work-dir", default=None, help="Directory to save the model. Defaults to OS cache dir.")
    dl_parser.add_argument("--output-filename", default="model.pkl", help="Name of the saved file.")
    dl_parser.set_defaults(func=_cmd_download_model)

    comp_parser = subparsers.add_parser("compile-model", help="Compile a .pkl model into ONNX.")
    comp_parser.add_argument("--weights-file", required=True, help="Input .pkl weights file.")
    comp_parser.add_argument("--output-file", required=True, help="Output .onnx model.")
    comp_parser.set_defaults(func=_cmd_compile_model)

    pipe_parser = subparsers.add_parser("pipeline", help="Run the full pipeline: download, compile, forecast, visualize.")
    pipe_parser.add_argument("--work-dir", default=None, help="Directory for all artifacts. Defaults to OS cache dir.")
    pipe_parser.add_argument("--model-url", required=True, help="URL to the original model or ONNX model.")
    _add_source_arguments(pipe_parser)
    _add_runtime_arguments(pipe_parser)
    pipe_parser.add_argument("--steps", type=int, default=1, help="Number of 6-hour autoregressive steps.")
    pipe_parser.add_argument("--vis-variable", default="t", help="Variable to visualize (e.g. 't', 'z').")
    pipe_parser.add_argument("--vis-resolution", choices=["high", "medium", "low"], default="medium", help="Resolution for visualization.")
    pipe_parser.set_defaults(func=_cmd_pipeline)

    return parser


def build_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    provider_options = _parse_json_object(args.execution_provider_options, "execution-provider-options")
    return {
        "model_path": args.model_path,
        "weights_dir": args.weights_dir,
        "intra_op_threads": args.intra_op_threads,
        "disable_cpu_mem_arena": args.disable_cpu_mem_arena,
        "disable_mem_pattern": args.disable_mem_pattern,
        "execution_provider": args.execution_provider,
        "execution_device_id": args.execution_device_id,
        "execution_memory_limit": args.execution_memory_limit,
        "execution_provider_options": provider_options,
        "disable_cpu_ep_fallback": args.disable_cpu_ep_fallback,
        "reference_grid_shape": args.reference_grid_shape,
        "reference_grid_resolution_degrees": args.reference_grid_resolution_degrees,
        "spatial_tiling": args.spatial_tiling,
        "tile_bundle_path": args.tile_bundle_path,
        "tile_state_backend": args.tile_state_backend,
        "tile_state_dir": args.tile_state_dir,
    }


def build_model(args: argparse.Namespace) -> WeatherGraphModel:
    return WeatherGraphModel(**build_model_kwargs(args))


def build_source(args: argparse.Namespace) -> Any:
    source_kwargs = _parse_json_object(args.source_kwargs, "source-kwargs") or {}
    source_kwargs.update(_parse_key_value_pairs(args.source_arg))
    if args.input_path is not None and "path" not in source_kwargs:
        source_kwargs["path"] = args.input_path
    return load_source(args.data_source, **source_kwargs)


def model_summary(model: WeatherGraphModel) -> dict[str, Any]:
    summary = {
        "output_shape": list(model.output_shape),
        "execution_provider": model.execution_provider,
        "cpu_ep_fallback_enabled": model.cpu_ep_fallback_enabled,
        "cpu_mem_arena_enabled": model.cpu_mem_arena_enabled,
        "mem_pattern_enabled": model.mem_pattern_enabled,
        "reference_grid_shape": list(model.reference_grid_shape) if model.reference_grid_shape is not None else None,
        "reference_grid_resolution_degrees": model.reference_grid_resolution_degrees,
        "spatial_tiling": model.spatial_tiling,
        "tile_state_backend": model.tile_state_backend,
        "global_state_bytes": model.estimate_state_bytes(),
        "runtime_options": model.runtime_options,
    }
    if model.spatial_tiling:
        summary["tile_memory_report"] = model.estimate_tiled_memory_report()
    return summary


def _emit_result(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    for key, value in result.items():
        print(f"{key}: {value}")


def _cmd_list_sources(_args: argparse.Namespace) -> int:
    list_sources()
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    summary = model_summary(build_model(args))
    _emit_result(summary, args.json)
    return 0


def _cmd_build_tile_bundle(args: argparse.Namespace) -> int:
    result = build_tile_bundle(
        output_dir=args.output_dir,
        senders_path=args.senders_path,
        receivers_path=args.receivers_path,
        tile_model_dir=args.tile_model_dir,
        tile_model_template=args.tile_model_template,
        num_nodes=args.num_nodes,
        reference_grid_shape=args.reference_grid_shape,
        reference_grid_resolution_degrees=args.reference_grid_resolution_degrees,
        tile_grid_shape=args.tile_grid_shape,
        tile_node_count=args.tile_node_count,
        tile_count=args.tile_count,
        halo_hops=args.halo_hops,
        halo_direction=args.halo_direction,
    )
    _emit_result(result, args.json)
    return 0


def _cmd_forecast(args: argparse.Namespace) -> int:
    if args.steps <= 0:
        raise ValueError("steps must be >= 1.")

    model = build_model(args)
    source = build_source(args)

    if args.output_format == "none":
        if args.steps == 1:
            output = model.predict_one_step(source)
            result = {
                "mode": "predict_one_step",
                "output_shape": list(output.shape),
                "execution_provider": model.execution_provider,
            }
        else:
            final_output = None
            for final_output in model.iter_forecast(source, steps=args.steps):
                pass
            result = {
                "mode": "iter_forecast",
                "steps": args.steps,
                "final_output_shape": list(final_output.shape),
                "execution_provider": model.execution_provider,
            }
        _emit_result(result, args.json)
        return 0

    if not args.output_path:
        raise ValueError("output-path is required when output-format is not 'none'.")

    model.forecast_export(
        source,
        steps=args.steps,
        output_path=args.output_path,
        fmt=args.output_format,
        t0=args.start_time,
    )
    _emit_result(
        {
            "mode": "forecast_export",
            "steps": args.steps,
            "output_format": args.output_format,
            "output_path": args.output_path,
        },
        args.json,
    )
    return 0


def _cmd_ensemble(args: argparse.Namespace) -> int:
    if args.steps <= 0:
        raise ValueError("steps must be >= 1.")
    if args.members <= 0:
        raise ValueError("members must be >= 1.")

    model = build_model(args)
    source = build_source(args)

    # Parse perturbation scale
    perturbation_scale = None
    if args.perturbation_scale is not None:
        raw = args.perturbation_scale.strip()
        if raw.startswith("{"):
            perturbation_scale = json.loads(raw)
        else:
            perturbation_scale = float(raw)

    # Parse thresholds: NAME=EXPR
    thresholds = None
    if args.threshold:
        thresholds = {}
        for item in args.threshold:
            name, sep, expr = item.partition("=")
            if not sep:
                raise ValueError(f"Invalid --threshold format: {item}. Expected NAME=EXPR.")
            thresholds[name.strip()] = expr.strip()

    # Parse aggregate steps
    aggregate_steps = None
    if args.aggregate_steps is not None:
        aggregate_steps = [
            int(s.strip()) for s in args.aggregate_steps.split(",") if s.strip()
        ]

    stats = model.predict_ensemble(
        initial_ds=source,
        steps=args.steps,
        members=args.members,
        perturbation_scale=perturbation_scale,
        thresholds=thresholds,
        aggregate_steps=aggregate_steps,
        seed=args.seed,
        as_dataset=(args.output_format != "none"),
    )

    result = {
        "mode": "predict_ensemble",
        "members": stats.members,
        "steps": stats.steps,
        "aggregated_steps": stats.aggregated_steps,
        "probability_rules": list(stats.probabilities.keys()) if stats.probabilities else [],
    }

    if args.output_format != "none" and args.output_path:
        import os
        os.makedirs(args.output_path, exist_ok=True)
        if hasattr(stats.mean, 'to_netcdf'):
            if args.output_format == "netcdf4":
                stats.mean.to_netcdf(os.path.join(args.output_path, "ensemble_mean.nc"))
                stats.std_dev.to_netcdf(os.path.join(args.output_path, "ensemble_std_dev.nc"))
            elif args.output_format == "zarr":
                stats.mean.to_zarr(os.path.join(args.output_path, "ensemble_mean.zarr"), mode="w")
                stats.std_dev.to_zarr(os.path.join(args.output_path, "ensemble_std_dev.zarr"), mode="w")
            for name, da in stats.probabilities.items():
                safe_name = name.replace("@", "_at_").replace(" ", "_")
                if args.output_format == "netcdf4":
                    da.to_netcdf(os.path.join(args.output_path, f"prob_{safe_name}.nc"))
                elif args.output_format == "zarr":
                    da.to_zarr(os.path.join(args.output_path, f"prob_{safe_name}.zarr"), mode="w")
        result["output_path"] = args.output_path
        result["output_format"] = args.output_format

    _emit_result(result, args.json)
    return 0


def _cmd_visualize(args: argparse.Namespace) -> int:
    import xarray as xr
    from .vis import create_interactive_map, create_animation

    print(f"Loading dataset from {args.input}...")
    ds = xr.open_dataset(args.input)

    if args.format == "html":
        print(f"Generating interactive map for '{args.variable}' at step {args.time_index}...")
        m = create_interactive_map(ds, args.variable, time_index=args.time_index, cmap_name=args.cmap)
        m.save(args.output)
        print(f"Map saved to {args.output}")
    else:
        print(f"Generating {args.format.upper()} animation for '{args.variable}'...")
        create_animation(ds, args.variable, args.output, format=args.format, cmap_name=args.cmap, fps=args.fps, resolution=getattr(args, "resolution", "medium"))
        print(f"Animation saved to {args.output}")

    return 0

def _cmd_download_model(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir) if args.work_dir else get_default_cache_dir()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / args.output_filename
    download_file(args.model_url, out_path)
    print(f"Model downloaded to {out_path}")
    return 0

def _cmd_compile_model(args: argparse.Namespace) -> int:
    try:
        from exporter.convert_to_onnx import convert_to_onnx
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from exporter.convert_to_onnx import convert_to_onnx

    convert_to_onnx(args.weights_file, args.output_file)
    return 0

def _cmd_pipeline(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir) if args.work_dir else get_default_cache_dir()
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using working directory: {work_dir}")

    # 1. Download Model
    is_onnx_url = args.model_url.endswith(".onnx")
    model_filename = "model.onnx" if is_onnx_url else "model.pkl"
    local_model_path = work_dir / model_filename
    if not local_model_path.exists():
        print(f"Downloading model from {args.model_url}...")
        download_file(args.model_url, local_model_path)
    else:
        print(f"Model already exists at {local_model_path}, skipping download.")

    # 2. Compile to ONNX if necessary
    if not is_onnx_url:
        onnx_path = work_dir / "model.onnx"
        if not onnx_path.exists():
            print(f"Compiling {local_model_path} to {onnx_path}...")
            # Reuse compile_model logic
            args.weights_file = str(local_model_path)
            args.output_file = str(onnx_path)
            _cmd_compile_model(args)
        else:
            print(f"Compiled ONNX model already exists at {onnx_path}.")
        model_path_for_inference = str(onnx_path)
    else:
        model_path_for_inference = str(local_model_path)

    # 3. Forecast
    print("Running forecast...")
    args.model_path = model_path_for_inference
    args.output_format = "netcdf4"
    args.output_path = str(work_dir / "forecast.nc")
    _cmd_forecast(args)

    # 4. Visualize
    print("Generating visualization...")
    # Setup args for visualization
    args.input = args.output_path
    args.variable = args.vis_variable
    args.format = "mp4"
    args.output = str(work_dir / f"forecast_{args.vis_variable}.mp4")
    args.cmap = "viridis"
    args.fps = 5
    args.resolution = args.vis_resolution
    _cmd_visualize(args)

    print(f"Pipeline complete! All artifacts saved to {work_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    try:
        from .utils import get_default_cache_dir, load_env_file, prompt_for_credentials
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from weathergraph.utils import get_default_cache_dir, load_env_file, prompt_for_credentials

    # Determine work_dir to load .env
    work_dir = Path(args.work_dir) if hasattr(args, 'work_dir') and args.work_dir else get_default_cache_dir()
    load_env_file(work_dir / ".env")

    try:
        return int(args.func(args) or 0)
    except Exception as exc:
        err_msg = str(exc).lower()
        auth_keywords = ["credentials", "accessdenied", "unauthorized", "missing/incomplete configuration", "forbidden", "access denied", "auth"]
        if any(keyword in err_msg for keyword in auth_keywords):
            if prompt_for_credentials(work_dir):
                print("Retrying command with new credentials...")
                try:
                    return int(args.func(args) or 0)
                except Exception as retry_exc:
                    print(f"weathergraph: error on retry: {retry_exc}", file=sys.stderr)
                    return 1
        
        print(f"weathergraph: error: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())