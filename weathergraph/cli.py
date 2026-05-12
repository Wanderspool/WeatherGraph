from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import REGISTRY, WeatherGraphModel, list_sources, load_source
from .tile_bundle import build_tile_bundle


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:
        print(f"weathergraph: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())