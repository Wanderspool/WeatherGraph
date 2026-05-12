variable "cloud" {
  description = "Target cloud provider: \"gcp\" or \"aws\""
  type        = string
  default     = "gcp"
  validation {
    condition     = contains(["gcp", "aws"], var.cloud)
    error_message = "cloud must be \"gcp\" or \"aws\"."
  }
}

# ── GCP ───────────────────────────────────────────────────────────────────────
variable "gcp_project" {
  description = "GCP project ID"
  type        = string
  default     = ""
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gcp_machine_type" {
  description = "GCP machine type (e2-standard-2 = 2 vCPU / 8 GB RAM)"
  type        = string
  default     = "e2-standard-2"
}

# ── AWS ───────────────────────────────────────────────────────────────────────
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "aws_instance_type" {
  description = "AWS EC2 instance type"
  type        = string
  default     = "t3.large"
}

# ── Model source ──────────────────────────────────────────────────────────────
variable "model_source" {
  description = "\"url\" to download a pre-built ONNX, or \"script\" to convert"
  type        = string
  default     = "url"
}

variable "model_url" {
  description = "HTTPS URL of the .onnx model (used when model_source = \"url\")"
  type        = string
  default     = "https://huggingface.co/Wanderspool/Keisler_2022/resolve/main/keisler_2022.onnx"
}

variable "model_script" {
  description = "Repo-relative conversion script path (used when model_source = \"script\")"
  type        = string
  default     = ""
}

variable "model_script_args" {
  description = "Extra CLI args for the conversion script"
  type        = string
  default     = ""
}

variable "model_script_pip" {
  description = "Space-separated pip packages required by the conversion script"
  type        = string
  default     = ""
}

# ── Forecast ──────────────────────────────────────────────────────────────────
variable "input_data" {
  description = "Path (or gs:// / s3:// URI) to the ERA5 initial state NetCDF file"
  type        = string
  default     = ""
}

variable "output_dir" {
  description = "Local directory on the instance where results are written"
  type        = string
  default     = "/tmp/weathergraph_out"
}

variable "output_bucket" {
  description = "gs:// or s3:// URI to upload results after the run (leave empty to skip)"
  type        = string
  default     = ""
}

variable "steps" {
  description = "Number of 6-hour forecast steps (40 = 10 days)"
  type        = number
  default     = 40
}

variable "output_fmt" {
  description = "Output format: \"netcdf4\" | \"zarr\" | \"npz\""
  type        = string
  default     = "netcdf4"
}

variable "t0" {
  description = "ISO-8601 start datetime for the output time axis (e.g. \"2024-01-01T00\")"
  type        = string
  default     = ""
}

variable "intra_op_threads" {
  description = "ONNX Runtime intra-op thread count"
  type        = number
  default     = 1
}

variable "execution_provider" {
  description = "Preferred ONNX Runtime execution provider: \"cpu\" | \"cuda\" | \"tensorrt\" | \"rocm\" | \"openvino\""
  type        = string
  default     = "cpu"
  validation {
    condition     = contains(["cpu", "cuda", "tensorrt", "rocm", "openvino"], var.execution_provider)
    error_message = "execution_provider must be one of \"cpu\", \"cuda\", \"tensorrt\", \"rocm\", or \"openvino\"."
  }
}

variable "execution_device_id" {
  description = "Accelerator device ordinal when execution_provider is not \"cpu\""
  type        = number
  default     = 0
}

variable "execution_memory_limit" {
  description = "Provider memory cap in bytes (0 keeps ONNX Runtime default)"
  type        = number
  default     = 0
}

variable "execution_provider_options" {
  description = "JSON object string forwarded to the selected execution provider"
  type        = string
  default     = ""
}

variable "disable_cpu_ep_fallback" {
  description = "Fail session creation if accelerator placement would fall back to CPU"
  type        = bool
  default     = false
}

variable "disable_cpu_mem_arena" {
  description = "Disable the ONNX Runtime CPU arena allocator to reduce reserved RSS"
  type        = bool
  default     = false
}

variable "disable_mem_pattern" {
  description = "Disable ONNX Runtime memory-pattern reuse"
  type        = bool
  default     = false
}

variable "spatial_tiling" {
  description = "Enable exact graph-aware tiled inference (requires tile_bundle_path)"
  type        = bool
  default     = false
}

variable "tile_bundle_path" {
  description = "Path to a tile-bundle manifest or directory with exact graph partition metadata"
  type        = string
  default     = ""
}

variable "reference_grid_shape" {
  description = "Optional export/reference-grid shape as LATxLON"
  type        = string
  default     = ""
}

variable "reference_grid_resolution_degrees" {
  description = "Optional regular global reference-grid resolution in degrees"
  type        = string
  default     = ""
}

variable "tile_state_backend" {
  description = "Backend used for global tiled-state buffers: \"ram\" | \"memmap\""
  type        = string
  default     = "ram"
  validation {
    condition     = contains(["ram", "memmap"], var.tile_state_backend)
    error_message = "tile_state_backend must be \"ram\" or \"memmap\"."
  }
}

variable "tile_state_dir" {
  description = "Optional directory for memmap-backed tiled-state buffers"
  type        = string
  default     = ""
}

# ── Data source ────────────────────────────────────────────────────────────────
variable "data_source" {
  description = <<-EOT
    Data-source adapter name.  Supported values:
      era5_netcdf  (default)  — local ERA5 NetCDF file (see input_data)
      ecmwf_open              — ECMWF free real-time forecast  (pip: ecmwf-opendata cfgrib)
      cds_era5                — Copernicus CDS ERA5 reanalysis (pip: cdsapi, requires ~/.cdsapirc)
      gfs                     — NOAA GFS via AWS S3 Open Data  (pip: herbie-data)
      open_meteo              — Open-Meteo free API            (single-point, no key)
      custom                  — custom file / variable mapping (see data_source_params)
  EOT
  type    = string
  default = "era5_netcdf"
}

variable "data_source_params" {
  description = <<-EOT
    JSON string of keyword arguments forwarded to the chosen data-source adapter.
    Leave empty for era5_netcdf (the input_data path is used automatically).

    Examples:
      ecmwf_open  : {"date":"2024-01-01","step":0}
      cds_era5    : {"date":"2024-01-01","time":"00:00"}
      gfs         : {"date":"2024-01-01 00:00","fxx":0}
      open_meteo  : {"latitude":51.5,"longitude":-0.1}
      custom      : {"source":"my.nc","variable_map":{"z":"geopotential","t":"temperature"}}
  EOT
  type    = string
  default = ""
}
