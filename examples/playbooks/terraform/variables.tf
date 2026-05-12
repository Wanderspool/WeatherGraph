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
