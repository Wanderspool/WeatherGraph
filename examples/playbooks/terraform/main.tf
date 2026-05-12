# ==============================================================================
# WeatherGraph — Terraform Module
#
# Provisions a single compute instance (GCP or AWS), runs the WeatherGraph
# forecast pipeline via cloud-init, and exposes the output storage path.
#
# Usage
# ─────
#   # GCP
#   terraform init
#   terraform apply -var="cloud=gcp" -var="gcp_project=my-project"
#
#   # AWS
#   terraform apply -var="cloud=aws" -var="aws_region=eu-west-1"
#
#   # Override model / forecast parameters
#   terraform apply \
#     -var="cloud=gcp" \
#     -var="gcp_project=my-project" \
#     -var="model_url=https://my-bucket.s3.amazonaws.com/models/my_gnn.onnx" \
#     -var="steps=16" \
#     -var="output_fmt=zarr" \
#     -var="output_bucket=gs://my-bucket/forecasts"
# ==============================================================================

terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ── Providers (only the selected cloud is actually used) ─────────────────────
provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

provider "aws" {
  region = var.aws_region
}

# ── Locals ────────────────────────────────────────────────────────────────────
locals {
  # Cloud-init script injected into both GCP and AWS instances.
  # Installs deps, builds WeatherGraph, downloads model, runs forecast.
  startup_script = templatefile("${path.module}/cloud-init.sh.tpl", {
    model_source      = var.model_source
    model_url         = var.model_url
    model_script      = var.model_script
    model_script_args = var.model_script_args
    model_script_pip  = var.model_script_pip
    input_data        = var.input_data
    output_dir        = var.output_dir
    output_bucket     = var.output_bucket
    steps             = var.steps
    output_fmt        = var.output_fmt
    t0                = var.t0
    intra_op_threads  = var.intra_op_threads
    execution_provider = var.execution_provider
    execution_device_id = var.execution_device_id
    execution_memory_limit = var.execution_memory_limit
    execution_provider_options = var.execution_provider_options
    disable_cpu_ep_fallback = var.disable_cpu_ep_fallback
    disable_cpu_mem_arena = var.disable_cpu_mem_arena
    disable_mem_pattern = var.disable_mem_pattern
    spatial_tiling    = var.spatial_tiling
    tile_bundle_path  = var.tile_bundle_path
    reference_grid_shape = var.reference_grid_shape
    reference_grid_resolution_degrees = var.reference_grid_resolution_degrees
    tile_state_backend = var.tile_state_backend
    tile_state_dir = var.tile_state_dir
    data_source = var.data_source
    data_source_params = var.data_source_params
  })
}

# ── GCP: Compute Engine instance ─────────────────────────────────────────────
resource "google_compute_instance" "weathergraph" {
  count        = var.cloud == "gcp" ? 1 : 0
  name         = "weathergraph-forecast"
  machine_type = var.gcp_machine_type
  zone         = "${var.gcp_region}-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = 30
    }
  }

  network_interface {
    network       = "default"
    access_config {}   # ephemeral public IP — remove for VPC-only
  }

  metadata = {
    startup-script = local.startup_script
  }

  service_account {
    scopes = ["cloud-platform"]
  }

  # Automatically delete after job completes (via shutdown in cloud-init.sh.tpl)
  tags = ["weathergraph"]
}

# ── AWS: EC2 instance ─────────────────────────────────────────────────────────
data "aws_ami" "debian" {
  count       = var.cloud == "aws" ? 1 : 0
  most_recent = true
  owners      = ["136693071363"]   # Debian official
  filter {
    name   = "name"
    values = ["debian-12-amd64-*"]
  }
}

resource "aws_instance" "weathergraph" {
  count         = var.cloud == "aws" ? 1 : 0
  ami           = data.aws_ami.debian[0].id
  instance_type = var.aws_instance_type
  user_data     = local.startup_script

  root_block_device {
    volume_size = 30
  }

  tags = {
    Name    = "weathergraph-forecast"
    Project = "weathergraph"
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────────
output "gcp_instance_ip" {
  value       = var.cloud == "gcp" ? google_compute_instance.weathergraph[0].network_interface[0].access_config[0].nat_ip : null
  description = "Public IP of the GCP forecast instance"
}

output "aws_instance_ip" {
  value       = var.cloud == "aws" ? aws_instance.weathergraph[0].public_ip : null
  description = "Public IP of the AWS forecast instance"
}

output "output_location" {
  value       = var.output_bucket != "" ? var.output_bucket : var.output_dir
  description = "Where forecast results are written"
}
