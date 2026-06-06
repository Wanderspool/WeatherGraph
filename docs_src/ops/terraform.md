# Cloud Provisioning (Terraform)

Deploying weather models on cloud providers (such as Google Cloud Platform or Amazon Web Services) requires coordinating hardware accelerators (GPUs), virtual machines, and fast scratch disks.

This guide provides a **Terraform** configuration file to provision a GPU-accelerated virtual machine on Google Cloud Platform, install NVIDIA CUDA drivers, and prepare the WeatherGraph environment.

---

## 1. Instance Specification

For global forecasts at regular resolution, we recommend the following instance profiles:
*   **Google Cloud Platform (GCP)**: A `g2-standard-4` instance containing an **NVIDIA L4 GPU** (24 GB VRAM) and 16 GB of system RAM.
*   **Amazon Web Services (AWS)**: A `g4dn.xlarge` instance containing an **NVIDIA T4 GPU** (16 GB VRAM) and 16 GB of system RAM.

---

## 2. Terraform Configuration (GCP)

Save the following configuration as `main.tf`:

```hcl
terraform {
  required_version = ">= 1.3.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.50.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

variable "project_id" {
  type        = string
  description = "GCP Project Identifier"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "zone" {
  type    = string
  default = "us-central1-a"
}

# 1. Allocate a static external IP address
resource "google_compute_address" "static_ip" {
  name = "weathergraph-static-ip"
}

# 2. Provision the GPU Virtual Machine
resource "google_compute_instance" "forecast_vm" {
  name         = "weathergraph-gpu-instance"
  machine_type = "g2-standard-4" # 4 vCPUs, 16 GB RAM, 1x NVIDIA L4 GPU
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 100 # 100 GB boot disk
      type  = "pd-ssd"
    }
  }

  guest_accelerator {
    type  = "nvidia-l4"
    count = 1
  }

  # Necessary metadata to authorize non-interactive CUDA installation
  metadata = {
    install-nvidia-driver = "True"
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.static_ip.address
    }
  }

  # 3. Startup Script to compile WeatherGraph and set up credentials
  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -e
    
    # Update system package registry
    apt-get update
    apt-get install -y build-essential cmake patchelf git python3-pip python3-venv python3-dev
    
    # Install CUDA Toolkit (NVIDIA drivers are installed via compute engine metadata above)
    apt-get install -y nvidia-cuda-toolkit
    
    # Set up project path
    mkdir -p /opt/weathergraph
    git clone https://github.com/Wanderspool/WeatherGraph.git /opt/weathergraph
    
    # Build python package with CUDA compilation flag active
    cd /opt/weathergraph
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    
    # Compile C++ backend for CUDA
    CMAKE_ARGS="-DONNXRUNTIME_CUDA=ON" pip install -e .
    
    # Pre-download default models
    .venv/bin/weathergraph download-model \
      --model-url https://assets.wanderspool.org/models/weathergraph_weights.pkl \
      --output-filename model.pkl
  EOT

  scheduling {
    on_host_maintenance = "TERMINATE" # GPU VM maintenance requirement
  }
}

output "instance_ip" {
  value       = google_compute_address.static_ip.address
  description = "The public IP address of the forecast server."
}
```

---

## 3. Deployment Workflow

1.  **Initialize Terraform**:
    ```bash
    terraform init
    ```
2.  **Generate an execution plan**:
    ```bash
    terraform plan -var="project_id=my-gcp-project-123"
    ```
3.  **Apply configurations**:
    ```bash
    terraform apply -var="project_id=my-gcp-project-123" -auto-approve
    ```

Once provisioning completes (usually 3–5 minutes for the startup script to download libraries and compile the C++ extensions), you can SSH into the instance using the output IP and find a fully compiled and operational GPU WeatherGraph installation in `/opt/weathergraph`.