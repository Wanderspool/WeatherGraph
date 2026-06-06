import os

files = {
    "intro/index.md": "# Welcome\n\nWelcome to the WeatherGraph documentation.\n\n<!-- TODO: Content in next iteration -->",
    "intro/install.md": "# Installation & Dependencies\n\nLearn how to install WeatherGraph and its dependencies.\n\n<!-- TODO: Content in next iteration -->",
    "intro/quickstart-cli.md": "# Quickstart (CLI)\n\nA rapid operational start using the `weathergraph` command line interface.\n\n<!-- TODO: Content in next iteration -->",
    "intro/quickstart-python.md": "# Quickstart (Python API)\n\nA rapid start for researchers integrating the engine directly in code.\n\n<!-- TODO: Content in next iteration -->",
    "tutorials/first-forecast.md": "# Your First Forecast\n\nA step-by-step tutorial on running an end-to-end global forecast.\n\n<!-- TODO: Content in next iteration -->",
    "tutorials/data-sources.md": "# Working with Different Data Sources\n\nLearn to load ERA5, GFS, and ECMWF data.\n\n<!-- TODO: Content in next iteration -->",
    "tutorials/visualization.md": "# Visualizing Predictions\n\nTutorial on plotting and animating inference outputs.\n\n<!-- TODO: Content in next iteration -->",
    "tutorials/exporting.md": "# Exporting to NetCDF/Zarr\n\nHow to save long rollout artifacts efficiently.\n\n<!-- TODO: Content in next iteration -->",
    "guide/engine-architecture.md": "# The Engine Architecture\n\nUnderstanding the C++ backend and Python orchestrator boundary.\n\n<!-- TODO: Content in next iteration -->",
    "guide/adapters.md": "# Data Ingestion & Adapters\n\nDeep dive into the adapter pattern for ingestion.\n\n<!-- TODO: Content in next iteration -->",
    "guide/inference-modes.md": "# Inference Modes\n\nExplanation of one-step, autoregressive, and iterative inference.\n\n<!-- TODO: Content in next iteration -->",
    "guide/exporting.md": "# Output Formats & Exporting\n\nGuide to formatting, schemas, and streaming limits.\n\n<!-- TODO: Content in next iteration -->",
    "guide/cli-operations.md": "# CLI Reference & Operations\n\nFull reference of the `weathergraph` command suite.\n\n<!-- TODO: Content in next iteration -->",
    "guide/climate-ecosystem.md": "# Climate Ecosystem Integration\n\nIntegrating with Xarray, MetPy, and xCDAT.\n\n<!-- TODO: Content in next iteration -->",
    "advanced/hardware-tuning.md": "# Hardware Tuning & Memory Limits\n\nConfiguring arenas and pattern tuning for constrained hardware.\n\n<!-- TODO: Content in next iteration -->",
    "advanced/execution-providers.md": "# Execution Providers (GPUs)\n\nDeploying on CUDA, TensorRT, ROCm, and OpenVINO.\n\n<!-- TODO: Content in next iteration -->",
    "advanced/spatial-tiling.md": "# Spatial Tiling & Large Grids\n\nMathematical exact tiling and tile bundle specifications.\n\n<!-- TODO: Content in next iteration -->",
    "advanced/probabilistic-ensembles.md": "# Probabilistic Ensembles\n\nUsing Welford's algorithm and noise injection for ensembles.\n\n<!-- TODO: Content in next iteration -->",
    "advanced/custom-constraints.md": "# Hard Constraints & Custom Graphs\n\nAppending physics constraints via in-place zero-copy execution.\n\n<!-- TODO: Content in next iteration -->",
    "advanced/halo-exchange.md": "# Halo Exchange & Stitching\n\nDealing with overlaps and stitching global domains.\n\n<!-- TODO: Content in next iteration -->",
    "ops/ansible.md": "# Remote Linux Hosts (Ansible)\n\nConfiguration management for bare metal deployments.\n\n<!-- TODO: Content in next iteration -->",
    "ops/terraform.md": "# Cloud Provisioning (Terraform)\n\nProvisioning ephemeral cloud execution instances.\n\n<!-- TODO: Content in next iteration -->",
    "ops/cloud-batch.md": "# Managed Batch Jobs\n\nRunning workloads on GCP and AWS Batch.\n\n<!-- TODO: Content in next iteration -->",
    "ops/ci-cd.md": "# CI/CD Validation\n\nAutomating tests and smoke checks with GitHub Actions/GitLab CI.\n\n<!-- TODO: Content in next iteration -->",
    "dev/building.md": "# Building from Source\n\nCompiling the C++ core and Pybind11 extension.\n\n<!-- TODO: Content in next iteration -->",
    "dev/model-conversion.md": "# Model Conversion & Export\n\nConverting arbitrary PyTorch/JAX weights into the ONNX reference format.\n\n<!-- TODO: Content in next iteration -->",
    "dev/tile-bundles.md": "# Generating Tile Bundles\n\nAuthoring and packaging exact manifest structures.\n\n<!-- TODO: Content in next iteration -->",
    "dev/testing.md": "# Testing & Validation\n\nRunning historical skill validation suites and unit tests.\n\n<!-- TODO: Content in next iteration -->",
    "api/cpp-backend.md": "# C++ Backend API\n\nOverview of the ONNX Runtime `WeatherGraphEngine` bindings.\n\n<!-- TODO: Content in next iteration -->"
}

base_dir = "docs_src"

for filepath, content in files.items():
    full_path = os.path.join(base_dir, filepath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)

print("Created all stub files successfully.")
