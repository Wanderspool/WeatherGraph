# WeatherGraph Examples and Pipelines Guide

## Purpose

This document explains the examples in `examples/` in plain language for readers who have not worked with infrastructure or automation pipelines before. The goal is not just to show commands, but to explain what each example is for, what problem it solves, and how to choose the right one.

## First: What "Pipeline" Means in This Repository

In WeatherGraph, a pipeline is an automated wrapper around the same core runtime.

No matter which example you choose, most of them perform some variation of the same steps:

1. prepare a machine or environment
2. build the C++ backend
3. obtain an ONNX model
4. obtain an initial weather state
5. run inference or autoregressive forecast
6. export the results
7. optionally upload or store the outputs elsewhere

The examples differ in who does that work and where it runs.

- Some examples are local and manual.
- Some are CI-oriented.
- Some are meant for remote servers.
- Some create cloud infrastructure for you.
- Some submit jobs into managed batch systems.

## The Shared Concepts Used by Almost All Examples

Before looking at individual files, it helps to understand the shared parameters.

### Model source

Most automation examples support one of two model acquisition modes.

- `url`: download a ready-made `.onnx` file.
- `script`: run a conversion script that produces an `.onnx` file.

Use `url` when you already have a compatible ONNX artifact.

Use `script` when your actual model exists in another framework and must be converted during the pipeline.

### Data source

Most examples use one of the adapter names from `weathergraph.data_sources`.

Common choices:

- `era5_netcdf`: local or mounted NetCDF file.
- `ecmwf_open`: download current ECMWF public forecast data.
- `cds_era5`: download ERA5 reanalysis through CDS.
- `gfs`: use NOAA GFS.
- `open_meteo`: single-point remote forecast.
- `custom`: custom file plus explicit variable mapping.

### Output format

Common output formats are:

- `netcdf4`: structured scientific files, streamed step-by-step.
- `zarr`: chunk-friendly directory output, also streamed.
- `npz`: compressed raw arrays, but not append-friendly.

### Runtime knobs

Several examples expose the same runtime options because they all call the same `WeatherGraphModel` constructor.

- `intra_op_threads`
- `execution_provider`
- `execution_device_id`
- `execution_memory_limit`
- `execution_provider_options`
- `disable_cpu_ep_fallback`
- `disable_cpu_mem_arena`
- `disable_mem_pattern`
- `spatial_tiling`
- `tile_bundle_path`
- `reference_grid_shape`
- `reference_grid_resolution_degrees`
- `tile_state_backend`
- `tile_state_dir`

### Bundle preparation

If you already have per-tile ONNX artifacts, the supported way to package a usable bundle is:

```bash
weathergraph build-tile-bundle \
	--output-dir tile_bundle \
	--senders-path data/graph_data/senders_receivers_encoder/senders.npy \
	--receivers-path data/graph_data/senders_receivers_encoder/receivers.npy \
	--tile-model-dir tile_models \
	--reference-grid-shape 1801x3600 \
	--tile-grid-shape 150x150
```

This command generates `manifest.json` plus the `*_input.npy` and `*_output.npy` index arrays expected by `WeatherGraphModel`.

### Important interpretation of the low-memory and tiling flags

- `execution_provider` defaults to `cpu`; accelerator values such as `cuda`, `tensorrt`, `rocm`, and `openvino` only make sense on hosts that have the matching ONNX Runtime provider bundle and vendor runtime.
- `disable_cpu_ep_fallback` only makes sense when `execution_provider` is not `cpu`.
- `disable_cpu_mem_arena` and `disable_mem_pattern` are optional and off by default.
- `spatial_tiling` is also optional and requires a valid tile bundle.
- `tile_state_backend="memmap"` is useful when tiled high-resolution runs are limited by RAM rather than disk bandwidth.
- These flags are runtime controls, not separate product modes.

## Which Example Should You Start With?

If you are new to the project, use this order.

1. `weathergraph` CLI if you want the supported operational entrypoint with inspect and forecast modes.
2. `examples/simulate_meteorologist.py` if you want to study a richer scenario runner around the same runtime.
3. `examples/playbooks/jupiter_notebook.ipynb` if you want an interactive walk-through.
4. `examples/playbooks/github_actions.yml` if you want automated validation in CI.
5. `examples/playbooks/ansible/site.yml` if you already manage your own Linux hosts.
6. `examples/playbooks/terraform/` if you want infrastructure provisioning plus execution.
7. `examples/playbooks/gcp/batch-job.yaml` or `examples/playbooks/aws/cloudformation.yaml` if you want managed cloud batch execution.

## 1. Supported CLI: `weathergraph`

### What this example is for

This is the supported front door for researchers who want to use WeatherGraph without writing Python glue code.

### What it provides

- `weathergraph list-sources` to discover adapters
- `weathergraph inspect` to validate runtime/provider/grid settings and print memory sizing
- `weathergraph forecast` to run one-step inference, iterative rollout, or export to disk

### Typical commands

```bash
weathergraph inspect \
	--model-path models/weather_gnn.onnx \
	--weights-dir data \
	--execution-provider cuda \
	--execution-device-id 0

weathergraph forecast \
	--model-path models/weather_gnn.onnx \
	--weights-dir data \
	--data-source era5_netcdf \
	--input-path data/era5_archives/init_state.nc \
	--steps 40 \
	--output-format zarr \
	--output-path forecast_out
```

From a source checkout without installation, use `python -m weathergraph.cli`.

## 2. Local Script: `examples/simulate_meteorologist.py`

### What this example is for

This is the most direct operational example. It is a local or self-managed runner that:

- verifies the model and data environment
- selects a data source
- constructs `WeatherGraphModel`
- runs a set of predefined simulation scenarios

### When to use it

Use this example when you want to:

- confirm that your local installation works
- test a real model against real meteorological input
- compare different data sources without writing new code
- experiment with low-memory, GPU-provider, or tiling flags in the simplest possible setup

### What you need to provide

At minimum:

- an ONNX model path
- a weights directory containing `means.npy` and `stds.npy`
- a valid data source configuration

Environment variables used by the script include:

- `WEATHERGRAPH_ONNX_MODEL`
- `WEATHERGRAPH_WEIGHTS_DIR`
- `ERA5_DATA_DIR`
- `WEATHERGRAPH_INTRA_OP_THREADS`
- `WEATHERGRAPH_EXECUTION_PROVIDER`
- `WEATHERGRAPH_EXECUTION_DEVICE_ID`
- `WEATHERGRAPH_EXECUTION_MEMORY_LIMIT`
- `WEATHERGRAPH_EXECUTION_PROVIDER_OPTIONS`
- `WEATHERGRAPH_DISABLE_CPU_EP_FALLBACK`
- `WEATHERGRAPH_DISABLE_CPU_MEM_ARENA`
- `WEATHERGRAPH_DISABLE_MEM_PATTERN`
- `WEATHERGRAPH_SPATIAL_TILING`
- `WEATHERGRAPH_TILE_BUNDLE_PATH`
- `WEATHERGRAPH_REFERENCE_GRID_SHAPE`
- `WEATHERGRAPH_REFERENCE_GRID_RESOLUTION_DEGREES`
- `WEATHERGRAPH_TILE_STATE_BACKEND`
- `WEATHERGRAPH_TILE_STATE_DIR`
- `DATA_SOURCE`
- source-specific variables such as `CDS_DATE`, `GFS_DATE`, `OPEN_METEO_LAT`, `OPEN_METEO_LON`, or `DATA_SOURCE_SCHEMA`

### Simple mental model

Think of this file as a reusable command-line harness around the Python API. It is the easiest place to learn how runtime flags and data-source adapters behave.

### Typical starting command

If you already have local ERA5 input files:

```bash
ERA5_DATA_DIR=data/era5_archives python examples/simulate_meteorologist.py
```

If you want to test the low-memory path:

```bash
WEATHERGRAPH_DISABLE_CPU_MEM_ARENA=true \
WEATHERGRAPH_DISABLE_MEM_PATTERN=true \
python examples/simulate_meteorologist.py
```

If you want to test the CUDA execution provider:

```bash
WEATHERGRAPH_EXECUTION_PROVIDER=cuda \
WEATHERGRAPH_EXECUTION_DEVICE_ID=0 \
WEATHERGRAPH_DISABLE_CPU_EP_FALLBACK=true \
python examples/simulate_meteorologist.py
```

If you want to test exact tiling prepared for a higher-resolution export path:

```bash
WEATHERGRAPH_SPATIAL_TILING=true \
WEATHERGRAPH_TILE_BUNDLE_PATH=tile_bundle/manifest.json \
WEATHERGRAPH_TILE_STATE_BACKEND=memmap \
WEATHERGRAPH_REFERENCE_GRID_RESOLUTION_DEGREES=0.1 \
python examples/simulate_meteorologist.py
```

### Common mistakes

- requesting `WEATHERGRAPH_EXECUTION_PROVIDER=cuda` without shipping the ONNX Runtime CUDA provider `.so` files or compatible vendor runtime libraries
- setting `WEATHERGRAPH_DISABLE_CPU_EP_FALLBACK=true` while still using the default CPU execution provider
- enabling `WEATHERGRAPH_SPATIAL_TILING` without providing a tile bundle
- asking for a high-resolution reference grid without confirming that the model or tile bundle exposes enough nodes for that export shape
- assuming the script can generate the ONNX model for you on a machine that cannot produce it
- using a data source that requires extra Python packages without installing them first

## 2. Interactive Notebook: `examples/playbooks/jupiter_notebook.ipynb`

### What this example is for

This notebook is the interactive learning path. It is useful when you want to inspect the workflow cell by cell instead of running a full automation script.

### When to use it

Use the notebook when you want to:

- explore the API interactively
- modify parameters step-by-step
- understand the sequence of loading data, creating a model, predicting, and exporting
- demonstrate the workflow to another person

### How to think about it

The notebook is not a separate runtime implementation. It is a guided front-end for the same Python package and the same model constructor used everywhere else.

### Practical usage advice

- open the notebook in VS Code or Jupyter
- run cells from top to bottom the first time
- do not skip the setup cells
- treat it as an exploratory workflow, not a deployment mechanism

### What it is good at

- learning
- experimentation
- manual inspection of outputs

### What it is not good at

- unattended production jobs
- repeatable infrastructure provisioning
- long-lived scheduled automation by itself

## 3. CI Template: `examples/playbooks/github_actions.yml`

### What this example is for

This file is a template that a user copies into their own repository to validate a model automatically with GitHub Actions.

It does not define the reusable engine pipeline itself. Instead, it calls the reusable workflow in `.github/workflows/model-pipeline.yml`.

### When to use it

Use this example when you want CI to answer questions like:

- does my ONNX artifact still load?
- does my conversion script still produce a valid graph?
- does WeatherGraph still run smoke validation against my model?

### Four modes shown in the template

The file demonstrates four common ways to validate a model:

1. use a pre-built ONNX URL
2. convert from PyTorch
3. build the prototype latent-output graph for experiments
4. convert from TensorFlow or Keras

### Important distinction

The prototype graph path is not the normal production path. It is specifically marked as a latent-output experiment and is not compatible with standard autoregressive `WeatherGraphModel` usage unless the prototype path is explicitly allowed.

### How to adopt it

1. copy the template into `.github/workflows/` in your own repository
2. uncomment the job that matches your model source
3. fill in the model URL or conversion script parameters
4. optionally add low-memory runtime flags for smoke validation

### Why this example matters

It teaches that WeatherGraph can be consumed as a reusable validation engine by downstream model repositories, not only as a standalone runtime repository.

## 4. Remote Host Automation: `examples/playbooks/ansible/site.yml`

### What this example is for

This playbook automates the full pipeline on a Linux machine that you already control.

It performs:

- package installation
- repository checkout
- Python virtual environment creation
- ONNX Runtime acquisition
- C++ backend build
- model download or conversion
- forecast execution

### When to use it

Use Ansible when:

- you already have one or more Linux hosts
- you want repeatable setup without manually logging into each machine
- you prefer infrastructure described as configuration rather than ad hoc shell commands

### How to think about it

Ansible is for machine configuration plus job execution. It is a good fit when the machine is long-lived or semi-permanent.

### What you edit

- inventory file with host addresses
- variables such as `model_url`, `input_data`, `steps`, `output_fmt`
- optional `data_source` and `data_source_params`

### Good first use case

You have one forecast server and want to be able to rebuild and rerun the same pipeline reliably.

## 5. Provision-and-Run Infrastructure: `examples/playbooks/terraform/`

### What this example is for

The Terraform example provisions a cloud instance and renders a cloud-init bootstrap script that performs the actual WeatherGraph run.

Relevant files:

- `examples/playbooks/terraform/variables.tf`: user inputs
- `examples/playbooks/terraform/main.tf`: resource definitions
- `examples/playbooks/terraform/cloud-init.sh.tpl`: machine bootstrap script

### When to use it

Use Terraform when:

- you want infrastructure creation recorded as code
- you want a one-shot machine created for the forecast run
- you want the same configuration to work for GCP or AWS with variable changes

### How to think about it

Terraform here is not the forecast runner by itself. It is the infrastructure orchestrator. The real forecast work happens in the rendered cloud-init script after the machine boots.

### What happens internally

1. Terraform creates a VM.
2. The VM runs the rendered cloud-init script.
3. The script clones the repository, builds WeatherGraph, downloads or converts the model, runs the forecast, and optionally uploads the output.

### Good first use case

You want disposable infrastructure that is created for the job and can disappear after the job completes.

## 6. Managed GCP Batch Job: `examples/playbooks/gcp/batch-job.yaml`

### What this example is for

This file submits the workload to Google Cloud Batch. Google manages the job execution environment, and you do not manage a persistent VM directly.

### When to use it

Use this when:

- you prefer a managed batch job model
- you want cloud execution without maintaining a long-lived instance
- your output is naturally written to local temporary storage and then copied to `gs://`

### How to think about it

This is a self-contained job definition. The machine exists only for the duration of the batch task.

### Typical usage pattern

1. enable the required GCP services
2. submit the job with `gcloud batch jobs submit`
3. override environment variables when needed
4. inspect logs in Cloud Logging

### Why it is different from Terraform

- Terraform is about provisioning infrastructure resources.
- GCP Batch is about submitting a managed job payload.

If you do not need persistent infrastructure state, Batch is usually simpler.

## 7. Managed AWS Batch Stack: `examples/playbooks/aws/cloudformation.yaml`

### What this example is for

This template creates the AWS Batch environment needed to run WeatherGraph in AWS.

It defines:

- IAM roles
- compute environment
- job queue
- job definition

### When to use it

Use this when:

- you are already in AWS
- you want managed batch execution
- you want the batch environment itself defined declaratively

### How to think about it

Unlike the GCP Batch file, this example includes the infrastructure stack definition around the batch environment, not just the job payload.

### Typical workflow

1. deploy the CloudFormation stack
2. wait for the batch environment to exist
3. submit jobs with parameter overrides
4. store outputs in `s3://` if desired

### Practical difference from Terraform

- Terraform provisions a generic compute instance and boots a script on it.
- this CloudFormation template provisions an AWS Batch system and then runs WeatherGraph inside that job environment.

## How the Examples Relate to One Another

All examples are wrappers around the same core runtime.

They differ across three axes.

### Axis 1. Where the run happens

- local machine
- your own server
- provisioned cloud instance
- managed batch service

### Axis 2. Who owns the environment

- you manually
- configuration management
- infrastructure-as-code tooling
- cloud batch service

### Axis 3. What the goal is

- learning and debugging
- CI validation
- repeatable server operations
- disposable production execution

## A Simple Decision Guide

Choose the example based on your actual need.

- If you want to learn the API, start with `simulate_meteorologist.py`.
- If you want interactive exploration, use the notebook.
- If you want model validation on every change, use the GitHub Actions template.
- If you already manage Linux machines, use Ansible.
- If you want one-shot cloud infrastructure, use Terraform.
- If you want managed cloud jobs, use GCP Batch or AWS Batch.

## Common Beginner Mistakes Across Pipelines

- treating every example as if it were a different inference engine
- forgetting that most pipelines still need a valid ONNX artifact and valid input data
- enabling exact tiling without a tile bundle
- assuming `forecast()` and `forecast_export()` have the same memory behavior
- using remote data-source adapters without their extra dependencies or credentials
- expecting the prototype exporter path to behave like the production autoregressive runtime

## Recommended Learning Path for a New Team Member

1. Read `Docs/feature-guide.md` to understand what the runtime can do.
2. Read `Docs/project-architecture.md` to understand how the code is organized.
3. Run `examples/simulate_meteorologist.py` locally with the default ERA5 path.
4. Open the notebook and step through the same ideas interactively.
5. Only after that, choose one automation surface for CI or deployment.

That order keeps the core runtime understandable before infrastructure complexity is introduced.