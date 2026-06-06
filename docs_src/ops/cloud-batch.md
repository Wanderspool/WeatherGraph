# Managed Batch Jobs

For massive, distributed operational runs—such as daily seasonal forecasts or historic hindcast validation suites—running predictions on persistent VMs is cost-inefficient. Instead, you can run containerized forecasts asynchronously using **Managed Batch Jobs**.

This guide covers configuring and executing WeatherGraph forecasts on Google Cloud Batch and AWS Batch.

---

## 1. Google Cloud Batch (GCP)

Google Cloud Batch manages the lifecycle of VM instances, allocating resources (such as NVIDIA GPUs) only for the duration of the container's execution.

### Batch Job JSON Specification (`gcp_batch_job.json`)
Create a file named `gcp_batch_job.json` outlining the task:

```json
{
  "taskGroups": [
    {
      "taskSpec": {
        "runnables": [
          {
            "container": {
              "imageUri": "gcr.io/my-gcp-project/weathergraph:latest",
              "commands": [
                "weathergraph", "forecast",
                "--model-path", "/models/weather_gnn.onnx",
                "--weights-dir", "/data",
                "--data-source", "gfs",
                "--source-arg", "date=2026-06-06 00:00",
                "--steps", "56",
                "--output-format", "zarr",
                "--output-path", "gs://my-forecast-bucket/runs/2026-06-06"
              ],
              "volumes": [
                "/var/tmp:/tmp"
              ]
            }
          }
        ],
        "computeResource": {
          "cpuMilli": 4000,
          "memoryMib": 16384
        },
        "maxRunDuration": "1800s"
      },
      "taskCount": 1
    }
  ],
  "allocationPolicy": {
    "instances": [
      {
        "installGpuDrivers": true,
        "policy": {
          "machineType": "g2-standard-4",
          "provisioningModel": "SPOT"
        }
      }
    ]
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
```

### Submitting the Job
Deploy the batch job via the Google Cloud CLI:

```bash
gcloud batch jobs submit weathergraph-run-01 \
  --config gcp_batch_job.json \
  --location us-central1
```

---

## 2. AWS Batch

On AWS, you configure a containerized job definition running on ECS Fargate or EC2 instances backing GPU instances.

### AWS Batch Job Definition (CloudFormation Template)
```yaml
AWSTemplateFormatVersion: '2012-10-17'
Description: AWS Batch Job Definition for WeatherGraph Forecasts

Resources:
  WeatherGraphJobQueue:
    Type: AWS::Batch::JobQueue
    Properties:
      JobQueueName: weathergraph-queue
      Priority: 1
      State: ENABLED
      ComputeEnvironmentOrder:
        - Order: 1
          ComputeEnvironment: !Ref ComputeEnvironmentArn

  WeatherGraphJobDefinition:
    Type: AWS::Batch::JobDefinition
    Properties:
      JobDefinitionName: weathergraph-forecast-job
      Type: container
      PlatformCapabilities:
        - EC2
      ContainerProperties:
        Image: !Sub "${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/weathergraph:latest"
        Vcpus: 4
        Memory: 16384
        ResourceRequirements:
          - Type: GPU
            Value: "1"
        Command:
          - "weathergraph"
          - "forecast"
          - "--model-path"
          - "/models/weather_gnn.onnx"
          - "--weights-dir"
          - "/data"
          - "--data-source"
          - "gfs"
          - "--source-arg"
          - "date=2026-06-06 00:00"
          - "--steps"
          - "56"
          - "--output-format"
          - "zarr"
          - "--output-path"
          - "s3://my-forecast-bucket/runs/2026-06-06"
        JobRoleArn: !Ref ECSJobRoleArn
```

---

## 3. Streaming Outputs to Cloud Buckets

When running batch tasks, the local disk space is ephemeral. WeatherGraph's `gfs` or `cds_era5` adapters fetch data directly from cloud mirrors, and the output format `"zarr"` streams the predicted outputs step-by-step to Google Cloud Storage (`gs://`) or Amazon S3 (`s3://`) using Dask's lazy writing and `s3fs`/`gcsfs` file mapping under the hood.

This means you do not need to attach persistent block storage (PD or EBS) to your batch containers, keeping operations simple and cost-efficient.