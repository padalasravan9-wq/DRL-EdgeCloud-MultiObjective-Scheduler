# Datasets

This study uses publicly available workload traces.

## 1. Google Cluster Trace

Source:
https://github.com/google/cluster-data

Used for:
- Realistic task arrival patterns
- Workload validation

## 2. Azure Functions Trace

Source:
https://github.com/Azure/AzurePublicDataset

Used for:
- Serverless workload characteristics
- Edge–cloud workload diversity

## Preprocessing

The repository includes configuration files and scripts required to reproduce the workload generation used in the paper. Due to the large size of the original traces, users are requested to download the datasets from the official sources above.

## Reproducibility

All random seeds used in experiments are provided in:

