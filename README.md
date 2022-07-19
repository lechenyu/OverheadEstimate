## Overhead Estimate Using BabelStream 4.0
This repository is to measure the overhead of our new debugging tool which is currently under development.
[BabelStream Repository](https://github.com/UoB-HPC/BabelStream)

## Overhead Estimate Result

| **Kernel** | **GPU Version (V100 GPU)** | **Arbalest (CPU)** | **Ballista (V100 GPU, device memory)** | **Ballista (V100 GPU, pinned memory)** |
|:---:|:---:|:---:|:---:|:---:|
| Copy | 0.67s | 141.37s | 10.03s | 47.46s |
| Mul | 0.67s | 141.46s | 10.48s | 47.45s |
| Add | 0.96s | 209.61s | 15.04s | 71.07s |
| Triad | 0.96s | 209.12s | 15.13s | 71.13s |
| Dot | 0.63s | 135.27s | 10.13s | 47.25s |


