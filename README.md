# nvbench

This repo mainly focuses on evaluating the various capabilities of graphics cards and providing examples of kernel optimizations.

#### 1. global_memory_bandwidth

```bash
nvcc standard/global_memory_bandwidth.cu ; ./a.out
```

| test/card(GBPS) | RTX4090  | A100-40GB |
| :-------------- | --------: | -------: |
| float 1 |  920.207  |   |
| float 2 | 904.335 | |
| float 4 | 921.850 | |
| float 8 | 914.626 | |
| float 16 | 921.066 | |

#### 2. single_precision_compute

```bash
nvcc standard/single_precision_compute.cu ; ./a.out
```

| Card | TFLOPS |
| :-------------- | --------: |
| RTX4090 |  85.7015  |

