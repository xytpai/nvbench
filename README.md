# nvbench

This repo mainly focuses on evaluating the various capabilities of graphics cards and providing examples of kernel optimizations.

#### 1. global_memory_bandwidth

```bash
bash build.sh standard/global_memory_bandwidth.cu ; ./a.out
# gen ptx file
# bash build.sh standard/global_memory_bandwidth.cu -ptx ; cat ./a.out
```

| test/card(GBPS) | RTX4090  | A100-40GB |
| :-------------- | --------: | -------: |
| float 1 |  920.207  | 1336.620 |
| float 2 | 904.335 | 1379.710 |
| float 4 | 921.850 | 1378.800 |
| float 8 | 914.626 | 1315.650 |
| float 16 | 921.066 | 1117.290 |

#### 2. single_precision_compute

```bash
bash build.sh standard/single_precision_compute.cu ; ./a.out
```

| Card | TFLOPS |
| :-------------- | --------: |
| RTX4090 |  85.7015  |
| A100-40GB | 19.4042 |

