import torch
import torch.nn.functional as F
import math

a = torch.randn(1024, 1024).cuda()
b = torch.randn(1024, 1024).cuda()
c = torch.randn(1024, 1024).cuda()

for i in range(2):
    with torch.profiler.profile(
        activities=[
            # torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]) as p:
        c = torch.addmm(c, a, b, beta=0.5, alpha=0.5)
    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))

# print(4*(1024**2) / (48/1000.0/1000.0) * 4 / 1024/1024/1024) RTX4090: 325 GBps
