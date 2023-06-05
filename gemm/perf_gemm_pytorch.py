import torch
from torch.utils import benchmark

typ = torch.half  #数据精度
m = 4096
n = 4096
k = 4096
# a = torch.randn(m, k).type(typ).cuda()
# b = torch.randn(k, n).type(typ).cuda()

# t = benchmark.Timer(
#       stmt='a @ b',
#       globals={'a': a, 'b': b})
# x = t.timeit(5)
# print(x)
# print(2*n**3 / x.median /1e12)


a = torch.randn(n, n).type(typ).cuda()
b = torch.randn(n, n).type(typ).cuda()
c = torch.randn(n, n).type(typ).cuda()
for i in range(3):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ]) as p:
        c = torch.addmm(c, a, b, beta=0.5, alpha=0.5)
    p.export_chrome_trace('test.json')
    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
