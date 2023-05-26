import torch
from torch.utils import benchmark

typ = torch.float  #数据精度
n = 2048
a = torch.randn(n, n).type(typ).cuda()
b = torch.randn(n, n).type(typ).cuda()
t = benchmark.Timer(
      stmt='a @ b',
      globals={'a': a, 'b': b})
x = t.timeit(5)
print(x)
print(2*n**3 / x.median /1e12)


a = torch.randn(n, n).cuda()
b = torch.randn(n, n).cuda()
c = torch.randn(n, n).cuda()
for i in range(2):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ]) as p:
        c = torch.addmm(c, a, b, beta=0.5, alpha=0.5)
    # p.export_chrome_trace('test.json')
    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
