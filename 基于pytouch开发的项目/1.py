#coding = utf-8
import torch
from time import perf_counter

X = torch.rand(1000, 10000)
Y = torch.rand(10000, 10000)

start = perf_counter()
X.mm(Y)
finish = perf_counter()
time = finish-start
print("CPU计算时间: %s" %time)

x = X.cuda()
y = Y.cuda()
start = perf_counter()
X.mm(Y)
finish = perf_counter()
time_cuda = finish - start
print("GPU加速计算的时间：%s" % time_cuda)
print("CPU计算时间是GPU加速计算时间的%s倍" % str(time/time_cuda))
