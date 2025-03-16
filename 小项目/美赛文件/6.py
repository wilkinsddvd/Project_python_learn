# %matplotlib inline 注释掉此行
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt # 增加此行

x = torch.arange(-8.0,8.0,0.1,requires_grad=True)
y = torch.relu(x)
# print(y)
print('zc4')
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5)) # 需要绘制的图像
plt.show() # 新增此行代码
