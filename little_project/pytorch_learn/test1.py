# 1. 导入必要的库
import torch
import numpy as np

# 2. 随机生成矩阵A和向量b
# 假设我们想要一个3x3的方程组
A = np.random.rand(3, 3)
b = np.random.rand(3, 1)

# 3. 使用PyTorch创建对应的张量
A_torch = torch.tensor(A, dtype=torch.float32)
b_torch = torch.tensor(b, dtype=torch.float32)

# 4. 使用PyTorch的线性代数功能求解线性方程组
x_torch = torch.linalg.solve(A_torch, b_torch)

# 打印结果
print("The solution vector x is:")
print(x_torch)