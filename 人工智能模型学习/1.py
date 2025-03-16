import torch
import torch.nn as nn
import torch.optim as optim


# 定义神经网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 32)  # 输入层到隐藏层
        self.fc2 = nn.Linear(32, 1)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 在隐藏层使用ReLU激活函数
        x = self.fc2(x)
        return x


# 初始化神经网络
net = Net()
print(net)


# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.SGD(net.parameters(), lr=0.01)  # 随机梯度下降优化器

# 假设我们有一些输入数据x和对应的真实值y（在实际应用中，x和y通常来自于训练数据集）
x = torch.randn(10, 16)
y = torch.randn(10, 1)

# 模型训练
for epoch in range(100):  # 训练100个epoch
    optimizer.zero_grad()  # 梯度清零
    outputs = net(x)  # 前向传播
    loss = criterion(outputs, y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数