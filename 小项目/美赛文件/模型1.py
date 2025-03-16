import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 定义性别比率和资源动态的模型方程
def model(y, t, params):
    R, A = y
    gamma, A_threshold, r_A, K_A = params
    dRdt = gamma * (A - A_threshold)
    dAdt = r_A * A * (1 - A / K_A) - R * A
    return [dRdt, dAdt]

# 初始条件
R0 = 0.56  # 初始性别比率
A0 = 1000  # 初始资源量
# ......
# 省略部分内容
# ......
y0 = [R0, A0]

# 时间点
t = np.linspace(0, 100, 1000)

# 解微分方程
solution = odeint(model, y0, t, args=(params,))

# 绘制结果
plt.figure(figsize=(10, 5))
plt.plot(t, solution[:, 0], label='Sex Ratio $R_t$')
plt.plot(t, solution[:, 1], label='Resource $A_t$')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Dynamics of Sex Ratio and Resource Availability')
plt.legend()
plt.show()