import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义生态系统模型方程
def ecosystem(y, t, params):
    N_lamprey, N_parasite = y
    r_lamprey, K_lamprey, sigma_lamprey, r_parasite, K_parasite = params
# ......
# 省略部分内容
# ......
    return [dN_lamprey_dt, dN_parasite_dt]

# 参数和初始条件
params = [0.05, 1000, 0.02, 0.1, 0.5]  # 示例参数
y0 = [200, 100]  # 初始种群大小
t = np.linspace(0, 100, 1000)  # 时间范围

# 解微分方程
solution = odeint(ecosystem, y0, t, args=(params,))

# 绘制结果
plt.plot(t, solution[:, 0], label='Lampreys')
plt.plot(t, solution[:, 1], label='Parasites')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.title('Dynamics of Lamprey and Parasite Populations')
plt.legend()
plt.show()