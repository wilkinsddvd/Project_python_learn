import numpy as np
import matplotlib.pyplot as plt

# 参数和初始条件
r = 0.1  # 基础生长率
alpha = 0.05  # 繁殖成功率系数
beta = 100  # 资源利用效率系数
A = 1000  # 资源量
R_values = np.linspace(0.1, 0.9, 9)  # 性别比率变化范围



# 计算繁殖成功率和资源利用效率
S_values = [alpha * R * (1-R) * A for R in R_values]
# ......
# ......

# 绘制繁殖成功率和资源利用效率随性别比率的变化
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(R_values, S_values, '-o', color='blue')
plt.title('Reproduction Success Rate vs. Sex Ratio')
plt.xlabel('Sex Ratio (R)')
plt.ylabel('Reproduction Success Rate (S)')

plt.subplot(1, 2, 2)
plt.plot(R_values, E_values, '-o', color='green')
plt.title('Resource Utilization Efficiency vs. Sex Ratio')
plt.xlabel('Sex Ratio (R)')
plt.ylabel('Resource Utilization Efficiency (E)')

plt.tight_layout()
plt.show()