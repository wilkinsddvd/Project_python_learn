import numpy as np
import matplotlib.pyplot as plt

# 模型参数
growth_rate = 0.1  # 基础生长率
food_availability = np.linspace(0.1, 1, 100)  # 资源可用性范围
male_ratio = np.zeros_like(food_availability)  # 性别比例初始化

# 性别比例模型函数
def sex_ratio(food):
    # 假设性别比例与食物供应量成非线性关系
    return 0.78 - 0.22 * np.tanh(10 * (food - 0.5))

# 计算不同资源可用性下的性别比例
for i, food in enumerate(food_availability):
    male_ratio[i] = sex_ratio(food)

# 绘图展示
plt.plot(food_availability, male_ratio)
plt.xlabel('Food Availability')
plt.ylabel('Male Ratio')
plt.title('Sex Ratio Variation with Food Availability')
plt.show()