import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

def draw_rose():
    theta = np.linspace(0, 2 * np.pi, 1000)
    r = 1 - np.sin(4 * theta)

    # 创建颜色渐变
    cmap = LinearSegmentedColormap.from_list('rose', ['red', 'pink'], N=1000)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, r, color='red')

    # 绘制渐变颜色的玫瑰花瓣
    for i in range(len(theta)):
        ax.plot(theta[i:i+2], r[i:i+2], color=cmap(i/len(theta)))

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_rmax(1.5)
    plt.title('玫瑰花', fontsize=20)
    plt.show()

draw_rose()

# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
#
# def draw_rose():
#     theta = np.linspace(0, 2 * np.pi, 1000)
#     r = 1 - np.sin(4 * theta)
#
#     plt.figure(figsize=(8, 8))
#     ax = plt.subplot(111, projection='polar')
#     ax.plot(theta, r, color='red')
#
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
#     ax.set_rmax(1.5)
#     plt.title('玫瑰花', fontsize=20)
#     plt.show()
#
# draw_rose()