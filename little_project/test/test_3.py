import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

def draw_heart():
    t = np.linspace(0, 2 * np.pi, 1000)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

    # 创建颜色渐变
    cmap = LinearSegmentedColormap.from_list('heart', ['red', 'pink'], N=1000)

    plt.figure(figsize=(8, 8))
    for i in range(len(t)):
        plt.plot(x[i:i+2], y[i:i+2], color=cmap(i/len(t)))

    plt.title('爱心', fontsize=20)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

draw_heart()

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
# def draw_heart():
#     t = np.linspace(0, 2 * np.pi, 1000)
#     x = 16 * np.sin(t)**3
#     y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
#
#     plt.figure(figsize=(8, 8))
#     plt.plot(x, y, color='red')
#     plt.title('爱心', fontsize=20)
#     plt.axis('equal')
#     plt.axis('off')
#     plt.show()
#
# draw_rose()
# draw_heart()