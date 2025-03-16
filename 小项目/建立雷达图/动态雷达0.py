import numpy as np
import matplotlib.pyplot as plt

t = []
r = []

# 建一个txt文件放自己的数据
data = np.loadtxt('11.txt')

for i in range(len(data)):
    plt.ion()  # 这个必须

    t.append(data[i, 0])
    r.append(data[i, 1])
    theta = np.array(t)

    # 清屏
    plt.clf()

    # 画极坐标图
    plt.polar(theta * np.pi, r, 'ro', lw=2)
    plt.ylim(0, 1000)
    plt.pause(0.5)

    plt.ioff()  # 这个也必须

plt.show()