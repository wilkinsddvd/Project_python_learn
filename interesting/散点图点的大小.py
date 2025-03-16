import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 8, 6, 4, 2]
size = [20, 40, 60, 80, 100]

plt.scatter(x, y, s=size)
plt.show()