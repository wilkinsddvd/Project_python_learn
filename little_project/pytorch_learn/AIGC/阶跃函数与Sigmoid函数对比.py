import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x =np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)

plt.plot(x, y1, label='step_function')
plt.plot(x, y2, label='sigmoid')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()