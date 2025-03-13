import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

x_values = np.linspace(-5, 5, 400)
y = np.tanh(x_values)
plt.plot(x_values, y)
plt.title("tanh function")
plt.show()