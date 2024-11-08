import torch
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.where(x > 0,1,0) # 0 if x<=0 else 1

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.title('Step Function')
plt.show()