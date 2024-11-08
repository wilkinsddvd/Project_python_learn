import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Times New Roman',size=15)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def plot_softmax():
    x = np.linspace(-10, 10, 200)
    y = softmax(x)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, color='r', lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([-10.05, 10.05])
    plt.ylim([-0.02, 0.1])
    plt.tight_layout()
    plt.show()
plot_softmax()
