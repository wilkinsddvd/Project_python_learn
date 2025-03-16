from threading import Timer
import numpy as np
import matplotlib.pyplot as plt


def hello():
    # print("hello, world")
    t = []
    r = []

    data = np.loadtxt('11.txt')

    for i in range(len(data)):
        plt.ion()

        t.append(data[i, 0])
        r.append(data[i, 1])
        theta = np.array(t)
        plt.clf()

        plt.polar(theta * np.pi, r, 'ro', lw=2)
        plt.ylim(0, 1000)
        plt.pause(0.01)
        plt.ioff()
    # plt.show()
    plt.pause(0.1)
    # plt.close()
    plt.clf()


class RepeatingTimer(Timer):
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)


t = RepeatingTimer(0.0, hello)
t.start()

# 没通过测试，等待调试
