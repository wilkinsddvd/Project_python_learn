from wuli import *

Phy.tready()
a=Phy(m=100, v=[0, 0, 0], p=[0, 0, 0])
b=Phy(m=100, v=[3, 0, 0], p=[0, -20, 0])

while True:
    a.resilience(x=None, k=100, other=b)
    # resilience 可以把两个点连在弹簧上
    # 第一个参数是原长，为None时为第一次调用时的长度
    # 第二个参数是劲度系数，第三个参数是另一个点
    Phy.run(t=0.01)
    Phy.tplay()
