from wuli import *

Phy.tready()
a=Phy(m=100, v=[0, 0, 0], p=[0, 0, 0])
b=Phy(m=100, v=[3, 0, 0], p=[0, -20, 0])  # 给一个初速对，就会转圈

while True:
    a.force2(20, b.p)
    b.force2(20, a.p)
    Phy.run(t=0.01)
    Phy.tplay()