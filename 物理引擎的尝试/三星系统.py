from wuli import *

Phy.tready()
a=Phy(m=200, v=[0, 0, 0], p=[20, 0, 0])
b=Phy(m=200, v=[5, 0, 0], p=[0, -20, 0])# 给一个初速对，就会转圈
c=Phy(m=200, v=[-3, 0, 0], p=[0, 20, 0])
d=Phy(m=100, v=[6, 0, 0], p=[0, 0, 0])
while True:
    a.force2(20, b.p and c.p and d.p)
    b.force2(20, a.p and c.p and d.p)
    c.force2(20, a.p and b.p and d.p)
    d.force2(20, a.p and b.p and c.p)
    Phy.run(t=0.01)
    Phy.tplay()