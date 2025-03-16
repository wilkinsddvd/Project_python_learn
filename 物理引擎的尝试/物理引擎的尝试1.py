from wuli import *

Phy.tready()
a = Phy(m=100, v=[0, 0, 0], p=[0, 0, 0])

while True:
    a.force([10, 0, 0])
    Phy.run(t=0.01)
    Phy.tplay()