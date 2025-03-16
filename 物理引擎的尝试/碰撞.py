import sys
from wuli import *

t=1
Phy.tready()
a=Phy(m=100, v=[3, 0, 0], p=[0, 0, 0])
b=Phy(m=100, v=[0, 0, 0], p=[50, 0, 0])

while t:
    a.bounce(1000)
    Phy.run(t=0.01)
    Phy.tplay()

