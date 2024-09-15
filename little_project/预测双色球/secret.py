# -*- coding: UTF-8 -*-
import random
for i in range (1,100):
    a = random.randint(1, 33)
    b = random.randint(1, 33)
    c = random.randint(1, 33)
    d = random.randint(1, 33)
    e = random.randint(1, 33)
    f = random.randint(1, 33)
    g = random.randint(1, 16)

    if a < b and b < c and c < d and d < e and e < f:
        print(a, b, c, d, e, f, g)

# 8 18 24 25 32 33 16
# 12 13 20 23 27 32 3
# 4 8 12 14 18 19 9
# 6 12 15 20 22 24 7
# 12 13 17 19 22 33 15

#2 10 12 14 19 28 12
#1 10 21 26 28 33 16

# 2 8 11 15 19 25 3
# 9 13 21 24 25 29 11

# 5 18 23 24 25 29 4