import time as t
import numpy as np

a = np.zeros(20000000)
b = np.ones(20000000)
c = np.zeros(20000000)
d = np.ones(20000000)

before = t.time()
x = a**2
after = t.time()

print(after - before)
