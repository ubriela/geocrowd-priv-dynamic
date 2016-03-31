__author__ = 'ubriela'

import math
import numpy

# probability the noisy count is larger than 0
def func(f, T):
    return 1 - 0.5 * math.exp(-math.sqrt(2) * (1-f)/(f*T))

T = 50

probs = [func(f, T) for f in numpy.arange(0.1, 0.99, 0.1)]

for i in probs:
    print i