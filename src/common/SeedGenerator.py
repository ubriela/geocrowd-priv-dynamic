import timeit

import random


list = []
for i in range(15):
    list.append(random.randint(0, 10000))
print list


def f():
    x = [i for j in range(1000)]


print timeit.timeit(f)