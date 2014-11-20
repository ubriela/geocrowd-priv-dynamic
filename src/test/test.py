#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.


import sys
import time

import numpy as np


sys.path.append('../geocast')
sys.path.append('../common')
from Params import Params
from Utils import acc_rate

eps_list = [0.01, 0.05, 0.1, 0.5, 1.0]
T = 1
size = 100000
# test laplace
# true = np.ndarray(shape=(size,1))
# noisy = np.ndarray(shape=(size,1))
#
# for i in range(size):
# true[i] = i
#     n_count = true[i] + np.random.laplace(0, 1/0.1, 1)[0]
#     noisy[i] = n_count
#
# print np.sqrt(2*(1/0.1)**2), np.sqrt(((noisy - true) ** 2).mean())


for eps in eps_list:
    true = np.genfromtxt("../log/true_count_KF_" + str(eps) + ".log", unpack=True)
    noisy = np.genfromtxt("../log/noisy_count_KF_" + str(eps) + ".log", unpack=True)
    true_flatten = true.flatten()
    noisy_flatten = noisy.flatten()
    p = Params(1000)
    print "KF\t", "\t", np.sum(true_flatten), "\t", np.sum(noisy_flatten), "\t", np.sqrt(
        2 * (T / (eps * (1 - p.structureEps))) ** 2), "\t", np.sqrt(((noisy_flatten - true_flatten) ** 2).mean())

    true = np.genfromtxt("../log/true_count_KFPID_" + str(eps) + ".log", unpack=True)
    noisy = np.genfromtxt("../log/noisy_count_KFPID_" + str(eps) + ".log", unpack=True)
    true_flatten = true.flatten()
    noisy_flatten = noisy.flatten()
    p = Params(1000)
    print "KFPID\t", "\t", np.sum(true_flatten), "\t", np.sum(noisy_flatten), "\t", np.sqrt(
        2 * (T / (eps * (1 - p.structureEps))) ** 2), "\t", np.sqrt(((noisy_flatten - true_flatten) ** 2).mean())

x1 = time.time()
# for i in range(100):
max_dist = 100
dist = range(100)
Params.ZIPF_STEPS = 20
Params.s = 1
Params.MAR = 0.1
Params.AR_FUNCTION = "linear"
y = "\n".join(map(str, [acc_rate(max_dist, x) for x in dist]))
print y
print time.time() - x1