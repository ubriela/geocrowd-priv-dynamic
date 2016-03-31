__author__ = 'ubriela'
import numpy as np
import random

from Params import Params

from Geocrowd import rect_query_points, hops_expansion
from Utils import distance, acc_rate, is_performed, performed_tasks

"""
This function implements WST mode, in which workers voluntarily select tasks.
When a task arrives, each worker generates a number between 0 and 1, if the number if less than p^a,
then the worker agrees to a task. Then, among all remaining workers, one of them is randomly chosen
(the one that responds first) to complete the task.

Data: workers' locations
t: task location
Return: no_workers_knn, performed, dist_knn, dist_knn_FCFS, no_hops, coverage, no_hops2
"""
def selection_WST(data, t, tree=None):
    # find all workers in MTD
    MTD_RECT = np.array([[t[0] - Params.ONE_KM * Params.MTD, t[1] - Params.ONE_KM * Params.MTD],
                         [t[0] + Params.ONE_KM * Params.MTD, t[1] + Params.ONE_KM * Params.MTD]])
    locs = rect_query_points(data, MTD_RECT).transpose()
    #locs = sorted(locs, key=lambda loc: distance(loc[0], loc[1], t[0], t[1]))

    #u = 0
    workers, dists = np.zeros(shape=(2, 0)), []

    # find workers who would perform the task
    for loc in locs:
        dist = distance(loc[0], loc[1], t[0], t[1])
        u_c = acc_rate(Params.MTD, dist)
        #u = 1 - (1 - u) * (1 - u_c)
        if is_performed(u_c):
            workers = np.concatenate([workers, np.array([[loc[0]], [loc[1]]])], axis=1)
            dists.append(dist)

    # simulation
    if workers.shape[1] == 0:  # no workers
            return 0, False, None, None, None, None, None

    return len(locs), True, 0, random.choice(dists), 0, 0, 0