__author__ = 'ubriela'

from KalmanFilterPID import KalmanFilterPID
from Params import Params

eps_list = [1.0]
q_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]


def distance(a, b):
    """ generated source for method distance """
    return abs((a - b + 0.0) / b)


def getRelError(publish, orig):
    """ generated source for method getRelError """
    error = 0
    for i in range(len(publish)):
        if orig[i] < 0:
            error += distance(publish[i], orig[i])
        elif orig[i] == 0:
            error += distance(publish[i], 1)
        else:
            error += distance(max(publish[i], 0), orig[i])
    return error / len(publish)


for q in q_list:
    for i in range(len(eps_list)):
        with open("../log/foursquare/true_count_KF_" + str(eps_list[i]) + ".log") as f:
            rel_errs = []
            for line in f.readlines():
                orig = map(float, line.strip().split('\t'))
                p = Params(1000)
                p.Eps = eps_list[i]

                kf = KalmanFilterPID(p)
                kf.setQ(q)
                budgetKF = budgetKF = eps_list[i] / 2
                # filter = kf.kalmanFilter(seq, budgetKF, p.samplingRate)
                publish = kf.kalmanFilter(orig, budgetKF)

                rel_err = getRelError(publish, orig)
                rel_errs.append(rel_err)

            print q, "\t", eps_list[i], "\t", sum(rel_errs) / len(rel_errs)




