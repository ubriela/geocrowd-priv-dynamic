__author__ = 'ubriela'

import time
import logging
import sys
import os.path
from os import listdir
from os.path import isfile, join

import numpy as np


sys.path.append('../minball')
sys.path.append('../common')
sys.path.append('../geocast')
sys.path.append('../grid')
sys.path.append('../htree')
sys.path.append('../icde12')
sys.path.append('../dynamic')

from Params import Params

from MultipleDAG import MultipleDAG

from Geocast2 import geocast, post_geocast

from GeocastKNN import geocast_knn
from Utils import is_rect_cover, performed_tasks

# eps_list = [1]
eps_list = [.1, .4, .7, 1.0]

seed_list = [1000]

# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

def read_tasks():
    p = Params(0)
    p.select_dataset()
    data = np.genfromtxt(Params.dataset_task, unpack=True)
    return data


def tasks_gen(data, taskNo, x1=-124.8193, y1=31.3322, x2=-103.0020, y2=49.0025):
    """
    Generate random points within dataset
    """
    all_points = []
    if os.path.isfile(Params.TASKPATH):
        with open(Params.TASKPATH) as f:
            content = f.readlines()
        for i in range(len(seed_list)):
            ran_points = []
            for j in range(taskNo):
                ran_points.append(map(float, content[i * taskNo + j].split()))
            all_points.append(ran_points)
    else:
        tasks = ""
        logging.debug('tasks_gen: generating tasks...')

        boundary = np.array([[x1, y1], [x2, y2]])
        for seed in seed_list:
            ran_points = []
            np.random.seed(seed)
            count = 0
            while count < taskNo:
                idx = np.random.randint(0, data.shape[1])
                _ran_point = data[:, idx]
                if is_rect_cover(boundary, _ran_point):
                    ran_points.append(_ran_point)
                    count = count + 1
            all_points.append(ran_points)
            for item in ran_points:
                tasks = tasks + ("%s\n" % " ".join(map(str, item)))
        outfile = open(Params.TASKPATH, "w")
        outfile.write(tasks)
        outfile.close()
    return all_points


# multiple time instance
def readInstances(input_dir):
    files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

    all_data = []
    for file in files:
        data = np.genfromtxt(input_dir + file, unpack=True)
        all_data.append(data)
    return all_data


def evalDynamic_Baseline(all_workers, all_tasks):
    """
    Evaluate geocast algorithm in privacy mode and in non-privacy mode
    """
    logging.info("evalDynamic_Baseline")
    exp_name = "Dynamic_Baseline"
    methodList = ["Dynamic", "Baseline"]

    res_cube_anw = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    res_cube_atd = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    res_cube_atd_fcfs = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    res_cube_appt = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    res_cube_cell = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    res_cube_cmp = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    res_cube_hop = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    res_cube_hop2 = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    res_cube_cov = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            Params.CUSTOMIZED_GRANULARITY = True
            Params.PARTIAL_CELL_SELECTION = True
            Params.COST_FUNCTION = "distance"
            p = Params(seed_list[j])
            p.Eps = eps_list[i]

            dag = MultipleDAG(all_workers, p)
            dag.publishSimple()
            # dag.applyKalmanFilter()

            totalANW_Naive, totalANW_Geocast, totalANW_Knn = 0, 0, 0
            totalATD_Naive, totalATD_Naive_FCFS, totalATD_Geocast, totalATD_FCFS_Geocast, totalATD_Knn, totalATD_Knn_FCFS = 0, 0, 0, 0, 0, 0
            totalCell_Geocast = 0
            totalCompactness_Geocast = 0
            totalPerformedTasks_Naive, totalPerformedTasks_Geocast, totalPerformedTasks_Knn = 0, 0, 0
            totalHop_Geocast, totalHop_Knn = 0, 0
            totalHop2_Geocast, totalHop2_Knn = 0, 0
            totalCov_Geocast, totalCov_Knn = 0, 0

            # test all tasks for all time instances
            for ti in range(len(dag.getAGs())):
                tree = dag.getAGs()[ti]
                tasks = all_tasks[j]
                for l in range(len(tasks)):
                    if (l + 1) % Params.LOGGING_STEPS == 0:
                        print ">> " + str(l + 1) + " tasks completed"
                    t = tasks[l]

                    # Geocast
                    q, q_log = geocast(tree, t, p.Eps)
                    no_workers, workers, Cells, no_hops, coverage, no_hops2 = post_geocast(t, q, q_log)
                    performed, worker, dist = performed_tasks(workers, Params.MTD, t, False)
                    if performed:
                        totalPerformedTasks_Geocast += 1
                        totalANW_Geocast += no_workers
                        totalATD_Geocast += dist
                        totalCell_Geocast += len(Cells)
                        totalCompactness_Geocast += q_log[-1][3]
                        totalHop_Geocast += no_hops
                        totalHop2_Geocast += no_hops2
                        totalCov_Geocast += coverage
                    performed, worker, dist_fcfs = performed_tasks(workers, Params.MTD, t, True)
                    if performed:
                        totalATD_FCFS_Geocast += dist_fcfs

                    # Baseline (no privacy)
                    no_workers_knn, performed, dist_knn, dist_knn_FCFS, no_hops, coverage, no_hops2 = geocast_knn(
                        all_workers[ti], t)
                    if performed:
                        totalPerformedTasks_Knn += 1
                        totalANW_Knn += no_workers_knn
                        totalATD_Knn += dist_knn
                        totalATD_Knn_FCFS += dist_knn_FCFS
                        totalHop_Knn += no_hops
                        totalHop2_Knn += no_hops2
                        totalCov_Knn += coverage


            # Geocast
            ANW_Geocast = (totalANW_Geocast + 0.0) / totalPerformedTasks_Geocast
            ATD_Geocast = totalATD_Geocast / totalPerformedTasks_Geocast
            ATD_FCFS_Geocast = totalATD_FCFS_Geocast / totalPerformedTasks_Geocast
            ASC_Geocast = (totalCell_Geocast + 0.0) / totalPerformedTasks_Geocast
            CMP_Geocast = totalCompactness_Geocast / totalPerformedTasks_Geocast
            APPT_Geocast = 100 * float(totalPerformedTasks_Geocast) / (Params.TASK_NO * len(dag.getAGs()))
            HOP_Geocast = float(totalHop_Geocast) / (Params.TASK_NO * len(dag.getAGs()))
            HOP2_Geocast = float(totalHop2_Geocast) / (Params.TASK_NO * len(dag.getAGs()))
            COV_Geocast = 100 * float(totalCov_Geocast) / (Params.TASK_NO * len(dag.getAGs()))

            # Baseline
            ANW_Knn = (totalANW_Knn + 0.0) / totalPerformedTasks_Knn
            ATD_Knn = totalATD_Knn / totalPerformedTasks_Knn
            ATD_FCFS_Knn = totalATD_Knn_FCFS / totalPerformedTasks_Knn
            APPT_Knn = 100 * float(totalPerformedTasks_Knn) / (Params.TASK_NO * len(dag.getAGs()))
            HOP_Knn = float(totalHop_Knn) / (Params.TASK_NO * len(dag.getAGs()))
            HOP2_Knn = float(totalHop2_Knn) / (Params.TASK_NO * len(dag.getAGs()))
            COV_Knn = 100 * float(totalCov_Knn) / (Params.TASK_NO * len(dag.getAGs()))

            res_cube_anw[i, j, 0] = ANW_Geocast
            res_cube_atd[i, j, 0] = ATD_Geocast
            res_cube_atd_fcfs[i, j, 0] = ATD_FCFS_Geocast
            res_cube_appt[i, j, 0] = APPT_Geocast
            res_cube_cell[i, j, 0] = ASC_Geocast
            res_cube_cmp[i, j, 0] = CMP_Geocast
            res_cube_hop[i, j, 0] = HOP_Geocast
            res_cube_hop2[i, j, 0] = HOP2_Geocast
            res_cube_cov[i, j, 0] = COV_Geocast

            res_cube_anw[i, j, 1] = ANW_Knn
            res_cube_atd[i, j, 1] = ATD_Knn
            res_cube_atd_fcfs[i, j, 1] = ATD_FCFS_Knn
            res_cube_appt[i, j, 1] = APPT_Knn
            res_cube_cell[i, j, 1] = 0
            res_cube_cmp[i, j, 1] = 0
            res_cube_hop[i, j, 1] = HOP_Knn
            res_cube_hop2[i, j, 1] = HOP2_Knn
            res_cube_cov[i, j, 1] = COV_Knn

    res_summary_anw = np.average(res_cube_anw, axis=1)
    np.savetxt(Params.resdir + exp_name + '_anw_' + `Params.TASK_NO`, res_summary_anw, fmt='%.4f\t')
    res_summary_atd = np.average(res_cube_atd, axis=1)
    np.savetxt(Params.resdir + exp_name + '_atd_' + `Params.TASK_NO`, res_summary_atd, fmt='%.4f\t')
    res_summary_atd = np.average(res_cube_atd_fcfs, axis=1)
    np.savetxt(Params.resdir + exp_name + '_atd_fcfs_' + `Params.TASK_NO`, res_summary_atd, fmt='%.4f\t')
    res_summary_appt = np.average(res_cube_appt, axis=1)
    np.savetxt(Params.resdir + exp_name + '_appt_' + `Params.TASK_NO`, res_summary_appt, fmt='%.4f\t')
    res_summary_cell = np.average(res_cube_cell, axis=1)
    np.savetxt(Params.resdir + exp_name + '_cell_' + `Params.TASK_NO`, res_summary_cell, fmt='%.4f\t')
    res_summary_cmp = np.average(res_cube_cmp, axis=1)
    np.savetxt(Params.resdir + exp_name + '_cmp_' + `Params.TASK_NO`, res_summary_cmp, fmt='%.4f\t')
    res_summary_hop = np.average(res_cube_hop, axis=1)
    np.savetxt(Params.resdir + exp_name + '_hop_' + `Params.TASK_NO`, res_summary_hop, fmt='%.4f\t')
    res_summary_hop2 = np.average(res_cube_hop2, axis=1)
    np.savetxt(Params.resdir + exp_name + '_hop2_' + `Params.TASK_NO`, res_summary_hop2, fmt='%.4f\t')
    res_summary_cov = np.average(res_cube_cov, axis=1)
    np.savetxt(Params.resdir + exp_name + '_cov_' + `Params.TASK_NO`, res_summary_cov, fmt='%.4f\t')


def evalDynamic_Test():
    logging.basicConfig(level=logging.DEBUG, filename='../log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")
    all_data = readInstances("../../dataset/dynamic/yelp/")

    # task_data = read_tasks()
    # all_tasks = tasks_gen(task_data, Params.TASK_NO, Params.x_min, Params.y_min, Params.x_max, Params.y_max)

    p = Params(1000)
    Params.NDIM, Params.NDATA = all_data[0].shape[0], all_data[0].shape[1]
    Params.LOW, Params.HIGH = np.amin(all_data[0], axis=1), np.amax(all_data[0], axis=1)

    dag = MultipleDAG(all_data, p)
    dag.publishSimple()
    dag.dumpSequenceCounts(True, "true_count")
    dag.dumpSequenceCounts(False, "noisy_count")
    dag.applyKalmanFilter()
    dag.dumpSequenceCounts(False, "noisy_count_kf")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename='../log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    all_workers = readInstances("../../dataset/dynamic/yelp/")
    Params.NDIM, Params.NDATA = all_workers[0].shape[0], all_workers[0].shape[1]
    Params.LOW, Params.HIGH = np.amin(all_workers[0], axis=1), np.amax(all_workers[0], axis=1)

    task_data = read_tasks()
    all_tasks = tasks_gen(task_data, Params.TASK_NO, Params.x_min, Params.y_min, Params.x_max, Params.y_max)

    # evalDynamic_Test()

    evalDynamic_Baseline(all_workers, all_tasks)
