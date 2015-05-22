__author__ = 'ubriela'

import time
import logging
import sys
from os import listdir
from os.path import isfile, join
import os
import gc
from multiprocessing import Pool
import random

import psutil
import numpy as np
from scipy import spatial


sys.path.append('../minball')
sys.path.append('../common')
sys.path.append('../geocast')
sys.path.append('../grid')
sys.path.append('../htree')
sys.path.append('../icde12')
sys.path.append('../dynamic')

from Params import Params

from MultipleDAG import MultipleDAG
from MultipleDAG_KF import MultipleDAG_KF

from GeocastM import geocast, post_geocast, simple_post_geocast
from Grid_adaptiveM import Grid_adaptiveM

from GeocastKNN import geocast_knn
from Utils import is_rect_cover, performed_tasks

# eps_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
eps_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

first_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

T_list = [50,60,70,80,90,100]

EU_list = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# seed_list = [9110]
seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]


def read_tasks(p):
    p.select_dataset()
    data = np.genfromtxt(p.dataset_task, unpack=True)
    return data


def tasks_gen(data, param):
    """
    Generate random points within dataset
    """
    taskNo, x1, y1, x2, y2 = param.TASK_NO, param.x_min, param.y_min, param.x_max, param.y_max
    all_points = []


    if os.path.isfile(param.TASKPATH):
        with open(param.TASKPATH) as f:
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
        outfile = open(param.TASKPATH, "w")
        outfile.write(tasks)
        outfile.close()
    return all_points


# multiple time instance
def readInstances(input_dir):
    files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

    all_data = []
    i = 0
    first_prob = 1
    prob = first_prob
    for file in files:
        data = np.genfromtxt(input_dir + file, unpack=True)

        data = filter(data, prob)
        print prob, data.shape[1]
        all_data.append(data)
        for p in np.transpose(data):
            if p[0] == 0 or p[1] == 0:
                print p, file
        i = i + 1
        prob = prob - 0.9/len(files)
    return all_data


"""
p percentage of the data online
"""
def filter(data, prob):
    idx = np.random.randint(0, data.shape[1], int(prob * data.shape[1]))
    filter_data = data[:, idx]


    return filter_data


def evalDynamic_Baseline(params):
    """
    Evaluate geocast algorithm in privacy mode and in non-privacy mode
    """

    all_workers = params[0]
    all_tasks = params[1]
    p = params[2]
    eps = params[3]

    logging.info("evalDynamic_Baseline")
    exp_name = "Dynamic_Baseline"
    methodList = ["BasicD", "BasicS", "KF", "KFPID", "Baseline"]
    # methodList = ["BasicS"]

    eps_list = [eps]

    res_cube_anw = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    # res_cube_atd = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    res_cube_atd_fcfs = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    res_cube_appt = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    res_cube_cell = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    # res_cube_cmp = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    res_cube_hop = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    # res_cube_hop2 = np.zeros((len(eps_list), len(seed_list), len(methodList)))
    # res_cube_cov = np.zeros((len(eps_list), len(seed_list), len(methodList)))

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            # Params(seed_list[j])
            p.Seed = seed_list[j]
            p.Eps = eps_list[i]

            for method in methodList[0:len(methodList)]:
                proc = psutil.Process(os.getpid())
                if method == "BasicD":
                    dag = MultipleDAG(all_workers, p)
                    dag.publish()
                elif method == "BasicS":
                    dag = MultipleDAG(all_workers, p, True)
                    dag.publish()
                elif method == "KF":
                    dag = MultipleDAG_KF(all_workers, p)
                    dag.publish()
                    dag.applyKalmanFilter()
                    # dag.dumpSequenceCounts(True, "true_count_" + method + "_" + str(eps_list[i]))
                    # dag.dumpSequenceCounts(False, "noisy_count_" + method + "_" + str(eps_list[i]))

                elif method == "KFPID":
                    dag = MultipleDAG_KF(all_workers, p, True)
                    dag.publish()
                    dag.applyKalmanFilter()
                    # dag.dumpSequenceCounts(True, "true_count_" + method + "_" + str(eps_list[i]))
                    # dag.dumpSequenceCounts(False, "noisy_count_" + method + "_" + str(eps_list[i]))
                else:
                    continue;

                totalANW_Geocast = 0
                # totalATD_Geocast = 0
                totalATD_FCFS_Geocast = 0
                totalCell_Geocast = 0
                # totalCompactness_Geocast = 0
                totalPerformedTasks_Geocast = 0
                totalHop_Geocast = 0
                # totalHop2_Geocast = 0
                # totalCov_Geocast = 0

                T = len(dag.getAGs())
                if T == 0:
                        continue
                # test all tasks for all time instances
                for ti in range(T):
                    # free memory of previous instances
                    for l in range(len(all_tasks[j])):
                        if (l + 1) % Params.LOGGING_STEPS == 0:
                            print ">> " + str(l + 1) + " tasks completed"
                        t = all_tasks[j][l]

                        # Geocast
                        q, q_log = geocast(dag.getAGs()[ti], t, p.Eps)
                        no_workers, workers, Cells, no_hops = simple_post_geocast(t, q, q_log)
                        # performed, worker, dist = performed_tasks(workers, Params.MTD, t, False)

                        performed, worker, dist_fcfs = performed_tasks(workers, Params.MTD, t, True)
                        if performed:
                            totalPerformedTasks_Geocast += 1
                            totalANW_Geocast += no_workers
                            # totalATD_Geocast += dist
                            totalCell_Geocast += Cells
                            # totalCompactness_Geocast += q_log[-1][3]
                            totalHop_Geocast += no_hops
                            # totalHop2_Geocast += no_hops2
                            # totalCov_Geocast += coverage

                            # if performed:
                            totalATD_FCFS_Geocast += dist_fcfs

                    if len(dag.getAGs()) - 1 == ti:
                        dag.clearMemory()
                        break

                # Geocast
                ANW_Geocast = (totalANW_Geocast + 0.0) / max(1, totalPerformedTasks_Geocast)
                # ATD_Geocast = totalATD_Geocast / totalPerformedTasks_Geocast
                ATD_FCFS_Geocast = totalATD_FCFS_Geocast / max(1, totalPerformedTasks_Geocast)
                ASC_Geocast = (totalCell_Geocast + 0.0) / max(1, totalPerformedTasks_Geocast)
                # CMP_Geocast = totalCompactness_Geocast / totalPerformedTasks_Geocast
                APPT_Geocast = 100 * float(totalPerformedTasks_Geocast) / (Params.TASK_NO * T)
                HOP_Geocast = float(totalHop_Geocast) / (Params.TASK_NO * T)
                # HOP2_Geocast = float(totalHop2_Geocast) / (Params.TASK_NO * T)
                # COV_Geocast = 100 * float(totalCov_Geocast) / (Params.TASK_NO * T)

                res_cube_anw[i, j, methodList.index(method)] = ANW_Geocast
                # res_cube_atd[i, j, methodList.index(method)] = ATD_Geocast
                res_cube_atd_fcfs[i, j, methodList.index(method)] = ATD_FCFS_Geocast
                res_cube_appt[i, j, methodList.index(method)] = APPT_Geocast
                res_cube_cell[i, j, methodList.index(method)] = ASC_Geocast
                # res_cube_cmp[i, j, methodList.index(method)] = CMP_Geocast
                res_cube_hop[i, j, methodList.index(method)] = HOP_Geocast
                # res_cube_hop2[i, j, methodList.index(method)] = HOP2_Geocast
                # res_cube_cov[i, j, methodList.index(method)] = COV_Geocast

                gc.collect()
                proc.get_memory_info().rss

    # do not need to varying eps for non-privacy technique!
    for j in range(len(seed_list)):
        totalANW_Knn = 0
        # totalATD_Knn = 0
        totalATD_Knn_FCFS = 0
        totalPerformedTasks_Knn = 0
        totalHop_Knn = 0
        # totalHop2_Knn = 0
        # totalCov_Knn = 0

        tasks = all_tasks[j]
        # test all tasks for all time instances
        for ti in range(len(all_workers)):
            for l in range(len(tasks)):
                t = tasks[l]

                # Baseline (no privacy)
                no_workers_knn, performed, dist_knn, dist_knn_FCFS, no_hops, coverage, no_hops2 = geocast_knn(
                    all_workers[ti], t)
                if performed:
                    totalPerformedTasks_Knn += 1
                    totalANW_Knn += no_workers_knn
                    # totalATD_Knn += dist_knn
                    totalATD_Knn_FCFS += dist_knn_FCFS
                    totalHop_Knn += no_hops
                    # totalHop2_Knn += no_hops2
                    # totalCov_Knn += coverage

        # Baseline
        ANW_Knn = (totalANW_Knn + 0.0) / max(1, totalPerformedTasks_Knn)
        # ATD_Knn = totalATD_Knn / totalPerformedTasks_Knn
        ATD_FCFS_Knn = totalATD_Knn_FCFS / max(1, totalPerformedTasks_Knn)
        APPT_Knn = 100 * float(totalPerformedTasks_Knn) / (Params.TASK_NO*len(all_workers))
        HOP_Knn = float(totalHop_Knn) / (Params.TASK_NO*len(all_workers))
        # HOP2_Knn = float(totalHop2_Knn) / (Params.TASK_NO * T)
        # COV_Knn = 100 * float(totalCov_Knn) / (Params.TASK_NO * T)

        res_cube_anw[:, j, len(methodList) - 1] = ANW_Knn
        # res_cube_atd[:, j, len(methodList) - 1] = ATD_Knn
        res_cube_atd_fcfs[:, j, len(methodList) - 1] = ATD_FCFS_Knn
        res_cube_appt[:, j, len(methodList) - 1] = APPT_Knn
        res_cube_cell[:, j, len(methodList) - 1] = 0
        # res_cube_cmp[:, j, len(methodList) - 1] = 0
        res_cube_hop[:, j, len(methodList) - 1] = HOP_Knn
        # res_cube_hop2[:, j, len(methodList) - 1] = HOP2_Knn
        # res_cube_cov[:, j, len(methodList) - 1] = COV_Knn

    res_summary_anw = np.average(res_cube_anw, axis=1)
    np.savetxt(p.resdir + exp_name + '_anw_' + str(eps) + "_" + `Params.TASK_NO`, res_summary_anw, fmt='%.4f\t')
    # res_summary_atd = np.average(res_cube_atd, axis=1)
    # np.savetxt(p.resdir + exp_name + '_atd_' + str(eps) + "_"  + `Params.TASK_NO`, res_summary_atd, fmt='%.4f\t')
    res_summary_atd = np.average(res_cube_atd_fcfs, axis=1)
    np.savetxt(p.resdir + exp_name + '_atd_fcfs_' + str(eps) + "_" + `Params.TASK_NO`, res_summary_atd, fmt='%.4f\t')
    res_summary_appt = np.average(res_cube_appt, axis=1)
    np.savetxt(p.resdir + exp_name + '_appt_' + str(eps) + "_" + `Params.TASK_NO`, res_summary_appt, fmt='%.4f\t')
    res_summary_cell = np.average(res_cube_cell, axis=1)
    np.savetxt(p.resdir + exp_name + '_cell_' + str(eps) + "_" + `Params.TASK_NO`, res_summary_cell, fmt='%.4f\t')
    # res_summary_cmp = np.average(res_cube_cmp, axis=1)
    # np.savetxt(p.resdir + exp_name + '_cmp_' + str(eps) + "_"  + `Params.TASK_NO`, res_summary_cmp, fmt='%.4f\t')
    res_summary_hop = np.average(res_cube_hop, axis=1)
    np.savetxt(p.resdir + exp_name + '_hop_' + str(eps) + "_" + `Params.TASK_NO`, res_summary_hop, fmt='%.4f\t')
    # res_summary_hop2 = np.average(res_cube_hop2, axis=1)
    # np.savetxt(p.resdir + exp_name + '_hop2_' + str(eps) + "_"  + `Params.TASK_NO`, res_summary_hop2, fmt='%.4f\t')
    # res_summary_cov = np.average(res_cube_cov, axis=1)
    # np.savetxt(p.resdir + exp_name + '_cov_' + str(eps) + "_"  + `Params.TASK_NO`, res_summary_cov, fmt='%.4f\t')


def evalDynamic_Baseline_F(params):
    """
    Evaluate geocast algorithm in privacy mode and in non-privacy mode
    """

    all_workers = params[0]
    all_tasks = params[1]
    p = params[2]
    first = params[3]

    logging.info("evalDynamic_Baseline_F")
    exp_name = "Dynamic_Baseline_First"
    methodList = ["BasicD", "KFPID", "Baseline"]

    first_list = [first]

    res_cube_anw = np.zeros((len(first_list), len(seed_list), len(methodList)))
    res_cube_atd_fcfs = np.zeros((len(first_list), len(seed_list), len(methodList)))
    res_cube_appt = np.zeros((len(first_list), len(seed_list), len(methodList)))
    res_cube_cell = np.zeros((len(first_list), len(seed_list), len(methodList)))
    res_cube_hop = np.zeros((len(first_list), len(seed_list), len(methodList)))

    for j in range(len(seed_list)):
        for i in range(len(first_list)):
            p.Seed = seed_list[j]
            p.structureEps = first_list[i]

            for method in methodList[0:len(methodList) - 1]:
                proc = psutil.Process(os.getpid())
                if method == "BasicD":
                    dag = MultipleDAG(all_workers, p)
                    dag.publish()

                elif method == "KF":
                    dag = MultipleDAG_KF(all_workers, p)
                    dag.publish()
                    dag.applyKalmanFilter()
                    # dag.dumpSequenceCounts(True, "true_count_" + method + "_" + str(eps_list[i]))
                    # dag.dumpSequenceCounts(False, "noisy_count_" + method + "_" + str(eps_list[i]))

                elif method == "KFPID":
                    dag = MultipleDAG_KF(all_workers, p, True)
                    dag.publish()
                    dag.applyKalmanFilter()
                    # dag.dumpSequenceCounts(True, "true_count_" + method + "_" + str(eps_list[i]))
                    # dag.dumpSequenceCounts(False, "noisy_count_" + method + "_" + str(eps_list[i]))

                totalANW_Geocast = 0
                totalATD_FCFS_Geocast = 0
                totalCell_Geocast = 0
                totalPerformedTasks_Geocast = 0
                totalHop_Geocast = 0

                T = len(dag.getAGs())
                if T == 0:
                        continue
                # test all tasks for all time instances
                for ti in range(T):
                    # free memory of previous instances
                    for l in range(len(all_tasks[j])):
                        if (l + 1) % Params.LOGGING_STEPS == 0:
                            print ">> " + str(l + 1) + " tasks completed"
                        t = all_tasks[j][l]

                        # Geocast
                        q, q_log = geocast(dag.getAGs()[ti], t, p.Eps)
                        no_workers, workers, Cells, no_hops = simple_post_geocast(t, q, q_log)
                        performed, worker, dist_fcfs = performed_tasks(workers, Params.MTD, t, True)
                        if performed:
                            totalPerformedTasks_Geocast += 1
                            totalANW_Geocast += no_workers
                            totalCell_Geocast += Cells
                            totalHop_Geocast += no_hops

                            # if performed:
                            totalATD_FCFS_Geocast += dist_fcfs

                    if len(dag.getAGs()) - 1 == ti:
                        dag.clearMemory()
                        break

                # Geocast
                ANW_Geocast = (totalANW_Geocast + 0.0) / max(1, totalPerformedTasks_Geocast)
                ATD_FCFS_Geocast = totalATD_FCFS_Geocast / max(1, totalPerformedTasks_Geocast)
                ASC_Geocast = (totalCell_Geocast + 0.0) / max(1, totalPerformedTasks_Geocast)
                APPT_Geocast = 100 * float(totalPerformedTasks_Geocast) / (Params.TASK_NO * T)
                HOP_Geocast = float(totalHop_Geocast) / (Params.TASK_NO * T)

                res_cube_anw[i, j, methodList.index(method)] = ANW_Geocast
                res_cube_atd_fcfs[i, j, methodList.index(method)] = ATD_FCFS_Geocast
                res_cube_appt[i, j, methodList.index(method)] = APPT_Geocast
                res_cube_cell[i, j, methodList.index(method)] = ASC_Geocast
                res_cube_hop[i, j, methodList.index(method)] = HOP_Geocast

                gc.collect()
                proc.get_memory_info().rss

    # do not need to varying eps for non-privacy technique!
    for j in range(len(seed_list)):
        totalANW_Knn = 0
        totalATD_Knn_FCFS = 0
        totalPerformedTasks_Knn = 0
        totalHop_Knn = 0

        tasks = all_tasks[j]
        # test all tasks for all time instances
        for ti in range(len(all_workers)):
            for l in range(len(tasks)):
                t = tasks[l]

                # Baseline (no privacy)
                no_workers_knn, performed, dist_knn, dist_knn_FCFS, no_hops, coverage, no_hops2 = geocast_knn(
                    all_workers[ti], t)
                if performed:
                    totalPerformedTasks_Knn += 1
                    totalANW_Knn += no_workers_knn
                    totalATD_Knn_FCFS += dist_knn_FCFS
                    totalHop_Knn += no_hops

        # Baseline
        ANW_Knn = (totalANW_Knn + 0.0) / max(1, totalPerformedTasks_Knn)
        ATD_FCFS_Knn = totalATD_Knn_FCFS / max(1, totalPerformedTasks_Knn)
        APPT_Knn = 100 * float(totalPerformedTasks_Knn) / (Params.TASK_NO * T)
        HOP_Knn = float(totalHop_Knn) / (Params.TASK_NO * T)

        res_cube_anw[:, j, len(methodList) - 1] = ANW_Knn
        res_cube_atd_fcfs[:, j, len(methodList) - 1] = ATD_FCFS_Knn
        res_cube_appt[:, j, len(methodList) - 1] = APPT_Knn
        res_cube_cell[:, j, len(methodList) - 1] = 0
        res_cube_hop[:, j, len(methodList) - 1] = HOP_Knn

    res_summary_anw = np.average(res_cube_anw, axis=1)
    np.savetxt(p.resdir + exp_name + '_anw_' + str(first) + "_" + `Params.TASK_NO`, res_summary_anw, fmt='%.4f\t')
    res_summary_atd = np.average(res_cube_atd_fcfs, axis=1)
    np.savetxt(p.resdir + exp_name + '_atd_fcfs_' + str(first) + "_" + `Params.TASK_NO`, res_summary_atd, fmt='%.4f\t')
    res_summary_appt = np.average(res_cube_appt, axis=1)
    np.savetxt(p.resdir + exp_name + '_appt_' + str(first) + "_" + `Params.TASK_NO`, res_summary_appt, fmt='%.4f\t')
    res_summary_cell = np.average(res_cube_cell, axis=1)
    np.savetxt(p.resdir + exp_name + '_cell_' + str(first) + "_" + `Params.TASK_NO`, res_summary_cell, fmt='%.4f\t')
    res_summary_hop = np.average(res_cube_hop, axis=1)
    np.savetxt(p.resdir + exp_name + '_hop_' + str(first) + "_" + `Params.TASK_NO`, res_summary_hop, fmt='%.4f\t')


def evalDynamic_Baseline_T(params):
    """
    Evaluate geocast algorithm in privacy mode and in non-privacy mode
    """

    all_workers = params[0]
    all_tasks = params[1]
    p = params[2]
    T = params[3]

    logging.info("evalDynamic_Baseline_T")
    exp_name = "Dynamic_Baseline_T"
    methodList = ["Basic", "KF", "KFPID", "Baseline"]


    res_cube_anw = np.zeros((1, len(seed_list), len(methodList)))
    res_cube_atd_fcfs = np.zeros((1, len(seed_list), len(methodList)))
    res_cube_appt = np.zeros((1, len(seed_list), len(methodList)))
    res_cube_cell = np.zeros((1, len(seed_list), len(methodList)))
    res_cube_hop = np.zeros((1, len(seed_list), len(methodList)))

    for j in range(len(seed_list)):
        for i in range(1):
            p.Seed = seed_list[j]

            for method in methodList[0:len(methodList) - 1]:
                proc = psutil.Process(os.getpid())
                if method == "Basic":
                    dag = MultipleDAG(all_workers, p)
                    dag.publish()

                elif method == "KF":
                    dag = MultipleDAG_KF(all_workers, p)
                    dag.publish()
                    dag.applyKalmanFilter()
                    # dag.dumpSequenceCounts(True, "true_count_" + method + "_" + str(eps_list[i]))
                    # dag.dumpSequenceCounts(False, "noisy_count_" + method + "_" + str(eps_list[i]))

                elif method == "KFPID":
                    dag = MultipleDAG_KF(all_workers, p, True)
                    dag.publish()
                    dag.applyKalmanFilter()
                    # dag.dumpSequenceCounts(True, "true_count_" + method + "_" + str(eps_list[i]))
                    # dag.dumpSequenceCounts(False, "noisy_count_" + method + "_" + str(eps_list[i]))

                totalANW_Geocast = 0
                totalATD_FCFS_Geocast = 0
                totalCell_Geocast = 0
                totalPerformedTasks_Geocast = 0
                totalHop_Geocast = 0

                T = len(dag.getAGs())
                # test all tasks for all time instances
                for ti in range(T):
                    if len(dag.getAGs()) == 0:
                        break
                    # free memory of previous instances
                    for l in range(len(all_tasks[j])):
                        if (l + 1) % Params.LOGGING_STEPS == 0:
                            print ">> " + str(l + 1) + " tasks completed"
                        t = all_tasks[j][l]

                        # Geocast
                        q, q_log = geocast(dag.getAGs()[ti], t, p.Eps)
                        no_workers, workers, Cells, no_hops = simple_post_geocast(t, q, q_log)
                        performed, worker, dist_fcfs = performed_tasks(workers, Params.MTD, t, True)
                        if performed:
                            totalPerformedTasks_Geocast += 1
                            totalANW_Geocast += no_workers
                            totalCell_Geocast += Cells
                            totalHop_Geocast += no_hops

                            # if performed:
                            totalATD_FCFS_Geocast += dist_fcfs

                    if len(dag.getAGs()) - 1 == ti:
                        dag.clearMemory()
                        break

                # Geocast
                ANW_Geocast = (totalANW_Geocast + 0.0) / totalPerformedTasks_Geocast
                ATD_FCFS_Geocast = totalATD_FCFS_Geocast / totalPerformedTasks_Geocast
                ASC_Geocast = (totalCell_Geocast + 0.0) / totalPerformedTasks_Geocast
                APPT_Geocast = 100 * float(totalPerformedTasks_Geocast) / (Params.TASK_NO * T)
                HOP_Geocast = float(totalHop_Geocast) / (Params.TASK_NO * T)

                res_cube_anw[i, j, methodList.index(method)] = ANW_Geocast
                res_cube_atd_fcfs[i, j, methodList.index(method)] = ATD_FCFS_Geocast
                res_cube_appt[i, j, methodList.index(method)] = APPT_Geocast
                res_cube_cell[i, j, methodList.index(method)] = ASC_Geocast
                res_cube_hop[i, j, methodList.index(method)] = HOP_Geocast

                gc.collect()
                proc.get_memory_info().rss

    # do not need to varying eps for non-privacy technique!
    for j in range(len(seed_list)):
        totalANW_Knn = 0
        totalATD_Knn_FCFS = 0
        totalPerformedTasks_Knn = 0
        totalHop_Knn = 0

        tasks = all_tasks[j]
        # test all tasks for all time instances
        for ti in range(len(all_workers)):
            for l in range(len(tasks)):
                t = tasks[l]

                # Baseline (no privacy)
                no_workers_knn, performed, dist_knn, dist_knn_FCFS, no_hops, coverage, no_hops2 = geocast_knn(
                    all_workers[ti], t)
                if performed:
                    totalPerformedTasks_Knn += 1
                    totalANW_Knn += no_workers_knn
                    totalATD_Knn_FCFS += dist_knn_FCFS
                    totalHop_Knn += no_hops

        # Baseline
        ANW_Knn = (totalANW_Knn + 0.0) / totalPerformedTasks_Knn
        ATD_FCFS_Knn = totalATD_Knn_FCFS / totalPerformedTasks_Knn
        APPT_Knn = 100 * float(totalPerformedTasks_Knn) / (Params.TASK_NO * T)
        HOP_Knn = float(totalHop_Knn) / (Params.TASK_NO * T)

        res_cube_anw[:, j, len(methodList) - 1] = ANW_Knn
        res_cube_atd_fcfs[:, j, len(methodList) - 1] = ATD_FCFS_Knn
        res_cube_appt[:, j, len(methodList) - 1] = APPT_Knn
        res_cube_cell[:, j, len(methodList) - 1] = 0
        res_cube_hop[:, j, len(methodList) - 1] = HOP_Knn

    res_summary_anw = np.average(res_cube_anw, axis=1)
    np.savetxt(p.resdir + exp_name + '_anw_' + str(T) + "_" + `Params.TASK_NO`, res_summary_anw, fmt='%.4f\t')
    res_summary_atd = np.average(res_cube_atd_fcfs, axis=1)
    np.savetxt(p.resdir + exp_name + '_atd_fcfs_' + str(T) + "_" + `Params.TASK_NO`, res_summary_atd, fmt='%.4f\t')
    res_summary_appt = np.average(res_cube_appt, axis=1)
    np.savetxt(p.resdir + exp_name + '_appt_' + str(T) + "_" + `Params.TASK_NO`, res_summary_appt, fmt='%.4f\t')
    res_summary_cell = np.average(res_cube_cell, axis=1)
    np.savetxt(p.resdir + exp_name + '_cell_' + str(T) + "_" + `Params.TASK_NO`, res_summary_cell, fmt='%.4f\t')
    res_summary_hop = np.average(res_cube_hop, axis=1)
    np.savetxt(p.resdir + exp_name + '_hop_' + str(T) + "_" + `Params.TASK_NO`, res_summary_hop, fmt='%.4f\t')



def evalDynamic_Baseline_EU(params):
    """
    Evaluate geocast algorithm in privacy mode and in non-privacy mode
    """

    all_workers = params[0]
    all_tasks = params[1]
    p = params[2]
    EU = params[3]

    logging.info("evalDynamic_Baseline_EU")
    exp_name = "Dynamic_Baseline_First"
    methodList = ["Basic", "KF", "KFPID", "Baseline"]

    EU_list = [EU]

    res_cube_anw = np.zeros((len(EU_list), len(seed_list), len(methodList)))
    res_cube_atd_fcfs = np.zeros((len(EU_list), len(seed_list), len(methodList)))
    res_cube_appt = np.zeros((len(EU_list), len(seed_list), len(methodList)))
    res_cube_cell = np.zeros((len(EU_list), len(seed_list), len(methodList)))
    res_cube_hop = np.zeros((len(EU_list), len(seed_list), len(methodList)))

    for j in range(len(seed_list)):
        for i in range(len(first_list)):
            p.Seed = seed_list[j]
            p.U = EU_list[i]

            for method in methodList[0:len(methodList) - 1]:
                proc = psutil.Process(os.getpid())
                if method == "Basic":
                    dag = MultipleDAG(all_workers, p)
                    dag.publish()

                elif method == "KF":
                    dag = MultipleDAG_KF(all_workers, p)
                    dag.publish()
                    dag.applyKalmanFilter()
                    # dag.dumpSequenceCounts(True, "true_count_" + method + "_" + str(eps_list[i]))
                    # dag.dumpSequenceCounts(False, "noisy_count_" + method + "_" + str(eps_list[i]))

                elif method == "KFPID":
                    dag = MultipleDAG_KF(all_workers, p, True)
                    dag.publish()
                    dag.applyKalmanFilter()
                    # dag.dumpSequenceCounts(True, "true_count_" + method + "_" + str(eps_list[i]))
                    # dag.dumpSequenceCounts(False, "noisy_count_" + method + "_" + str(eps_list[i]))

                totalANW_Geocast = 0
                totalATD_FCFS_Geocast = 0
                totalCell_Geocast = 0
                totalPerformedTasks_Geocast = 0
                totalHop_Geocast = 0

                T = len(dag.getAGs())
                # test all tasks for all time instances
                for ti in range(T):
                    if len(dag.getAGs()) == 0:
                        break
                    # free memory of previous instances
                    for l in range(len(all_tasks[j])):
                        if (l + 1) % Params.LOGGING_STEPS == 0:
                            print ">> " + str(l + 1) + " tasks completed"
                        t = all_tasks[j][l]

                        # Geocast
                        q, q_log = geocast(dag.getAGs()[ti], t, p.Eps)
                        no_workers, workers, Cells, no_hops = simple_post_geocast(t, q, q_log)
                        performed, worker, dist_fcfs = performed_tasks(workers, Params.MTD, t, True)
                        if performed:
                            totalPerformedTasks_Geocast += 1
                            totalANW_Geocast += no_workers
                            totalCell_Geocast += Cells
                            totalHop_Geocast += no_hops

                            # if performed:
                            totalATD_FCFS_Geocast += dist_fcfs

                    if len(dag.getAGs()) - 1 == ti:
                        dag.clearMemory()
                        break

                # Geocast
                ANW_Geocast = (totalANW_Geocast + 0.0) / totalPerformedTasks_Geocast
                ATD_FCFS_Geocast = totalATD_FCFS_Geocast / totalPerformedTasks_Geocast
                ASC_Geocast = (totalCell_Geocast + 0.0) / totalPerformedTasks_Geocast
                APPT_Geocast = 100 * float(totalPerformedTasks_Geocast) / (Params.TASK_NO * T)
                HOP_Geocast = float(totalHop_Geocast) / (Params.TASK_NO * T)

                res_cube_anw[i, j, methodList.index(method)] = ANW_Geocast
                res_cube_atd_fcfs[i, j, methodList.index(method)] = ATD_FCFS_Geocast
                res_cube_appt[i, j, methodList.index(method)] = APPT_Geocast
                res_cube_cell[i, j, methodList.index(method)] = ASC_Geocast
                res_cube_hop[i, j, methodList.index(method)] = HOP_Geocast

                gc.collect()
                proc.get_memory_info().rss

    # do not need to varying eps for non-privacy technique!
    for j in range(len(seed_list)):
        totalANW_Knn = 0
        totalATD_Knn_FCFS = 0
        totalPerformedTasks_Knn = 0
        totalHop_Knn = 0

        tasks = all_tasks[j]
        # test all tasks for all time instances
        for ti in range(len(all_workers)):
            for l in range(len(tasks)):
                t = tasks[l]

                # Baseline (no privacy)
                no_workers_knn, performed, dist_knn, dist_knn_FCFS, no_hops, coverage, no_hops2 = geocast_knn(
                    all_workers[ti], t)
                if performed:
                    totalPerformedTasks_Knn += 1
                    totalANW_Knn += no_workers_knn
                    totalATD_Knn_FCFS += dist_knn_FCFS
                    totalHop_Knn += no_hops

        # Baseline
        ANW_Knn = (totalANW_Knn + 0.0) / totalPerformedTasks_Knn
        ATD_FCFS_Knn = totalATD_Knn_FCFS / totalPerformedTasks_Knn
        APPT_Knn = 100 * float(totalPerformedTasks_Knn) / (Params.TASK_NO * T)
        HOP_Knn = float(totalHop_Knn) / (Params.TASK_NO * T)

        res_cube_anw[:, j, len(methodList) - 1] = ANW_Knn
        res_cube_atd_fcfs[:, j, len(methodList) - 1] = ATD_FCFS_Knn
        res_cube_appt[:, j, len(methodList) - 1] = APPT_Knn
        res_cube_cell[:, j, len(methodList) - 1] = 0
        res_cube_hop[:, j, len(methodList) - 1] = HOP_Knn

    res_summary_anw = np.average(res_cube_anw, axis=1)
    np.savetxt(p.resdir + exp_name + '_anw_' + str(EU) + "_" + `Params.TASK_NO`, res_summary_anw, fmt='%.4f\t')
    res_summary_atd = np.average(res_cube_atd_fcfs, axis=1)
    np.savetxt(p.resdir + exp_name + '_atd_fcfs_' + str(EU) + "_" + `Params.TASK_NO`, res_summary_atd, fmt='%.4f\t')
    res_summary_appt = np.average(res_cube_appt, axis=1)
    np.savetxt(p.resdir + exp_name + '_appt_' + str(EU) + "_" + `Params.TASK_NO`, res_summary_appt, fmt='%.4f\t')
    res_summary_cell = np.average(res_cube_cell, axis=1)
    np.savetxt(p.resdir + exp_name + '_cell_' + str(EU) + "_" + `Params.TASK_NO`, res_summary_cell, fmt='%.4f\t')
    res_summary_hop = np.average(res_cube_hop, axis=1)
    np.savetxt(p.resdir + exp_name + '_hop_' + str(EU) + "_" + `Params.TASK_NO`, res_summary_hop, fmt='%.4f\t')




def evalGeocast_Baseline(all_workers, all_tasks):
    """
    Evaluate geocast algorithm in privacy mode and in non-privacy mode
    """
    logging.info("evalGeocast_Baseline")
    exp_name = "Geocast_Baseline"
    methodList = ["Geocast", "Baseline"]

    for ti in range(len(all_workers)):

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

                # test all tasks for all time instances
                tree = Grid_adaptiveM(all_workers[ti], p.Eps, p)
                tree.buildIndex()
                if Params.CONSTRAINT_INFERENCE:
                    tree.adjustConsistency()

                totalANW_Geocast = 0
                totalATD_Geocast, totalATD_FCFS_Geocast = 0, 0
                totalCell_Geocast = 0
                totalCompactness_Geocast = 0
                totalPerformedTasks_Geocast = 0
                totalHop_Geocast = 0
                totalHop2_Geocast = 0
                totalCov_Geocast = 0

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

                # Geocast
                ANW_Geocast = (totalANW_Geocast + 0.0) / totalPerformedTasks_Geocast
                ATD_Geocast = totalATD_Geocast / totalPerformedTasks_Geocast
                ATD_FCFS_Geocast = totalATD_FCFS_Geocast / totalPerformedTasks_Geocast
                ASC_Geocast = (totalCell_Geocast + 0.0) / totalPerformedTasks_Geocast
                CMP_Geocast = totalCompactness_Geocast / totalPerformedTasks_Geocast
                APPT_Geocast = 100 * float(totalPerformedTasks_Geocast) / Params.TASK_NO
                HOP_Geocast = float(totalHop_Geocast) / Params.TASK_NO
                HOP2_Geocast = float(totalHop2_Geocast) / Params.TASK_NO
                COV_Geocast = 100 * float(totalCov_Geocast) / Params.TASK_NO

                res_cube_anw[i, j, 0] = ANW_Geocast
                res_cube_atd[i, j, 0] = ATD_Geocast
                res_cube_atd_fcfs[i, j, 0] = ATD_FCFS_Geocast
                res_cube_appt[i, j, 0] = APPT_Geocast
                res_cube_cell[i, j, 0] = ASC_Geocast
                res_cube_cmp[i, j, 0] = CMP_Geocast
                res_cube_hop[i, j, 0] = HOP_Geocast
                res_cube_hop2[i, j, 0] = HOP2_Geocast
                res_cube_cov[i, j, 0] = COV_Geocast

    for ti in range(len(all_workers)):
        # do not need to varying eps for non-privacy technique!
        for j in range(len(seed_list)):
            totalANW_Knn = 0
            totalATD_Knn, totalATD_Knn_FCFS = 0, 0
            totalPerformedTasks_Knn = 0
            totalHop_Knn = 0
            totalHop2_Knn = 0
            totalCov_Knn = 0

            tasks = all_tasks[j]
            # test all tasks for all time instances
            for l in range(len(tasks)):
                t = tasks[l]

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

            # Baseline
            ANW_Knn = (totalANW_Knn + 0.0) / totalPerformedTasks_Knn
            ATD_Knn = totalATD_Knn / totalPerformedTasks_Knn
            ATD_FCFS_Knn = totalATD_Knn_FCFS / totalPerformedTasks_Knn
            APPT_Knn = 100 * float(totalPerformedTasks_Knn) / Params.TASK_NO
            HOP_Knn = float(totalHop_Knn) / Params.TASK_NO
            HOP2_Knn = float(totalHop2_Knn) / Params.TASK_NO
            COV_Knn = 100 * float(totalCov_Knn) / Params.TASK_NO

            res_cube_anw[:, j, 1] = ANW_Knn
            res_cube_atd[:, j, 1] = ATD_Knn
            res_cube_atd_fcfs[:, j, 1] = ATD_FCFS_Knn
            res_cube_appt[:, j, 1] = APPT_Knn
            res_cube_cell[:, j, 1] = 0
            res_cube_cmp[:, j, 1] = 0
            res_cube_hop[:, j, 1] = HOP_Knn
            res_cube_hop2[:, j, 1] = HOP2_Knn
            res_cube_cov[:, j, 1] = COV_Knn

        res_summary_anw = np.average(res_cube_anw, axis=1)
        np.savetxt(Params.resdir + exp_name + "_" + str(ti) + '_anw_' + `Params.TASK_NO`, res_summary_anw, fmt='%.4f\t')
        res_summary_atd = np.average(res_cube_atd, axis=1)
        np.savetxt(Params.resdir + exp_name + "_" + str(ti) + '_atd_' + `Params.TASK_NO`, res_summary_atd, fmt='%.4f\t')
        res_summary_atd = np.average(res_cube_atd_fcfs, axis=1)
        np.savetxt(Params.resdir + exp_name + "_" + str(ti) + '_atd_fcfs_' + `Params.TASK_NO`, res_summary_atd,
                   fmt='%.4f\t')
        res_summary_appt = np.average(res_cube_appt, axis=1)
        np.savetxt(Params.resdir + exp_name + "_" + str(ti) + '_appt_' + `Params.TASK_NO`, res_summary_appt,
                   fmt='%.4f\t')
        res_summary_cell = np.average(res_cube_cell, axis=1)
        np.savetxt(Params.resdir + exp_name + "_" + str(ti) + '_cell_' + `Params.TASK_NO`, res_summary_cell,
                   fmt='%.4f\t')
        res_summary_cmp = np.average(res_cube_cmp, axis=1)
        np.savetxt(Params.resdir + exp_name + "_" + str(ti) + '_cmp_' + `Params.TASK_NO`, res_summary_cmp, fmt='%.4f\t')
        res_summary_hop = np.average(res_cube_hop, axis=1)
        np.savetxt(Params.resdir + exp_name + "_" + str(ti) + '_hop_' + `Params.TASK_NO`, res_summary_hop, fmt='%.4f\t')
        res_summary_hop2 = np.average(res_cube_hop2, axis=1)
        np.savetxt(Params.resdir + exp_name + "_" + str(ti) + '_hop2_' + `Params.TASK_NO`, res_summary_hop2,
                   fmt='%.4f\t')
        res_summary_cov = np.average(res_cube_cov, axis=1)
        np.savetxt(Params.resdir + exp_name + "_" + str(ti) + '_cov_' + `Params.TASK_NO`, res_summary_cov, fmt='%.4f\t')


def evalDynamic_Test(all_workers, all_tasks):
    logging.basicConfig(level=logging.DEBUG, filename='../log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    p = Params(1000)
    p.Eps = 1
    Params.NDIM, Params.NDATA = all_workers[0].shape[0], all_workers[0].shape[1]
    Params.LOW, Params.HIGH = np.amin(all_workers[0], axis=1), np.amax(all_workers[0], axis=1)

    dag = MultipleDAG_KF(all_workers, p)
    dag.publish()
    dag.applyKalmanFilter()

    tree = spatial.KDTree(all_workers[0].transpose())
    start = time.time()
    for t in all_tasks[0]:
        no_workers_knn, performed, dist_knn, dist_knn_FCFS, no_hops, coverage, no_hops2 = geocast_knn(all_workers[0], t)
    print time.time() - start


# p.resdir + exp_name + '_cmp_' + str(eps) + "_"  + `Params.TASK_NO`, res_summary_cmp, fmt='%.4f\t')
def createGnuData(p, exp_name, var_list):
    """
    Post-processing output files to generate Gnuplot-friendly outcomes
    """
    metrics = ['_anw_', '_appt_', '_atd_fcfs_', '_cell_', '_hop_']

    for metric in metrics:
        out = open(p.resdir + exp_name + metric + `Params.TASK_NO`, 'w')
        for var in var_list:
            fileName = p.resdir + exp_name + metric + str(var) + "_" + `Params.TASK_NO`
            print fileName
            try:
                thisfile = open(fileName, 'r')
            except:
                sys.exit('no input result file!')
            out.write(thisfile.readlines()[0])
            thisfile.close()
        out.close()

        # for metric in metrics:
        # for eps in eps_list:
        #         fileName = p.resdir + exp_name + metric + str(eps) + "_"  + `Params.TASK_NO`
        #         os.remove(fileName)


def exp1():
    logging.basicConfig(level=logging.DEBUG, filename='../log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    param = Params(1000)
    all_workers = readInstances("../../dataset/dynamic/yelp/100/")
    param.NDIM, param.NDATA = all_workers[0].shape[0], all_workers[0].shape[1]
    param.LOW, param.HIGH = np.amin(all_workers[0], axis=1), np.amax(all_workers[0], axis=1)

    print param.NDIM, param.NDATA, param.LOW, param.HIGH
    task_data = read_tasks(param)
    all_tasks = tasks_gen(task_data, param)

    param.debug()

    pool = Pool(processes=len(eps_list))
    params = []
    for eps in eps_list:
        # evalDynamic_Baseline((all_workers, all_tasks, param, eps))
        params.append((all_workers, all_tasks, param, eps))
    pool.map(evalDynamic_Baseline, params)
    pool.join()

    # time.sleep(5)
    #
    # param.resdir = '../../output/yelp/'
    # createGnuData(param,"Dynamic_Baseline", eps_list)

def exp2():
    logging.basicConfig(level=logging.DEBUG, filename='../log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    param = Params(1000) 
    # all_workers = readInstances("../../dataset/dynamic/gowallasf/100/")
    # param.NDIM, param.NDATA = all_workers[0].shape[0], all_workers[0].shape[1]
    # param.LOW, param.HIGH = np.amin(all_workers[0], axis=1), np.amax(all_workers[0], axis=1)
    #
    # print param.NDIM, param.NDATA, param.LOW, param.HIGH
    # task_data = read_tasks(param)
    # all_tasks = tasks_gen(task_data, param)
    #
    # param.debug()
    #
    # pool = Pool(processes=len(first_list))
    # params = []
    # for first in first_list:
    #     # evalDynamic_Baseline_F((all_workers, all_tasks, param, first))
    #     params.append((all_workers, all_tasks, param, first))
    # pool.map(evalDynamic_Baseline_F, params)
    # pool.join()
   # time.sleep(5)
    param.resdir = '../../output/gowalla_sf/'
    createGnuData(param, "Dynamic_Baseline_First", first_list)


def exp3():
    logging.basicConfig(level=logging.DEBUG, filename='../log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    # pool = Pool(processes=len(eps_list))
    # params = []
    # for T in T_list:
    #     all_workers = readInstances("../../dataset/dynamic/yelp/" + str(T) + "/")
    #     param = Params(1000)
    #     param.NDIM, param.NDATA = all_workers[0].shape[0], all_workers[0].shape[1]
    #     param.LOW, param.HIGH = np.amin(all_workers[0], axis=1), np.amax(all_workers[0], axis=1)
    #
    #     task_data = read_tasks(param)
    #     all_tasks = tasks_gen(task_data, param)
    #     params.append((all_workers, all_tasks, param, T))
    # pool.map(evalDynamic_Baseline_T, params)
    # pool.join()

    param = Params(1000)
    param.resdir = '../../output/yelp/'
    createGnuData(param,"Dynamic_Baseline_First", T_list)



def exp4():
    logging.basicConfig(level=logging.DEBUG, filename='../log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    all_workers = readInstances("../../dataset/dynamic/yelp/100/")
    param = Params(1000)
    param.NDIM, param.NDATA = all_workers[0].shape[0], all_workers[0].shape[1]
    param.LOW, param.HIGH = np.amin(all_workers[0], axis=1), np.amax(all_workers[0], axis=1)

    print param.NDIM, param.NDATA, param.LOW, param.HIGH
    task_data = read_tasks(param)
    all_tasks = tasks_gen(task_data, param)

    param.debug()

    pool = Pool(processes=len(first_list))
    params = []
    for EU in EU_list:
        params.append((all_workers, all_tasks, param, EU))
    pool.map(evalDynamic_Baseline_EU, params)
    pool.join()

    #    param.resdir = '../../output/gowalla_sf/'
    createGnuData(param, "Dynamic_Baseline_EU", first_list)

#    param.resdir = '../../output/gowalla_sf/'
#    createGnuData(param,"Dynamic_Baseline", eps_list)
# kdtrees = []
#    for i in range(len(all_workers)):
#        kdtree = spatial.KDTree(all_workers[i].transpose())
#        kdtrees.append(kdtree)

if __name__ == '__main__':
    exp1()

