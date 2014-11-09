import math
import logging
from collections import deque

import numpy as np

from Log import log
from GenericT import GenericT
from Params import Params
from KalmanFilterPID import KalmanFilterPID


class Grid_adaptiveT(GenericT):
    """
    Budget is allocated for each AG
    """

    def __init__(self, data, param, sampling=False, use_domain_knowledge=False):
        """
        two levels grid
        """
        GenericT.__init__(self, data, param)

        self.firstBudget = self.param.Eps * self.param.structureEps
        self.budgetKF = self.param.Eps * (1 - self.param.structureEps)
        self.sampling = sampling

        # use domain knowledge to specify the second level granularity
        self.DOMAIN_KNOWLEDGE = use_domain_knowledge

        # compute the best grid size at level 1
        if Params.FIX_GRANULARITY:
            self.m = Params.PARTITION_AG[0]
        else:
            self.m = int(max(10, int(0.25 * math.ceil((self.param.NDATA * self.eps / param.c) ** (1.0 / 2)))))
        logging.debug("Grid_adaptiveT: Level 1 size: %d" % self.m)


    def addInstance(self, data):
        """

        :param data:
        :return:
        """
        self.root.n_count = self.getCount(self.root, 0)  # can remove this
        queue = deque()
        queue.append(self.root)

        # ## main loop
        while len(queue) > 0:
            curr = queue.popleft()
            if curr.n_data is None:
                curr.a_count.append(0)
            else:
                curr.a_count.append(curr.n_data.shape[1])

            if curr.n_isLeaf is False and curr.children is not None:  # if curr is a leaf node
                gran = len(curr.children)
                if gran == 1:
                    curr.n_isLeaf, curr.children = True, None
                    continue  # if the first level cell is leaf node

                # get data points in these partitions
                n_data_matrix = np.ndarray(shape=(gran, gran), dtype=object)
                if curr.n_data is not None and curr.n_data.shape[1] >= 1:
                    # iterate all points
                    for p in np.transpose(curr.n_data):
                        x = max(0,
                                min(gran - 1, (p[0] - curr.n_box[0, 0]) * gran / (curr.n_box[1, 0] - curr.n_box[0, 0])))
                        y = max(0,
                                min(gran - 1, (p[1] - curr.n_box[0, 1]) * gran / (curr.n_box[1, 1] - curr.n_box[0, 1])))

                        if n_data_matrix[x][y] is None:
                            n_data_matrix[x][y] = []
                        n_data_matrix[x][y].append(p)

                # add all nodes to queue
                for x in range(gran):
                    for y in range(gran):
                        node = curr.children[x][y]
                        if n_data_matrix[x][y] is None:
                            node.n_data = None
                        else:
                            node.n_data = np.transpose(n_data_matrix[x][y])
                        queue.append(node)

                curr.n_data = None  # ## do not need the data points coordinates now

    def applyKalmanFilter(self):
        """
        Apply KalmanFilter for each 2nd level grid cell
        Note: If Kalman filter is NOT applied, the naive approach is already applied for 2nd time instances.
        :return:
        """

        # init Kalman filter
        kf = KalmanFilterPID(self.param)

        for x1 in range(len(self.root.children)):
            for y1 in range(len(self.root.children)):
                child_l1 = self.root.children[x1][y1]
                if child_l1.n_isLeaf or child_l1.children is None:
                    seq_l1 = child_l1.a_count
                    if self.sampling:
                        seq_l1_filtered = kf.kalmanFilter(seq_l1, self.budgetKF, self.param.samplingRate)
                    else:
                        seq_l1_filtered = kf.kalmanFilter(seq_l1, self.budgetKF)
                    child_l1.l_count = seq_l1_filtered

                else:
                    for x2 in range(len(child_l1.children)):
                        for y2 in range(len(child_l1.children)):
                            seq_l2 = self.root.children[x1][y1].children[x2][y2].a_count
                            if self.sampling:
                                seq_l2_filtered = kf.kalmanFilter(seq_l2, self.budgetKF, self.param.samplingRate)
                            else:
                                seq_l2_filtered = kf.kalmanFilter(seq_l2, self.budgetKF)
                            self.root.children[x1][y1].children[x2][y2].l_count = seq_l2_filtered

    def dumpSequenceCounts(self, actual, filename):
        """
        Dump counts for each left node across multiple time instances
        :param actual: actual count, otherwise noisy count
        :param filename: dump file
        :return:
        """
        sequences = []
        for x1 in range(len(self.root.children)):
            for y1 in range(len(self.root.children)):
                child_l1 = self.root.children[x1][y1]
                if child_l1.n_isLeaf or child_l1.children is None:
                    if actual:
                        sequences.append(child_l1.a_count)
                    else:
                        sequences.append(child_l1.l_count)
                else:
                    for x2 in range(len(child_l1.children)):
                        for y2 in range(len(child_l1.children)):
                            if actual:
                                sequences.append(self.root.children[x1][y1].children[x2][y2].a_count)
                            else:
                                sequences.append(child_l1.self.root.children[x1][y1].children[x2][y2].l_count)

        content = ""
        for s in sequences:
            if not actual:
                content += "\t".join([str(float("%.1f" % n)) for n in s])
            else:
                content += "\t".join([str(n) for n in s])
            content = content + "\n"
        log(filename, content)

        return sequences

    def getCountBudget(self):
        if not self.DOMAIN_KNOWLEDGE:
            return [0, self.firstBudget, 0]
        else:
            return [0, 0, self.eps]

    def getCoordinates(self, curr):
        """
        get coordinates of the point which defines the sub nodes
        return
        @gran : granularity
        @split_arr_x : split points in x coord
        @split_arr_y : split points in y coord
        @n_data_matrix : data points in each cell
        """
        n_box = curr.n_box
        curr_depth = curr.n_depth
        # find the number of partitions
        if curr_depth <= 0:
            parts = self.m
        elif curr_depth == 1:
            if Params.FIX_GRANULARITY:
                self.m2 = Params.PARTITION_AG[1]
            elif not self.DOMAIN_KNOWLEDGE:
                # compute the best grid size at level 2
                N_p = max(0, curr.n_count)  # N_prime
                self.m2 = int(math.sqrt(N_p * self.eps * (1 - Params.PercentGrid) / self.param.c2) + 0.5)
                if Params.CUSTOMIZED_GRANULARITY:
                    self.m2 = int(math.sqrt(N_p * self.eps * (1 - Params.PercentGrid) / Params.c2_c) + 0.5)
                parts = self.m2
            else:
                N_p = curr.a_count  # actual count
                self.m2 = int(math.sqrt(N_p * self.eps / self.param.c2) + 0.5)
                if Params.CUSTOMIZED_GRANULARITY:
                    self.m2 = int(math.sqrt(N_p * self.eps / Params.c2_c) + 0.5)
                parts = self.m2
        # print parts
        if parts <= 1:
            return parts, None, None, None  # leaf node

        split_arr_x = self.getEqualSplit(parts, n_box[0, 0], n_box[1, 0])
        split_arr_y = self.getEqualSplit(parts, n_box[0, 1], n_box[1, 1])

        # get data points in these partitions
        n_data_matrix = np.ndarray(shape=(parts, parts), dtype=object)
        _data = curr.n_data
        if _data is not None and _data.shape[1] >= 1:

            # iterate all points
            for p in np.transpose(_data):
                x = min(parts - 1, (p[0] - n_box[0, 0]) * parts / (n_box[1, 0] - n_box[0, 0]))
                y = min(parts - 1, (p[1] - n_box[0, 1]) * parts / (n_box[1, 1] - n_box[0, 1]))

                # if x < 0:
                # print n_data_matrix.shape, x, y, p, n_box
                if n_data_matrix[x][y] is None:
                    n_data_matrix[x][y] = []
                n_data_matrix[x][y].append(p)

        return parts, split_arr_x, split_arr_y, n_data_matrix

    def adjustConsistency(self):  # used for adaptive grid only
        # similar to htree variants, adaptive grid do not apply constraint inference on root node
        for (_, _), l1_child in np.ndenumerate(self.root.children):
            if not l1_child.n_isLeaf and l1_child.children is not None:
                sum = 0
                for (_, _), l2_child in np.ndenumerate(l1_child.children):  # child1 is a first-level cell
                    sum += l2_child.n_count
                    # print l2_child.a_count, l2_child.n_count
                adjust = (l1_child.n_count - sum + 0.0) / len(self.root.children) ** 2

                for (_, _), l2_child in np.ndenumerate(l1_child.children):
                    l2_child.n_count += adjust