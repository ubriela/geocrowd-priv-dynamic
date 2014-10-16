import math
import logging

import numpy as np

from Generic2 import Generic2
from Params import Params


class Grid_adaptive2(Generic2):
    """
    Budget is allocated for each AG
    """

    def __init__(self, data, eps, param, use_domain_knowledge=None):
        """
        two levels grid
        """
        Generic2.__init__(self, data, param)

        self.eps = eps

        # use domain knowledge to specify the second level granularity
        self.DOMAIN_KNOWLEDGE = use_domain_knowledge

        # compute the best grid size at level 1
        if Params.FIX_GRANULARITY:
            self.m = Params.PARTITION_AG[0]
        else:
            self.m = int(max(10, int(0.25 * math.ceil((self.param.NDATA * self.eps / param.c) ** (1.0 / 2)))))
        logging.debug("Grid_adaptive2: Level 1 size: %d" % self.m)

    def getCountBudget(self):
        if not self.DOMAIN_KNOWLEDGE:
            count_eps_1 = self.eps * Params.PercentGrid
            count_eps_2 = self.eps * (1 - Params.PercentGrid)
            return [0, count_eps_1, count_eps_2]
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

                if n_data_matrix[x][y] is None:
                    n_data_matrix[x][y] = []
                n_data_matrix[x][y].append(p)

        return parts, split_arr_x, split_arr_y, n_data_matrix

    def adjustConsistency(self):  # used for adaptive grid only
        # similar to htree variants, adaptive grid do not apply constraint inference on root node
        for (_, _), l1_child in np.ndenumerate(self.root.children):
            if not l1_child.n_isLeaf:
                sum = 0
                for (_, _), l2_child in np.ndenumerate(l1_child.children):  # child1 is a first-level cell
                    sum += l2_child.n_count
                    # print l2_child.a_count, l2_child.n_count
                adjust = (l1_child.n_count - sum + 0.0) / len(self.root.children) ** 2

                for (_, _), l2_child in np.ndenumerate(l1_child.children):
                    l2_child.n_count += adjust