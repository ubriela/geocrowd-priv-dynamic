import logging
from collections import deque

import numpy as np

from NodeT import NodeT
from Params import Params
from Differential import Differential


class GenericT(object):
    """
    Generic data structure, used for grid
    """

    def __init__(self, data, param):
        self.param = param
        self.differ = Differential(self.param.Seed)

        # initialize the root
        self.root = NodeT()
        # self.children = [] # all level 2 grids
        self.root.n_data = data
        self.root.n_box = np.array([param.LOW, param.HIGH])

    def getEqualSplit(self, partitions, min, max):
        """return equal split points, including both ends"""
        if min > max:
            logging.debug("getEqualSplit: Error: min > max")
        if partitions <= 1:
            return [min, max]
        return [min + (max - min) * i / partitions for i in range(partitions + 1)]

    def getCountBudget(self):
        """return noisy count budget for different levels of the indices"""
        raise NotImplementedError

    def getCoordinates(self, curr):
        """return the split dimension, the split points and the data points in each subnodes"""
        raise NotImplementedError

    def getCount(self, curr, epsilon):
        """
        return true count or noisy count of a node, depending on epsilon.
        Note that the noisy count can be negative
        """
        if curr.n_data is None:
            count = 0
        else:
            count = curr.n_data.shape[1]

        if epsilon < 10 ** (-8):
            return count
        else:
            return count + self.differ.getNoise(1, epsilon)


    def intersect(self, hrect, query):
        """
        checks if the hyper-rectangle intersects with the
        hyper-rectangle defined by the query in every dimension
        """
        bool_m1 = query[0, :] >= hrect[1, :]
        bool_m2 = query[1, :] <= hrect[0, :]
        bool_m = np.logical_or(bool_m1, bool_m2)
        if np.any(bool_m):
            return False
        else:
            return True

    def testLeaf(self, curr):
        """ test whether a node should be a leaf node """
        if (curr.n_depth == Params.maxHeightAdaptiveGrid) or \
                (curr.n_data is None or curr.n_data.shape[1] == 0) or \
                (curr.n_count <= self.param.minPartSize):
            return True
        return False

    def buildIndex(self):
        """build the grid structure."""
        budget_c = self.getCountBudget()  # an array with two elements
        # print budget_c
        self.root.n_count = self.getCount(self.root, 0)  # add noisy count to the root
        queue = deque()
        queue.append(self.root)
        # ## main loop
        while len(queue) > 0:
            curr = queue.popleft()
            if curr.n_data is None:
                curr.a_count.append(0)
            else:
                curr.a_count.append(curr.n_data.shape[1])

            if self.testLeaf(curr) is True:  # if curr is a leaf node
                remainingEps = sum(budget_c[curr.n_depth:])
                curr.n_count, curr.eps, curr.n_isLeaf = self.getCount(curr, remainingEps), remainingEps, True
                curr.l_count.append(curr.n_count)
            else:  # curr needs to split --> find splitting granularity
                gran, split_arr_x, split_arr_y, n_data_matrix = self.getCoordinates(curr)
                if gran == 1:
                    remainingEps = sum(budget_c[curr.n_depth:])
                    curr.n_count, curr.eps, curr.n_isLeaf = self.getCount(curr, remainingEps), remainingEps, True
                    curr.children = None
                    curr.l_count.append(curr.n_count)
                    continue  # if the first level cell is leaf node

                # add all nodes to queue
                for x in range(gran):
                    for y in range(gran):
                        node = NodeT()
                        node.n_box = np.array(
                            [[split_arr_x[x], split_arr_y[y]], [split_arr_x[x + 1], split_arr_y[y + 1]]])
                        node.index, node.parent, node.n_depth = x * gran + y, curr, curr.n_depth + 1
                        if n_data_matrix[x][y] is None:
                            node.n_data = None
                        else:
                            node.n_data = np.transpose(n_data_matrix[x][y])
                        node.n_count = self.getCount(node, budget_c[node.n_depth])
                        node.eps = budget_c[node.n_depth]
                        if node.n_depth == 2:
                            node.n_isLeaf = True
                        if curr.children is None:
                            curr.children = np.ndarray(shape=(gran, gran), dtype=NodeT)
                        curr.children[x][y] = node
                        queue.append(node)

                curr.n_data = None  # ## do not need the data points coordinates now
                # end of while


    # canonical range query does apply
    def rangeCount(self, query):
        """
        Query answering function. Find the number of data points within a query rectangle.
        This function assume that the tree is constructed with noisy count for every node
        """
        queue = deque()
        queue.append(self.root)
        count = 0.0
        while len(queue) > 0:
            curr = queue.popleft()
            _box = curr.n_box
            if curr.n_isLeaf is True:
                frac = 1
                if self.intersect(_box, query):
                    for i in range(_box.shape[1]):
                        if _box[1, i] == _box[0, i] or Params.WorstCase == True:
                            frac *= 1
                        else:
                            frac *= (min(query[1, i], _box[1, i]) - max(query[0, i], _box[0, i])) / (
                                _box[1, i] - _box[0, i])
                    count += curr.n_count * frac
            else:  # if not leaf

                for (_, _), node in np.ndenumerate(curr.children):
                    bool_matrix = np.zeros((2, query.shape[1]))
                    bool_matrix[0, :] = query[0, :] <= _box[0, :]
                    bool_matrix[1, :] = query[1, :] >= _box[1, :]

                    if np.all(bool_matrix):  # if query range contains node range
                        count += node.n_count
                    elif self.intersect(_box, query):
                        queue.append(node)
        return float(count)


    def leafCover(self, loc):
        """
        find a leaf node that cover the location
        """
        gran_1st = len(self.root.children)
        x1 = min(gran_1st - 1,
                 (loc[0] - self.root.n_box[0, 0]) * gran_1st / (self.root.n_box[1, 0] - self.root.n_box[0, 0]))
        y1 = min(gran_1st - 1,
                 (loc[1] - self.root.n_box[0, 1]) * gran_1st / (self.root.n_box[1, 1] - self.root.n_box[0, 1]))

        node_1st = self.root.children[x1][y1]
        """
        Note that there are cases when the actual count of first level cell is zero but the noisy count is > 0, 
        thus the cell may be splited into a number of empty cells
        """
        if node_1st.n_isLeaf or node_1st.children is None:
            return node_1st
        else:
            gran_2st = len(node_1st.children)
            x2 = min(gran_2st - 1,
                     (loc[0] - node_1st.n_box[0, 0]) * gran_2st / (node_1st.n_box[1, 0] - node_1st.n_box[0, 0]))
            y2 = min(gran_2st - 1,
                     (loc[1] - node_1st.n_box[0, 1]) * gran_2st / (node_1st.n_box[1, 1] - node_1st.n_box[0, 1]))
            return node_1st.children[x2][y2]


    def checkCorrectness(self, node, nodePoints=None):
        """
        Total number of data points of all leaf nodes should equal to the total data points
        only check the FIRST time instance
        """
        totalPoints = 0
        if node is None:
            return 0
        if (node.n_isLeaf and node.n_data is not None) or node.children is None:
            return node.a_count[0]

        for (_, _), child in np.ndenumerate(node.children):
            totalPoints += self.checkCorrectness(child)

        if nodePoints is None:
            return totalPoints

        if totalPoints == nodePoints:
            return True
        return False