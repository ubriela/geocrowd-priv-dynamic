__author__ = 'ubriela'

from collections import deque
import copy

import numpy as np

from Grid_adaptiveM import Grid_adaptiveM
from Differential import Differential


class DynamicAG(Grid_adaptiveM):
    def __init__(self, data, eps, param, firstGrid=None, use_domain_knowledge=None):
        """
        two levels grid
        """
        self.eps = eps
        self.first = False
        self.DOMAIN_KNOWLEDGE = use_domain_knowledge

        if firstGrid is None:
            # this this first grid --> need to construct
            self.first = True
            Grid_adaptiveM.__init__(self, data, eps, param, self.DOMAIN_KNOWLEDGE)
        else:
            self.param = param
            self.differ = Differential(self.param.Seed)

            # update root
            self.root = copy.deepcopy(firstGrid.root)
            self.root.n_data = data

    def sortData(self):
        """sort data on both dimension"""
        for dim in range(self.data.shape[0]):
            idx = np.argsort(self.data[dim, :], kind='mergesort')
            self.data[:, :] = self.data[:, idx]

    def rangeData(self, query):
        """Get true answer by linear search along each dimension"""
        _data = self.data.copy()
        for dim in range(_data.shape[0]):
            x = np.searchsorted(_data[dim, :], query[0, dim], side='left')
            y = np.searchsorted(_data[dim, :], query[1, dim], side='right')
            _data = _data[:, x:y + 1]
        return _data

    def getCountBudget(self):
        """
        depend on whether this is the first grid or not,
        the budget is used for both levels or for 2nd level grid only
        :return:
        """
        if self.first:
            return super(DynamicAG, self).getCountBudget()
        else:
            return [0, 0, self.eps]


    def freeData(self):
        """
        free data from left nodes
        :return:
        """
        queue = deque()
        queue.append(self.root)

        # ## main loop
        while len(queue) > 0:
            curr = queue.popleft()
            if curr.n_data is not None:
                curr.n_data = None

            if curr.n_isLeaf is False and curr.children is not None:  # if curr is a leaf node
                gran = len(curr.children)
                if gran == 1:
                    if curr.n_data is not None:
                        curr.n_data = None
                    continue  # if the first level cell is leaf node

                # add all nodes to queue
                for x in range(gran):
                    for y in range(gran):
                        node = curr.children[x][y]
                        if node.n_data is not None:
                            node.n_data = None
                        queue.append(node)

        del queue

    def buildIndexFromTemplateRelease(self):
        """build the grid structure and release noisy count"""

        budget_c = self.getCountBudget()  # an array with two elements
        # print budget_c
        self.root.n_count = self.getCount(self.root, 0)  # add noisy count to the root
        queue = deque()
        queue.append(self.root)

        # ## main loop
        while len(queue) > 0:
            curr = queue.popleft()

            if curr.n_isLeaf is True or curr.children is None:  # if curr is a leaf node
                remainingEps = sum(budget_c[curr.n_depth:])
                curr.n_count, curr.eps = self.getCount(curr, remainingEps), remainingEps
            else:
                gran = len(curr.children)
                if gran == 1:
                    remainingEps = sum(budget_c[curr.n_depth:])
                    curr.n_count, curr.eps = self.getCount(curr, remainingEps), remainingEps
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
                        node.n_count = self.getCount(node, budget_c[node.n_depth])
                        if node.n_data is None:
                            node.a_count = 0
                        else:
                            node.a_count = node.n_data.shape[1]
                        node.eps = budget_c[node.n_depth]
                        queue.append(node)

                del n_data_matrix
                curr.n_data = None  # ## do not need the data points coordinates now
        del queue
        del budget_c

    def buildIndexFromTemplate(self):
        """build the grid structure."""
        self.root.n_count = self.getCount(self.root, 0)  # add noisy count to the root
        queue = deque()
        queue.append(self.root)

        # ## main loop
        while len(queue) > 0:
            curr = queue.popleft()

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
                        if node.n_data is None:
                            node.a_count = 0
                        else:
                            node.a_count = node.n_data.shape[1]
                        queue.append(node)

                del n_data_matrix
                curr.n_data = None  # ## do not need the data points coordinates now

        del queue