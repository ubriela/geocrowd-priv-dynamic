__author__ = 'ubriela'

from Log import log

from DynamicAG import DynamicAG
from KalmanFilterPID import KalmanFilterPID


class MultipleDAG(object):
    def __init__(self, data, param):
        self.grids = []
        self.instances = len(data)
        self.data = data
        self.param = param

        # sequence of adaptive grids
        self.AGs = [None] * self.instances

        self.epsPerSeq = (self.param.Eps + 0.0) / (self.instances + 1)
        self.budgetKF = (self.instances - 1.0) * self.param.Eps / (self.instances + 1)

    def getCountBudget(self):
        """
        budget for the first instance double the budgets of the rest
        :return:
        """
        firstBudget = 2 * self.epsPerSeq
        ret = [self.epsPerSeq for i in range(self.instances - 1)]
        ret.insert(0, firstBudget)
        return ret

    def getAGs(self):
        return self.AGs

    def publishSimple(self):
        """
        publish the first grid, use its structure for the following ones
        :return:
        """
        for i in range(self.instances):
            print "construct AG at time", i
            eps_i = self.getCountBudget()[i]
            if i == 0:
                # init the first grid
                self.AGs[0] = DynamicAG(self.data[0], eps_i, self.param)
                self.AGs[0].buildIndex()
                # self.AGs[0].adjustConsistency()
            else:
                # the following grid use the partition provided by the first grid
                self.AGs[i] = DynamicAG(self.data[i], eps_i, self.param, self.AGs[0])
                self.AGs[i].buildIndexFromTemplate()
                # self.AGs[i].adjustConsistency()

            print self.AGs[i].checkCorrectness(self.AGs[i].root)


    def getCountAt(self, actual, i, x1, y1, x2=None, y2=None):
        """
        Get cell count at a particular time instance
        :param i: instance number
        :param x1: x coord at level 1
        :param y1: y coord at level 1
        :param x2: x coord at level 2
        :param y2: y coord at level 2
        :return:
        """
        if actual:
            if x2 is None:
                return self.AGs[i].root.children[x1][y1].a_count
            else:
                return self.AGs[i].root.children[x1][y1].children[x2][y2].a_count
        else:
            if x2 is None:
                return self.AGs[i].root.children[x1][y1].n_count
            else:
                return self.AGs[i].root.children[x1][y1].children[x2][y2].n_count

    def updateCountAt(self, n_count, i, x1, y1, x2=None, y2=None):
        """
        Update noisy count
        :param i: instance number
        :param x1: x coord at level 1
        :param y1: y coord at level 1
        :param x2: x coord at level 2
        :param y2: y coord at level 2
        :return:
        """
        if x2 is None:
            self.AGs[i].root.children[x1][y1].n_count = n_count
        else:
            self.AGs[i].root.children[x1][y1].children[x2][y2].n_count = n_count


    def applyKalmanFilter(self):
        root = self.AGs[0].root

        # init Kalman filter
        kf = KalmanFilterPID(self.param)

        for x1 in range(len(root.children)):
            for y1 in range(len(root.children)):
                child_l1 = root.children[x1][y1]
                if child_l1.n_isLeaf or child_l1.children is None:
                    seq_l1 = [self.getCountAt(True, i, x1, y1) for i in range(self.instances)]
                    seq_l1_filtered = kf.kalmanFilter(seq_l1, self.budgetKF)
                    [self.updateCountAt(seq_l1_filtered[i], i, x1, y1) for i in range(self.instances)]

                else:
                    for x2 in range(len(child_l1.children)):
                        for y2 in range(len(child_l1.children)):
                            # child_l2 = child_l1.children[x2][y2]
                            seq_l2 = [self.getCountAt(True, i, x1, y1, x2, y2) for i in range(self.instances)]
                            seq_l2_filtered = kf.kalmanFilter(seq_l2, self.budgetKF)
                            [self.updateCountAt(seq_l2_filtered[i], i, x1, y1, x2, y2) for i in range(self.instances)]


    def dumpSequenceCounts(self, actual, filename):
        """
        Dump counts for each left node across multiple time instances
        :param actual: actual count, otherwise noisy count
        :param filename: dump file
        :return:
        """
        root = self.AGs[0].root
        sequences = []
        for x1 in range(len(root.children)):
            for y1 in range(len(root.children)):
                child_l1 = root.children[x1][y1]
                if child_l1.n_isLeaf:
                    sequence_l1 = [self.getCountAt(actual, i, x1, y1) for i in range(self.instances)]
                    sequences.append(sequence_l1)
                else:
                    for x2 in range(len(child_l1.children)):
                        for y2 in range(len(child_l1.children)):
                            # child_l2 = child_l1.children[x2][y2]
                            sequence_l2 = [self.getCountAt(actual, i, x1, y1, x2, y2) for i in range(self.instances)]
                            sequences.append(sequence_l2)

        content = ""
        for s in sequences:
            if not actual:
                content += "\t".join([str(float("%.1f" % n)) for n in s])
            else:
                content += "\t".join([str(n) for n in s])
            content = content + "\n"
        log(filename, content)

        return sequences