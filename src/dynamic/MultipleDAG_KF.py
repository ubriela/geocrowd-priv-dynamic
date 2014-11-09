__author__ = 'ubriela'

from Log import log

from DynamicAG import DynamicAG
from KalmanFilterPID import KalmanFilterPID
from MultipleDAG import MultipleDAG


class MultipleDAG_KF(MultipleDAG):
    def __init__(self, data, param, sampling=False):

        MultipleDAG.__init__(self, data, param)

        # self.grids = []
        # self.instances = len(data)
        # self.data = data
        # self.param = param

        # sequence of adaptive grids
        # self.AGs = [None] * self.instances

        self.firstBudget = 2 * self.param.Eps * self.param.structureEps
        self.budgetKF = self.param.Eps * (1 - self.param.structureEps)
        self.epsPerTime = self.budgetKF / self.instances
        self.sampling = sampling

        print "budget for kalman filter", self.budgetKF

    def getCountBudget(self):
        """
        budget for the first instance double the budgets of the rest
        :return:
        """
        ret = [self.epsPerTime for i in range(self.instances - 1)]
        ret.insert(0, self.firstBudget)
        return ret

    def publish(self):
        """
        publish the first grid, use its structure for the following ones
        :return:
        """
        budgets = self.getCountBudget()
        for i in range(self.instances):
            if (i + 1) % 10 == 0:
                if self.sampling:
                    print "KFPID: construct AG at time", i, "eps", self.param.Eps
                else:
                    print "KF: construct AG at time", i, "eps", self.param.Eps
            if i == 0:
                # init the first grid
                ag = DynamicAG(self.data[0], budgets[0], self.param)
                ag.buildIndex()
                ag.freeData()
                self.AGs.append(ag)
                # self.AGs[0].adjustConsistency()
            else:
                # the following grid use the partition provided by the first grid
                ag = DynamicAG(self.data[i], budgets[i], self.param, self.AGs[0])
                ag.buildIndexFromTemplate()
                self.AGs.append(ag)

        del budgets

    # print self.AGs[i].checkCorrectness(self.AGs[i].root)


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
        """
        Apply KalmanFilter for each 2nd level grid cell
        Note: If Kalman filter is NOT applied, the naive approach is already applied for 2nd time instances.
        :return:
        """

        # init Kalman filter
        kf = KalmanFilterPID(self.param)

        for x1 in range(len(self.AGs[0].root.children)):
            for y1 in range(len(self.AGs[0].root.children)):
                child_l1 = self.AGs[0].root.children[x1][y1]
                if child_l1.n_isLeaf or child_l1.children is None:
                    seq_l1 = [self.getCountAt(True, i, x1, y1) for i in range(self.instances)]
                    if self.sampling:
                        seq_l1_filtered = kf.kalmanFilter(seq_l1, self.budgetKF, self.param.samplingRate)
                    else:
                        seq_l1_filtered = kf.kalmanFilter(seq_l1, self.budgetKF)
                    [self.updateCountAt(seq_l1_filtered[i], i, x1, y1) for i in range(self.instances)]
                    del seq_l1
                    del seq_l1_filtered

                else:
                    for x2 in range(len(child_l1.children)):
                        for y2 in range(len(child_l1.children)):
                            # child_l2 = child_l1.children[x2][y2]
                            seq_l2 = [self.getCountAt(True, i, x1, y1, x2, y2) for i in range(self.instances)]
                            if self.sampling:
                                seq_l2_filtered = kf.kalmanFilter(seq_l2, self.budgetKF, self.param.samplingRate)
                            else:
                                seq_l2_filtered = kf.kalmanFilter(seq_l2, self.budgetKF)
                            [self.updateCountAt(seq_l2_filtered[i], i, x1, y1, x2, y2) for i in range(self.instances)]

                            del seq_l2
                            del seq_l2_filtered

        del kf

    def dumpSequenceCounts(self, actual, filename):
        """
        Dump counts for each left node across multiple time instances
        :param actual: actual count, otherwise noisy count
        :param filename: dump file
        :return:
        """
        sequences = []
        for x1 in range(len(self.AGs[0].root.children)):
            for y1 in range(len(self.AGs[0].root.children)):
                child_l1 = self.AGs[0].root.children[x1][y1]
                if child_l1.n_isLeaf or child_l1.children is None:
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
        del sequences