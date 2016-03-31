__author__ = 'ubriela'

from DynamicAG import DynamicAG


class MultipleDAG(object):
    def __init__(self, data, param, static=False):
        self.instances = len(data)
        self.data = data
        self.param = param

        # sequence of adaptive grids
        self.AGs = []
        self.static = static

    # [None] * self.instances

    def getCountBudget(self):
        """
        :return: budget for each time instance is equal
        """
        if self.static:  # static
            epsPerTime = self.param.Eps
        else:  # dynamic
            epsPerTime = (self.param.Eps + 0.0) / self.instances
        ret = [epsPerTime for i in range(self.instances)]
        return ret

    def getAGs(self):
        return self.AGs

    def clearMemory(self):
        del self.instances
        del self.data
        del self.param
        del self.AGs[:]

    def publish(self):
        """
        publish the first grid, use its structure for the following ones
        :return:
        """
        budgets = self.getCountBudget()

        for i in range(self.instances):
            if not self.static:
                if (i + 1) % 10 == 0:
                    print "Basic: construct AG at time", i, "eps", self.param.Eps
                ag = DynamicAG(self.data[i], budgets[i], self.param)
                ag.buildIndex()
                ag.adjustConsistency()
                self.AGs.append(ag)
            else:
                if i == 0:
                    # init the first grid
                    ag = DynamicAG(self.data[i], budgets[0], self.param)
                    ag.buildIndex()
                    ag.adjustConsistency()
                    self.AGs.append(ag)
                    # self.AGs[0].adjustConsistency()
                else:
                    # the following grid use the partition AND NOISY COUNTS provided by the first grid
                    ag = DynamicAG(self.data[i], budgets[0], self.param, self.AGs[0])
                    ag.buildIndexFromTemplateS()
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