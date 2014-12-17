import math
import time

import random
import sys

sys.path.append('../geocast')
sys.path.append('../icde12')

from bitstring import BitArray
from Differential import Differential
from Parser import Parser
from Params import Params


class KalmanFilterPID(Parser):
    """ generated source for class KalmanFilterPID """

    # sampling rate
    def __init__(self, param):
        """
        generated source for method __init__
        """
        Parser.__init__(self)

        self.param = param
        self.differ = Differential(self.param.Seed)

        self.predict = []
        self.interval = None

        # Kalman Filter params
        self.P = 100

        # estimation error covariance (over all time instance)
        self.Q = 1000

        # process noise synthetic data
        self.R = 1000000

        # measurement noise optimal for alpha = 1, synthetic data
        self.K = 0

        # kalman gain
        # PID control params - default
        self.Cp = 0.9  # proportional gain, to keep output proportional to current error
        self.Ci = 0.1  # integral gain, to eliminate offset
        self.Cd = 0.0  # derivative gain, to ensure stability - prevent large error in future

        # fixed internally
        self.theta = 1  # magnitude of changes
        self.xi = 0.2  # gamma (10%)
        self.minIntvl = 1  # make sure the interval is greater than 1

        self.windowPID = 5  # I(integration) window
        self.ratioM = 0.2  # sampling rate

        #
        self.isSampling = False


    def adjustParams(self):
        # adjust params
        if self.ratioM < 0.1:
            self.theta = 20
        if 0.1 <= self.ratioM < 0.2:
            self.theta = 14
        if 0.2 <= self.ratioM < 0.3:
            self.theta = 2
        if 0.3 <= self.ratioM < 0.4:
            self.theta = 0.5
        if 0.4 <= self.ratioM < 0.5:
            self.theta = 0.3
        if 0.5 <= self.ratioM:
            self.theta = 0.1

    # test
    @classmethod
    def main(self, args):
        """ generated source for method main """
        if len(args) < 5:
            print "Usage: python KalmanFilterPID.py input output privacy-budget process-variance Cp(optional) Ci(optional) Cd(optional)"
            sys.exit()

        output = open(args[2], "w")
        budget = eval(args[3])
        Q = float(args[4])
        if budget <= 0 or Q <= 0:
            print "Usage: privacy-budget AND process-variance are positive values"
            sys.exit()

        p = Params(1000)
        kfPID = KalmanFilterPID(p)
        kfPID.setTotalBudget(budget)
        kfPID.setQ(Q)

        kfPID.orig = Parser.getData(args[1])

        kfPID.publish = [None] * len(kfPID.orig)

        # adjust R based on T and alpha
        kfPID.setR(len(kfPID.orig) * len(kfPID.orig) / (0.0 + budget * budget))

        # set optional control gains
        if len(args) >= 6:
            d = args[5]
            if d > 1:
                d = 1
            kfPID.setCp(d)

        if len(args) >= 7:
            d = args[6]
            if d + kfPID.Cp > 1:
                d = 1 - kfPID.Cp
            kfPID.setCi(d)
        else:
            kfPID.setCi(1 - kfPID.Cp)

        if len(args) >= 8:
            d = args[7]
            if d + kfPID.Cp + kfPID.Ci > 1:
                d = 1 - kfPID.Cp - kfPID.Ci
            kfPID.setCd(d)
        else:
            kfPID.setCd(1 - kfPID.Cp - kfPID.Ci)

        # kfPID.adjustParams()

        start = time.time()
        kfPID.publishCounts()
        end = time.time()

        Parser.outputData(output, kfPID.publish)

        print "Method:\tKalman Filter with Adaptive Sampling"
        print "Data Series Length:\t" + str(len(kfPID.orig))
        print "Queries Issued:\t" + str(kfPID.query.count(1))
        print "Privacy Budget Used:\t" + str(kfPID.query.count(1) * kfPID.epsilon)
        print "Average Relative Error:\t" + str(kfPID.getRelError())
        print "Time Used (in second):\t" + str(end - start)

    def kalmanFilter(self, orig, budget, samplingRate=None):
        self.totalBudget = budget
        self.orig = orig
        if samplingRate is not None:
            self.isSampling = True
            self.ratioM = samplingRate
        else:
            self.isSampling = False

        # self.adjustParams()

        self.publish = [None] * len(self.orig)

        # adjust R based on T and alpha
        self.setR(len(self.orig) * len(self.orig) / (0.0 + budget * budget))

        self.publishCounts()

        return self.publish

    def getCount(self, value, epsilon):
        """
        return true count or noisy count of a node, depending on epsilon.
        Note that the noisy count can be negative
        """
        if epsilon < 10 ** (-8):
            return value
        else:
            return value + self.differ.getNoise(1, epsilon)  # sensitivity is 1


    # data publication procedure
    def publishCounts(self):
        """ generated source for method publish """

        self.query = BitArray(len(self.orig))
        self.predict = [None] * len(self.orig)

        # recalculate individual budget based on M
        if (self.isSampling):
            M = int(self.ratioM * (len(self.orig)))  # 0.25 optimal percentile
        else:
            M = len(self.orig)

        if M <= 0:
            M = 1
        self.epsilon = (self.totalBudget + 0.0) / M

        # error = 0
        self.interval = 1
        nextQuery = max(1, self.windowPID) + self.interval - 1

        for i in range(len(self.orig)):
            if i == 0:
                # the first time instance
                self.publish[i] = self.getCount(self.orig[i], self.epsilon)
                self.query[i] = 1
                self.correctKF(i, 0)
            else:
                predct = self.predictKF(i)
                self.predict[i] = predct
                if self.query.count(1) < self.windowPID and self.query.count(1) < M:
                    # i is NOT the sampling point

                    self.publish[i] = self.getCount(self.orig[i], self.epsilon)
                    self.query[i] = 1

                    # update count using observation
                    self.correctKF(i, predct)
                elif i == nextQuery and self.query.count(1) < M:
                    # if i is the sampling point

                    # query
                    self.publish[i] = self.getCount(self.orig[i], self.epsilon)
                    self.query[i] = 1

                    # update count using observation
                    self.correctKF(i, predct)

                    # update freq
                    if (self.isSampling):
                        ratio = self.PID(i)
                        frac = min(20, (ratio - self.xi) / self.xi)
                        deltaI = self.theta * (1 - math.exp(frac))
                        deltaI = int(deltaI) + (random.random() < deltaI - int(deltaI))
                        self.interval += deltaI
                    else:
                        self.interval = 1

                    if self.interval < self.minIntvl:
                        self.interval = self.minIntvl
                    nextQuery += self.interval  # nextQuery is ns in the paper
                else:
                    # --> predict
                    self.publish[i] = predct

                    # del self.orig
                    # del self.predict
                    # del self.query

                    # if self.isPostProcessing:
                    # self.postProcessing()

    # def postProcessing(self):
    # print len(self.samples), self.samples
    # remainedEps = self.totalBudget - len(self.samples) * self.epsilon
    #     self.epsilon = self.epsilon + remainedEps/len(self.samples)
    #
    #     # recompute noisy counts
    #     prev = 0
    #     for i in self.samples:
    #         self.publish[i] = self.getCount(self.orig[i], self.epsilon)
    #         if i > prev + 1:
    #             self.publish[prev + 1 : i] = [self.publish[prev]] * (i - prev - 1)
    #         prev = i

    def setR(self, r):
        """ generated source for method setR """
        self.R = r

    def setQ(self, q):
        """ generated source for method setQ """
        self.Q = q

    def setCp(self, cp):
        """ generated source for method setCp """
        self.Cp = cp

    def setCi(self, ci):
        """ generated source for method setCi """
        self.Ci = ci

    def setCd(self, cd):
        """ generated source for method setCd """
        self.Cd = cd

    # prediction step
    def predictKF(self, curr):
        """ generated source for method predictKF """
        # predict using Kalman Filter
        lastValue = self.getLastQuery(curr)

        # project estimation error
        self.P += self.Q  # Q is gaussian noise
        return lastValue

    # correction step
    def correctKF(self, curr, predict):
        """ generated source for method correctKF """
        self.K = (self.P + 0.0) / (self.P + self.R)
        correct = predict + self.K * (self.publish[curr] - predict)

        # publish[curr] = Math.max((int) correct, 0)
        if curr > 0:
            # only correct from 2nd values
            self.publish[curr] = correct

        # print correct, "\t", self.publish[curr], self.K, self.P

        # update estimation error variance
        self.P *= (1 - self.K)

    def getLastQuery(self, curr):
        """ generated source for method getLastQuery """
        for i in reversed(range(curr)):
            if self.query[i]:
                break
        return self.publish[i]

    # adaptive sampling - return feedback error
    def PID(self, curr):
        """ generated source for method PID """
        sum = 0
        lastValue = 0
        change = 0
        timeDiff = 0
        next = curr
        for j in reversed(range(self.windowPID - 1)):
            index = next
            while index >= 0:
                if self.query[index]:
                    next = index - 1  # the last nextQuery
                    break
                index -= 1
            if j == self.windowPID - 1:
                lastValue = abs(self.publish[index] - self.predict[index]) / (0.0 + max(self.publish[index], 1))
                change = abs(self.publish[index] - self.predict[index]) / (0.0 + max(self.publish[index], 1))
                timeDiff = index
            if j == self.windowPID - 2:
                change -= abs(self.publish[index] - self.predict[index]) / (0.0 + max(self.publish[index], 1))
                timeDiff -= index
            sum += (abs(self.publish[index] - self.predict[index]) / (0.0 + max(self.publish[index], 1)))

        ratio = self.Cp * lastValue + self.Ci * sum + self.Cd * change / (0.0 + timeDiff)
        return ratio


if __name__ == '__main__':
    import sys

    KalmanFilterPID.main(sys.argv)

