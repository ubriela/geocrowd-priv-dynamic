class Parser(object):
    """ generated source for class Experiment """

    def __init__(self):
        self.orig = []
        self.publish = []
        self.query = None
        self.epsilon = 0
        self.totalBudget = 1.0


    @classmethod
    def getData(self, file_):
        """ generated source for method getData """
        # data file: one value per line
        data = []
        for line in open(file_, "r"):
            striped = line.strip()
            if striped != "":
                data.append(float(striped))
        return data


    @classmethod
    def outputData(self, output, data):
        for i in data:
            output.write(str(i) + "\n")

        output.close()

    def setTotalBudget(self, eps):
        """ generated source for method setTotalBudget """
        self.totalBudget = eps

    def publish(self):
        """ generated source for method publish """

    def distance(self, a, b):
        """ generated source for method distance """
        return abs((a - b + 0.0) / b)

    def getRelError(self):
        """ generated source for method getRelError """
        error = 0
        for i in range(len(self.publish)):
            if self.orig[i] < 0:
                error += self.distance(self.publish[i], self.orig[i])
            elif self.orig[i] == 0:
                error += self.distance(self.publish[i], 1)
            else:
                error += self.distance(max(self.publish[i], 0), self.orig[i])
        return error / len(self.publish)

