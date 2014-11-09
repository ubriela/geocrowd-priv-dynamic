class NodeT(object):
    """
    used for adaptive grid
    NodeT is different than NodeM, actual count and noisy count are two lists of time-series counts
    """

    def __init__(self):
        self.n_data = None  # list of data points
        self.n_box = None  # 2x2 matrix [[x_min,y_min],[x_max,y_max]]
        self.children = None  # matrix of its children
        self.n_count = 0  # noisy count of this node
        self.l_count = []  # publish noisy count of leave node
        self.a_count = []  # actual counts
        self.n_depth = 0
        self.n_budget = 0  # represented by height of the tree
        self.n_isLeaf = False
        self.index = None  # the order of the node in the parent's children
        self.eps = None
        self.neighbor = None

    def debug(self):
        print "Actual count " + str(len(self.n_data))
        print "Noisy count " + str(self.n_count)
        print "Depth " + str(self.n_depth)
        print "Leaf? " + str(self.n_isLeaf)
        print "Boundary " + str(self.n_box)
        print "Index " + str(self.index)