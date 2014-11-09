"""
Contains all experimental methods
"""

import time
import logging

from pylab import *
import numpy as np


sys.path.append('../minball')
sys.path.append('../common')
sys.path.append('../geocast')
sys.path.append('../grid')
sys.path.append('../htree')
sys.path.append('../icde12')
sys.path.append('../dynamic')
sys.path.append('../exp')

from Params import Params
from PSDExp import data_readin

# from Grid_adaptive import Grid_adaptive
from Grid_adaptiveM import Grid_adaptiveM
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def getPathData(data, param):
    path_data = []
    p = param

    tree = Grid_adaptiveM(data, p.Eps, p)
    tree.buildIndex()
    tree.adjustConsistency()

    left_boxes = []
    for (_, _), l1_child in np.ndenumerate(tree.root.children):
        if not l1_child.n_isLeaf and l1_child.children is not None:
            for (_, _), l2_child in np.ndenumerate(l1_child.children):  # child1 is a first-level cell
                left_boxes.append((l2_child.n_box, l2_child.n_count))
        left_boxes.append((l1_child.n_box, l1_child.n_count))

    for data in left_boxes:
        # [[x_min,y_min],[x_max,y_max]]
        path = []
        box = data[0]
        # (x_min, y_min) --> (x_min, y_max) --> (x_max, y_max) --> (x_max, y_min) --> (x_min, y_min)
        path.append((mpath.Path.MOVETO, (box[0][0], box[0][1])))
        path.append((mpath.Path.LINETO, (box[0][0], box[1][1])))
        path.append((mpath.Path.LINETO, (box[1][0], box[1][1])))
        path.append((mpath.Path.LINETO, (box[1][0], box[0][1])))
        path.append((mpath.Path.CLOSEPOLY, (box[0][0], box[0][1])))

        path_data.append((path, data[1]))

    return path_data


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename='../log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")


    # eps_list = [0.1, 0.4, 0.7, 1.0]
    # dataset_list = ['yelp', 'foursquare', 'gowallasf', 'gowallala']

    eps_list = [1.0]
    dataset_list = ['yelp']

    for dataset in dataset_list:
        for eps in eps_list:
            param = Params(1000)
            all_workers = data_readin(param)
            param.NDIM, param.NDATA = all_workers.shape[0], all_workers.shape[1]
            param.LOW, param.HIGH = np.amin(all_workers, axis=1), np.amax(all_workers, axis=1)

            param.DATASET = dataset
            param.select_dataset()
            param.Eps = eps
            param.debug()

            path_data = getPathData(all_workers, param)

            # max_count = 0
            # for data in path_data:
            # if data[1] > max_count:
            #         max_count = data[1]

            fig, ax = plt.subplots()
            for data in path_data:
                path = data[0]
                codes, verts = zip(*path)
                path = mpath.Path(verts, codes)
                weight = min(1, (data[1] + 0.0) / 100)
                patch = mpatches.PathPatch(path, facecolor='orange', alpha=weight)
                ax.add_patch(patch)

                # plot control points and connecting lines
                x, y = zip(*path.vertices)
                line, = ax.plot(x, y, 'k-', linewidth=0.1)

            ax.grid()
            ax.axis('equal')
            savefig('../../dataset/graph/grid/' + param.DATASET + '_' + str(param.Eps) + '.eps', format='eps', dpi=1000)