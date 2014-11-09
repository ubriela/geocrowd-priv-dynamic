"""
Demo of a PathPatch object.
"""
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

Path = mpath.Path

path_data = [
    (Path.MOVETO, (1.58, -2.57)),
    (Path.LINETO, (0.35, -1.1)),
    (Path.LINETO, (-1.75, 2.0)),
    (Path.LINETO, (0.375, 2.0)),
    (Path.LINETO, (0.85, 1.15)),
    (Path.LINETO, (2.2, 3.2)),
    (Path.LINETO, (3, 0.05)),
    (Path.LINETO, (2.0, -0.5)),
    (Path.CLOSEPOLY, (1.58, -2.57)),
]

codes, verts = zip(*path_data)
path = mpath.Path(verts, codes)
patch = mpatches.PathPatch(path, facecolor='orange', alpha=0.152343)
ax.add_patch(patch)

# plot control points and connecting lines
x, y = zip(*path.vertices)
line, = ax.plot(x, y, 'b-')

ax.grid()
ax.axis('equal')
plt.show()