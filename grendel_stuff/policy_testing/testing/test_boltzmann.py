import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from asla.modules.policies.SegmentPolicies import SegmentBoltzmannPolicy
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from IPython import embed # noqa


def ceil(var, d=0):
    return round(var + 0.5 * 10**(-d), d)


mask = np.load('mask.npy')
Qvalues = np.load('Qvalues.npy')
Qvalues[mask == 1] = np.nan

kappa = 16

policy = SegmentBoltzmannPolicy(n_segments=1, epsilon=0., kappa=kappa)

labels = policy.get_labels(Qvalues, mask)
seg_mask = policy.choose_segment(Qvalues, labels)

mQvalues = ma.masked_array(Qvalues, mask=~seg_mask)
seg_argmax = np.unravel_index(np.nanargmax(mQvalues), Qvalues.shape)

actions = []
for i in range(100):
    distribution = policy.get_boltzmann(Qvalues, mask, seg_mask)
    action, t = policy.draw_action(distribution)
    actions.append((action[0] - seg_argmax[0], action[1] - seg_argmax[1]))

x = [x[0] for x in actions]
y = [x[1] for x in actions]
pos_mask = [np.unique(x), np.unique(y)]
_xrange = max(abs(max(x)), abs(min(x)))
_yrange = max(abs(max(y)), abs(min(y)))
_xyrange = max(_xrange, _yrange) + 0.5
hist, xedges, yedges = np.histogram2d(
        x, y, bins=[int(max(x) - min(x)) + 1, int(max(y) - min(y)) + 1], 
        range=[[min(pos_mask[0]), max(pos_mask[0]) + 1], [min(pos_mask[1]), max(pos_mask[1]) + 1]]
        )
width = depth = 0.75
height = hist.ravel()
height_mask = height != 0.
height = height[height_mask]

xpos, ypos = np.meshgrid(xedges[:-1] - width / 2, yedges[:-1] - depth / 2, indexing='ij')
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
xpos = xpos[height_mask]
ypos = ypos[height_mask]

seg_borders = [
        [min(pos_mask[0]) - .5, max(pos_mask[0]) + .5], 
        [min(pos_mask[1]) - .5, max(pos_mask[1]) + .5]
        ]
Qmasked = np.zeros(hist.shape)
Qmasked[np.where(hist != 0)] = Qvalues[seg_mask]

Qmasked = Qmasked.ravel()
Qmasked = Qmasked[height_mask]

Qtransform = 2 * (1 + Qmasked)**kappa / (1 + np.max(Qmasked))**kappa - 1

probs = np.exp(Qtransform)
probs /= np.sum(probs)
# Plotting

fig = plt.figure(figsize=(18, 5))

Qmz = min(min(Qmasked), 0)
ax = fig.add_subplot(141, projection='3d')
ax.bar3d(xpos, ypos, Qmz, width, depth, Qmasked - Qmz)
ax.plot(
        [seg_borders[0][0], seg_borders[0][0], seg_borders[0][1], seg_borders[0][1], seg_borders[0][0]],
        [seg_borders[1][0], seg_borders[1][1], seg_borders[1][1], seg_borders[1][0], seg_borders[1][0]],
        [Qmz] * 5, c='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Q')
ax.set_xlim(-_xyrange, _xyrange)
ax.set_ylim(-_xyrange, _xyrange)
ax.set_zlim(Qmz, ceil(np.max(Qmasked), 1))
ax.set_title('Q-values')

Qtz = - 1
ax = fig.add_subplot(142, projection='3d')
ax.bar3d(xpos, ypos, Qtz, width, depth, Qtransform - Qtz)
ax.plot(
        [seg_borders[0][0], seg_borders[0][0], seg_borders[0][1], seg_borders[0][1], seg_borders[0][0]],
        [seg_borders[1][0], seg_borders[1][1], seg_borders[1][1], seg_borders[1][0], seg_borders[1][0]],
        [Qtz] * 5, c='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Q')
ax.set_xlim(-_xyrange, _xyrange)
ax.set_ylim(-_xyrange, _xyrange)
ax.set_zlim(Qtz, ceil(np.max(Qtransform), 1))
ax.set_title('Transformed Q-values')

pz = 0
ax = fig.add_subplot(143, projection='3d')
ax.bar3d(xpos, ypos, pz, width, depth, probs - pz)
ax.plot(
        [seg_borders[0][0], seg_borders[0][0], seg_borders[0][1], seg_borders[0][1], seg_borders[0][0]],
        [seg_borders[1][0], seg_borders[1][1], seg_borders[1][1], seg_borders[1][0], seg_borders[1][0]],
        [pz] * 5, c='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('P')
ax.set_xlim(-_xyrange, _xyrange)
ax.set_ylim(-_xyrange, _xyrange)
ax.set_zlim(pz, ceil(np.max(probs), 1))
ax.set_title('Probabilities')

ax = fig.add_subplot(144, projection='3d')
ax.bar3d(xpos, ypos, zpos, width, depth, height)
ax.plot(
        [seg_borders[0][0], seg_borders[0][0], seg_borders[0][1], seg_borders[0][1], seg_borders[0][0]],
        [seg_borders[1][0], seg_borders[1][1], seg_borders[1][1], seg_borders[1][0], seg_borders[1][0]],
        [0] * 5, c='k')
ax.set_xlim(-_xyrange, _xyrange)
ax.set_ylim(-_xyrange, _xyrange)
ax.set_zlim(0, ceil(np.max(height), 0))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Counts')

plt.savefig('boltzmann.png')
plt.show()
