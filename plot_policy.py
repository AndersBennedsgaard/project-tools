from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, PatchCollection
from matplotlib.patches import Rectangle, ConnectionPatch
from asla.modules.policies.ImageSegmentation import watershed
from asla.utils.plot_segments import plot_segments
from lib.useful import remove_ticks


class Faces:
    def __init__(self, tri, sig_dig=12, method="convexhull"):
        self.method = method
        self.tri = np.around(np.array(tri), sig_dig)
        self.grpinx = list(range(len(tri)))
        norms = np.around([self.norm(s) for s in self.tri], sig_dig)
        _, self.inv = np.unique(norms, return_inverse=True, axis=0)

    def norm(self, sq):
        cr = np.cross(sq[2] - sq[0], sq[1] - sq[0])
        return np.abs(cr / np.linalg.norm(cr))

    def isneighbor(self, tr1, tr2):
        a = np.concatenate((tr1, tr2), axis=0)
        return len(a) == len(np.unique(a, axis=0)) + 2

    def order(self, v):
        if len(v) <= 3:
            return v
        v = np.unique(v, axis=0)
        n = self.norm(v[:3])
        y = np.cross(n, v[1] - v[0])
        y = y / np.linalg.norm(y)
        c = np.dot(v, np.c_[v[1] - v[0], y])
        if self.method == "convexhull":
            h = ConvexHull(c)
            return v[h.vertices]
        else:
            mean = np.mean(c, axis=0)
            d = c - mean
            s = np.arctan2(d[:, 0], d[:, 1])
            return v[np.argsort(s)]

    def simplify(self):
        for i, tri1 in enumerate(self.tri):
            for j, tri2 in enumerate(self.tri):
                if j > i:
                    if self.isneighbor(tri1, tri2) and self.inv[i] == self.inv[j]:
                        self.grpinx[j] = self.grpinx[i]
        groups = []
        for i in np.unique(self.grpinx):
            u = self.tri[self.grpinx == i]
            u = np.concatenate([d for d in u])
            u = self.order(u)
            groups.append(u)
        return groups

    def order_along_axis(self, faces, axis):
        midpoints = np.array([f.mean(axis=0) for f in faces])
        s = np.dot(np.array(axis), midpoints.T)
        return np.argsort(s)

    def remove_last_n(self, faces, order, n=1):
        return np.array(faces)[order][::-1][n:][::-1]


def rotate_3d(array, yaw, pitch, roll):
    yaw = yaw * np. pi / 180
    pitch = pitch * np. pi / 180
    roll = roll * np. pi / 180
    yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    rot_mat = yaw @ pitch @ roll
    return (rot_mat @ array.T).T


# x = np.array(
#     [0.16257299, -0.370805, -1.09232295, 1.62570095, -1.62570095, 1.09232295, 0.370805, -0.16257299]
# )
# y = np.array(
#     [-1.71022499, -0.81153202, -0.52910602, -0.36958599, 0.369587, 0.52910602, 0.81153202, 1.71022499]
# )
# z = np.array(
#     [0.22068501, -1.48456001, 1.23566902, 0.469576, -0.469576, -1.23566902, 1.48456001, -0.22068501]
# )

# verts = np.c_[x, y, z]

# verts = np.array([
#     [0, 0, 0],
#     [-2, 0, 0],
#     [-0.2, 1, -0.2],
#     [-2.2, 1, -0.2],
#     [0, 0, -1],
#     [-2, 0, -1],
#     [-0.2, 1, -1.2],
#     [-2.2, 1, -1.2],
# ])

# hull = ConvexHull(verts)
# simplices = hull.simplices

# org_triangles = [verts[s] for s in simplices]
# f = Faces(org_triangles, sig_dig=4)
# g = f.simplify()
# order = f.order_along_axis(g, [0, 1, 0])
# g = f.remove_last_n(g, order, 3)

# # Reduce dimension, ommit y axis:
# g2D = g[:, :, [0, 2]]

# fig = plt.figure(figsize=(8, 3))
# ax = fig.add_subplot(111)

# colors = np.random.rand(len(g), 3)

# pc2 = PolyCollection(
#     g2D, facecolors=colors, edgecolor="k", alpha=0.9
# )
# ax.add_collection(pc2)
# ax.autoscale()
# ax.set_aspect("equal")

# plt.show()


def add_inset(axes, xanchor, yanchor, width=1 / 3, height=1 / 3, axis=False, box=False, limited=True, **kwargs):
    ax = axes.inset_axes((xanchor, yanchor, width, height), **kwargs)
    if limited:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    if not limited:
        ax.set_aspect('equal', 'box')
    if box:
        remove_ticks(ax)
        axis = True
    if not axis:
        ax.axis('off')
    return ax


def get_positions(radius, angle=55, displacement=(0, 0)):
    angle *= np.pi / 180
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return x + displacement[0], y + displacement[1]


def get_cdots(start, angle=55):
    cdots = np.array([get_positions(0.05 * j, angle=angle, displacement=start) for j in range(3)])
    return cdots


def shrink_array(arr):
    assert arr.ndim == 2
    while np.isnan(arr[:, 0]).all():
        arr = arr[:, 1:]
    while np.isnan(arr[0, :]).all():
        arr = arr[1:, :]
    while np.isnan(arr[:, -1]).all():
        arr = arr[:, :-1]
    while np.isnan(arr[-1, :]).all():
        arr = arr[:-1, :]
    return arr


N_layers = 5

Qvalues = np.zeros((20, 30, N_layers))
for i in range(N_layers):
    N = 50
    for n in range(N):
        x, y = np.meshgrid(
            np.arange(Qvalues.shape[0]), np.arange(Qvalues.shape[1])  # pylint: disable=E1136
        )
        center = (np.random.randint(Qvalues.shape[0]), np.random.randint(Qvalues.shape[1]))  # pylint: disable=E1136
        Qvalues[:, :, i] += np.exp(-(x - center[0])**2 / 2 / 1 - (y - center[1])**2 / 2 / 1).T
Qvalues = Qvalues / np.max(Qvalues)
labels = np.zeros(Qvalues.shape, dtype=int)
for i in range(N_layers):
    labels[:, :, i] = watershed(Qvalues[:, :, i])

cdots_settings = dict(
    c='k', s=2
)
pcolor_settings = dict(
    vmin=-1, vmax=1, edgecolors='k', linewidth=.1, cmap='Blues'
)
plot_segments_settings = dict(
    correct=False, colors='k', linestyle='-', linewidths=0.5,
)
text_settings = dict(
    fontsize=12, horizontalalignment='center',
)
ylabel_settings = dict(
    fontsize=10, ha='right', va='center', rotation=0
)
xlabel_settings = ylabel_settings.copy()
xlabel_settings['ha'] = 'center'

##########
# Figure #
##########

figwidth, figheight = 15, 12
fig = plt.figure(figsize=(15, 12))
ax = fig.gca()
ax.axis('off')

###############
# All Qvalues #
###############

axin1 = add_inset(ax, 0, 0.66)

Q_box = Rectangle(((1 - 0.7) / 2, (1 - 0.7) / 2), 0.7, 0.7, fill=True, edgecolor='k')
axin1.add_patch(Q_box)
axin1.text(0.5, 0.5, r'$Q(Z_i)$', **text_settings)

############
# Per atom #
############

axin2_x, axin2_y = 1 / 3, 2 / 3
axin2 = add_inset(ax, axin2_x, axin2_y)
radius = 0.08

width, height = 1 / 3, 1 / 3

patches = []
for i in range(3, 0, -1):
    patches.append(Rectangle((get_positions(i * radius)), width, height, zorder=4 - i))

patches.append(Rectangle(get_positions(9 * radius), width, height, zorder=1))
axin2.add_collection(PatchCollection(patches, edgecolors='k'))

# Text

settings = text_settings.copy()
settings['va'] = 'top'
for i, r in enumerate([1, 2, 3, 9]):
    x, y = get_positions(r * radius)
    x += width * 1.2
    if i == 3:
        axin2.text(x, y, r'$Z_N$', **text_settings)
    else:
        axin2.text(x, y, rf'$Z_{{{i + 1}}}$', **text_settings)

# Cdots

cdots = get_cdots(get_positions(8.5 * radius))
axin2.scatter(cdots[:, 0], cdots[:, 1], zorder=6, **cdots_settings)

##############
# Per height #
##############

axin3_x, axin3_y = 2 / 3, 2 / 3
axin3 = add_inset(ax, axin3_x, axin3_y)

width, height = 1 / 2, 1 / 2

radius = 0.15

anchor = get_positions(2 * radius)
axin3_1 = add_inset(axin3, anchor[0], anchor[1], width=width, height=height, limited=False, zorder=3)
axin3_1.pcolor(Qvalues[:, :, 2], **pcolor_settings)
for label in np.unique(labels[:, :, 2]):
    plot_segments((labels[:, :, 2] == label).T, ax=axin3_1, zorder=3, **plot_segments_settings)

anchor = get_positions(1 * radius)
axin3_2 = add_inset(axin3, anchor[0], anchor[1], width=width, height=height, limited=False, zorder=4)
axin3_2.pcolor(Qvalues[:, :, 1], **pcolor_settings)
for label in np.unique(labels[:, :, 1]):
    plot_segments((labels[:, :, 1] == label).T, ax=axin3_2, zorder=4, **plot_segments_settings)

axin3_3 = add_inset(axin3, 0, 0, width=width, height=height, limited=False, zorder=5)
axin3_3.pcolor(Qvalues[:, :, 0], **pcolor_settings)
for label in np.unique(labels[:, :, 0]):
    plot_segments((labels[:, :, 0] == label).T, ax=axin3_3, zorder=5, **plot_segments_settings)

cdots = get_cdots(get_positions(6.5 * radius))
axin3.scatter(cdots[:, 0], cdots[:, 1], zorder=6, **cdots_settings)

axin3.annotate(
    r'$Z_1$', (0.7, 0.375), (0.75, 0.375), zorder=6, ha='left', va='center',
    arrowprops=dict(arrowstyle='-[, widthB=8, lengthB=1', connectionstyle='arc3,rad=0')
)

# Zoom

con_r1 = patches[2].get_extents().get_points()[0]
con_z1 = (0, 0)

con_r2 = patches[2].get_extents().get_points()[1]
con_z2 = Qvalues.shape[::-1][1:]

ax.add_artist(ConnectionPatch(con_r1, con_z1, 'data', 'data', axesA=axin2, axesB=axin3_3, zorder=100))
ax.add_artist(ConnectionPatch(con_r2, con_z2, 'data', 'data', axesA=axin2, axesB=axin3_1, zorder=100))

############
# segments #
############

axin4 = add_inset(ax, 0, 1 / 3 * 0.9, width=1 / 3, box=True)

pixelwidth = pixelheight = 0.025

n_segments1 = 4
n_segments2 = 3

sub_width, sub_height = 0.25, 0.7

radius = 0.33
anchor = get_positions(0 * radius, angle=30)
axin4_1 = add_inset(axin4, anchor[0], anchor[1], sub_width, sub_height, box=True, zorder=2)
axin4_1.set_aspect('equal', 'box')
axin4.text(0.25 / 2 + anchor[0], 0.725 + anchor[1], r'$Z_1$', **text_settings)

anchor = get_positions(1 * radius, angle=30)
axin4_2 = add_inset(axin4, anchor[0], anchor[1], sub_width, sub_height, box=True, zorder=1)
axin4.text(0.25 / 2 + anchor[0], 0.725 + anchor[1], r'$Z_2$', **text_settings)

Qvalues1 = Qvalues[:, :, :3]
labels1 = labels[:, :, :3]

Qmaxs = []
Qlabels = []
Qts = []

for t in range(Qvalues1.shape[2]):
    labelarr = labels1[:, :, t]
    for label in np.unique(labelarr):
        if label in [-2, -1]:  # nan and unlabeled are ignored
            continue
        labelmask = np.ones(labelarr.shape, dtype=bool)
        labelmask[labelarr == label] = 0
        Qmax = np.nanmax(np.ma.masked_array(Qvalues1[:, :, t], mask=labelmask))

        Qmaxs.append(Qmax)
        Qlabels.append(label)
        Qts.append(t)

idxs = np.argsort(Qmaxs)[::-1]

Qmaxs = [Qmaxs[i] for i in idxs]
Qlabels = [Qlabels[i] for i in idxs]
Qts = [Qts[i] for i in idxs]

segments1 = []
xmax, ymax = 0, 0
for n in range(n_segments1):
    label = Qlabels[n]
    t = Qts[n]
    mask = np.ones(labels1[:, :, t].shape, dtype=bool)
    mask[labels1[:, :, t] == label] = 0
    segment = Qvalues1[:, :, t].copy()
    segment[mask == 1] = np.nan

    segment = shrink_array(segment)

    xmax = max(segment.shape[0], xmax)
    ymax = max(segment.shape[1], ymax)
    segments1.append(segment)

for segment in segments1:
    if segment.shape == (xmax, ymax):
        continue
    xpad = xmax - segment.shape[0]
    ypad = ymax - segment.shape[1]
    s = np.pad(
        segment, ((int(np.floor(xpad / 2)), int(np.ceil(xpad / 2))), (int(np.floor(ypad / 2)), int(np.ceil(ypad / 2)))),
        mode='constant', constant_values=np.nan
    )
    segments1[segments1.index(segment)] = s


def plot_sequence(sequence, ax, cdots=True, text=None, ax_width=None, ax_height=None):
    assert isinstance(sequence, list)
    xmax = ymax = 0
    if cdots:
        cdots = len(sequence) - 1
    else:
        cdots = len(sequence)

    for entry in sequence:
        y, x = entry.shape
        xmax = max(x, xmax)
        ymax = max(y, ymax)

    ypos = 0
    for i, entry in enumerate(sequence[:cdots]):
        height, width = entry.shape
        ax.imshow(entry, extent=(0, width, ypos, ypos + height), vmin=-1, vmax=1, cmap='Blues')
        if text is not None:
            txt, text = text[0], text[1:]
            t = ax.text(xmax * 1.5, ypos + height / 2, txt, ha='left')
        ypos += height + 1

    if cdots < len(sequence):
        ypos += 5
        ax.scatter([xmax / 2] * 3, [ypos + i * ymax / 5 for i in range(-1, 2)], s=2, c='k')
        ypos += 5

    for i, entry in enumerate(sequence[cdots:]):
        if cdots == len(sequence):
            break
        height, width = entry.shape
        ax.imshow(entry, extent=(0, width, ypos, ypos + height), vmin=-1, vmax=1, cmap='Blues')
        if text is not None:
            txt, text = text[0], text[1:]
            t = ax.text(xmax * 1.5, ypos + height / 2, txt, ha='left')
        ypos += height + 1
    inv = ax.transData.inverted()
    text_width, _ = inv.transform((t.clipbox.width, 0))

    if ax_width is None:
        ax_width = xmax
    if ax_height is None:
        ax_height = ypos
    # ax_height =
    ax.set_xlim(0, ax_width)
    ax.set_ylim(ypos, 0)


H, W = axin4_1.get_figure().get_size_inches()
print(H, W)
_, _, w, h = axin4_1.get_position().bounds
print(w, h)
ratio = (H * h) / (W + w)
print(ratio)
Qs = segments1.copy()
del Qs[-1]
text = [rf'$m = {{{m}}}$' for m in range(1, len(Qs) + 1)]
text[-1] = r'$m = M_{Z_1}$'
plot_sequence(Qs, axin4_1, text=text, cdots=True, ax_width=xmax * 2.5)

# height = 0
# for i, segment in enumerate(segments1):
#     axin4_1.imshow(segment, extent=(0, segment.shape[1], height, height + ymax), vmin=-1, vmax=1)
#     height += ymax
# axin4_1.set_xlim(0, xmax * 2)
# axin4_1.set_ylim(0, height)


# heights = [i * 0.25 for i in [0, 2, 3]]
# subsub_height = ymax * pixelheight
# subsub_width = xmax * pixelwidth

# axin4_1_1 = add_inset(
#     axin4_1, 0, heights[2],
#     width=subsub_width, height=subsub_height, limited=False, zorder=1
# )
# axin4_1_1.pcolor(segments1[0], **pcolor_settings)
# plot_segments((~np.isnan(segments1[0])).T, ax=axin4_1_1, zorder=1, **plot_segments_settings)

# axin4_1_2 = add_inset(
#     axin4_1, 0, heights[1],
#     width=subsub_width, height=subsub_height, limited=False, zorder=1
# )
# axin4_1_2.pcolor(segments1[1], **pcolor_settings)
# plot_segments((~np.isnan(segments1[1])).T, ax=axin4_1_2, zorder=1, **plot_segments_settings)

# axin4_1_3 = add_inset(
#     axin4_1, 0, heights[0],
#     width=subsub_width, height=subsub_height, limited=False, zorder=1
# )
# axin4_1_3.pcolor(segments1[2], **pcolor_settings)
# plot_segments((~np.isnan(segments1[2])).T, ax=axin4_1_3, zorder=1, **plot_segments_settings)

# for i, h in enumerate(heights):
#     m = len(heights) - i
#     if m == 3:
#         axin4_1.text(0.4, h + subsub_height / 2, r"$m=M_{Z_1}$")
#     else:
#         axin4_1.text(0.4, h + subsub_height / 2, rf"$m={{{m}}}$")

# axin4_1.scatter([0.2 for _ in range(3)], [0.25, 0.3, 0.35], **cdots_settings)

Qvalues2 = Qvalues[:, :, 4:]
labels2 = labels[:, :, 4:]

Qmaxs = []
Qlabels = []
Qts = []

for t in range(Qvalues2.shape[2]):
    labelarr = labels2[:, :, t]
    for label in np.unique(labelarr):
        if label in [-2, -1]:  # nan and unlabeled are ignored
            continue
        labelmask = np.ones(labelarr.shape, dtype=bool)
        labelmask[labelarr == label] = 0
        Qmax = np.nanmax(np.ma.masked_array(Qvalues2[:, :, t], mask=labelmask))

        Qmaxs.append(Qmax)
        Qlabels.append(label)
        Qts.append(t)

idxs = np.argsort(Qmaxs)[::-1]

Qmaxs = [Qmaxs[i] for i in idxs]
Qlabels = [Qlabels[i] for i in idxs]
Qts = [Qts[i] for i in idxs]

segments2 = []
np.set_printoptions(linewidth=180, suppress=True, precision=2)
xmax, ymax = 0, 0
for n in range(n_segments2):
    label = Qlabels[n]
    t = Qts[n]
    mask = np.ones(labels2[:, :, t].shape, dtype=bool)
    mask[labels2[:, :, t] == label] = 0
    segment = Qvalues2[:, :, t].copy()
    segment[mask == 1] = np.nan

    segment = shrink_array(segment)

    xmax = max(segment.shape[0], xmax)
    ymax = max(segment.shape[1], ymax)
    segments2.append(segment)

for segment in segments2:
    if segment.shape == (xmax, ymax):
        continue
    xpad = xmax - segment.shape[0]
    ypad = ymax - segment.shape[1]
    s = np.pad(
        segment, ((int(np.floor(xpad / 2)), int(np.ceil(xpad / 2))), (int(np.floor(ypad / 2)), int(np.ceil(ypad / 2)))),
        mode='constant', constant_values=np.nan
    )
    segments2[segments2.index(segment)] = s

subsub_height = ymax * pixelheight
subsub_width = xmax * pixelwidth

heights = [i * 0.2 + 0.4 for i in range(n_segments2)]

axin4_2_1 = add_inset(
    axin4_2, 0, 0.8,
    width=subsub_width, height=subsub_height, limited=False, zorder=1
)
axin4_2_1.set_aspect('equal', 'box')
axin4_2_1.pcolor(segments2[0], **pcolor_settings)
plot_segments((~np.isnan(segments2[0])).T, ax=axin4_2_1, zorder=1, **plot_segments_settings)

axin4_2_2 = add_inset(
    axin4_2, 0, 0.6,
    width=subsub_width, height=subsub_height, limited=False, zorder=1
)
axin4_2_2.set_aspect('equal', 'box')
axin4_2_2.pcolor(segments2[1], **pcolor_settings)
plot_segments((~np.isnan(segments2[1])).T, ax=axin4_2_2, zorder=1, **plot_segments_settings)

axin4_2_3 = add_inset(
    axin4_2, 0, 0.4,
    width=subsub_width, height=subsub_height, limited=False, zorder=1
)
axin4_2_3.pcolor(segments2[2], **pcolor_settings)
plot_segments((~np.isnan(segments2[2])).T, ax=axin4_2_3, zorder=1, **plot_segments_settings)

for i, h in enumerate(heights):
    m = len(heights) - i
    axin4_2.text(0.4, h + subsub_height / 2, rf"$m={{{m}}}$")

cdots = get_cdots(get_positions(3.5 * radius), angle=40)
axin4.scatter(cdots[:, 0], cdots[:, 1], zorder=101, **cdots_settings)

##########
# Choice #
##########

axin5 = add_inset(ax, 1 / 3, 1 / 3 * 0.8, zorder=2)
axin5_1 = add_inset(axin5, 0, 0.4, height=1 - 0.4, limited=False)
axin5_2 = add_inset(axin5, n_segments2 * 0.1 * 2, 0.6 + 2 * 0.1, 0.3, 0.25, axis=True)

probs = np.zeros((2, n_segments1))
for m in range(n_segments1):
    probs[0, m] = max(np.nanmax(segments1[0]) - 1 * (np.nanmax(segments1[0]) - np.nanmax(segments1[m]) ** 5), 0)

for m in range(n_segments1):
    try:
        probs[1, m] = max(np.nanmax(segments2[0]) - 1 * (np.nanmax(segments2[0]) - np.nanmax(segments2[m]) ** 5), 0)
    except IndexError:
        probs[1, m] = np.nan
probs /= np.nansum(probs, axis=1)[:, np.newaxis]

settings = pcolor_settings.copy()
settings['cmap'] = 'Reds'
settings['vmin'] = 0
radius = 0.25

x, y = get_positions(radius)

p = probs[1, :][:, np.newaxis]
p = p[~np.isnan(p)][:, np.newaxis][::-1]
axin5_1_1 = add_inset(axin5_1, x, y, height=p.size * 0.3, limited=False, box=True)
axin5_1_1.set_aspect('equal', 'box')
axin5_1_1.pcolor(p, **settings)

p = probs[0, :][:, np.newaxis][::-1]
axin5_1_2 = add_inset(axin5_1, 0, 0, height=p.size * 0.3, limited=False, box=True)
axin5_1_2.set_aspect('equal', 'box')
axin5_1_2.pcolor(p, **settings)

settings = text_settings.copy()
settings['va'] = 'top'
axin5_1.text(0.45, -0.05, r'$Z_1$', **text_settings)
axin5_1.text(0.45 + x, y - 0.05, r'$Z_2$', **text_settings)

for i in range(4):
    h = 0.3 * (i + 0.4)
    m = 4 - i
    axin5_1.text(0, h, rf"$m={{{m}}}$", ha='right')

cdots = get_cdots((0.2, 0.9))
axin5.scatter(cdots[:, 0], cdots[:, 1], zorder=101, **cdots_settings)

x = np.linspace(0, 2)
y = np.max(np.array([1 - 1 * (x), np.zeros(x.size)]), axis=0)
axin5_2.plot(x, y, 'r')
axin5_2.set_xlim(0, 2)
axin5_2.set_ylim(0, 1.5)
axin5_2.set_xlabel(r'$Q_1^{max} - Q_m^{max}$', **xlabel_settings)
axin5_2.set_ylabel(r'$p_m(Z_i)$', **ylabel_settings)
axin5_2.set_yticklabels([])
axin5_2.set_xticks([0, 1, 2])
axin5_2.set_yticks([])


def plot_columns(columns, axis, radius=0.1, angle=55, size=1 / 3):
    global pcolor_settings
    zorder = len(columns)
    for i, column in enumerate(columns):
        x, y = get_positions(i * radius, angle)
        ax = add_inset(axis, x, y, 0.1, size, limited=False, zorder=zorder, box=True)
        if column.ndim == 2:
            ax.pcolor(column, **pcolor_settings)
        else:
            ax.pcolor(column[:, np.newaxis], **pcolor_settings)
        zorder -= 1


# axin5_3 = add_inset(axin5, 0, 0.525, n_segments2 * 0.15, 0.6, limited=False, axis=True)
# plot_columns([probs[0, :], probs[1, :]], axin5_3, size=0.8)
###########
# Segment #
###########


axin6 = add_inset(ax, 1 - 1 / 3 * figheight / figwidth, 1 / 3 * 0.9, zorder=1, width=1 / 3 * figheight / figwidth)

segment = segments1[1]

pixelwidth = pixelheight = 0.04
sub_height = segment.shape[0] * pixelheight
sub_width = segment.shape[1] * pixelwidth

axin6_1 = add_inset(
    axin6, 0.5 - sub_width / 2, 0.4 - sub_height / 2, width=sub_width, height=sub_height, zorder=1, limited=False
)
axin6_1.pcolor(segment, **pcolor_settings)
plot_segments((~np.isnan(segment)).T, ax=axin6_1, zorder=1, **plot_segments_settings)

ax.add_artist(
    ConnectionPatch((0.75, 0.25), (0.0, 0.25), 'data', 'data', axesA=axin4, axesB=axin6, zorder=100, arrowstyle='->')
)

########
# Copy #
########

axin7 = add_inset(ax, 0, 0, zorder=1, width=1 / 3 * figheight / figwidth)

axin7_1 = add_inset(
    axin7, 0.3 - sub_width / 2, 0.4 - sub_height / 2, width=sub_width, height=sub_height, zorder=1, limited=False
)
axin7_1.pcolor(segment, **pcolor_settings)
plot_segments((~np.isnan(segment)).T, ax=axin7_1, zorder=1, **plot_segments_settings)

#############
# Boltzmann #
#############

probs_width, probs_height = segment.shape[0] * 0.03, segment.shape[1] * 0.03 * figwidth / figheight

axin8 = add_inset(ax, 1 / 3, 0, zorder=2)
axin8_1 = add_inset(
    axin8,
    0, 0.3, probs_width, 0.9 * probs_height,
    limited=False
)
axin8_2 = add_inset(
    axin8,
    probs_width * 1.5, 0.6 + 2 * 0.1, 0.3, 0.25,
    axis=True
)

beta = 5
probs = np.exp(segment * beta)
probs /= np.nansum(probs)

settings = pcolor_settings.copy()
settings['cmap'] = 'Reds'
settings['vmin'] = 0
settings['vmax'] = min(np.nanmax(probs) * 1.2, 1)
axin8_1.pcolor(probs, **settings)
plot_segments((~np.isnan(probs)).T, ax=axin8_1, zorder=1, **plot_segments_settings)

beta = 2
x = np.linspace(-1, 1)
y = np.exp(beta * x)
y /= np.max(y)
axin8_2.plot(x, y)
# axin8_2.text(0.0, 0.7, r'$\exp(\beta * Q(a))$', **text_settings)
axin8_2.set_xlim(-1, 1)
# axin8_2.set_ylim(0, 1.5)
axin8_2.set_xlabel(r'$Q(a)$', **xlabel_settings)
axin8_2.set_ylabel(r'$p(Q)$', **ylabel_settings)
axin8_2.set_yticklabels([])
axin8_2.set_xticks([-1, 0.0, 1.0])
axin8_2.set_xticks([-1, 0, 1])


##########
# Action #
##########

axin9 = add_inset(ax, 2 / 3, 0, zorder=1, width=1 / 3 * figheight / figwidth)

mask = (~np.isnan(segment)).astype(float)
mask[mask == 0.] = np.nan

probs[np.nanargmax(probs)] = 0
action = np.unravel_index(np.nanargmax(probs), probs.shape)

axin9_1 = add_inset(
    axin9, 0.7 - sub_width / 2, 0.4 - sub_height / 2, width=sub_width, height=sub_height, zorder=1, limited=False
)
axin9_1.pcolor(segment, **pcolor_settings)
plot_segments((~np.isnan(segment)).T, ax=axin9_1, zorder=1, **plot_segments_settings)
axin9_1.scatter(action[1] + 0.5, action[0] + 0.5, c='r', marker='x')

ax.add_artist(
    ConnectionPatch((0.8, 0.25), (0.2, 0.25), 'data', 'data', axesA=axin7, axesB=axin9, zorder=100, arrowstyle='->')
)

plt.show()
# fig.savefig('Policy_sketch.png')
