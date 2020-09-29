import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate as interpolate
from ase.io import read
from ase.visualize.plot import plot_atoms
import asla.modules.policies.ImageSegmentation as ims
from IPython import embed  # noqa
from lib.plot_segments import plot_world_outlines


def remove_ticks(ax):
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks_position('none')


def plot_mask(ax):
    plot_atoms(struc, ax, scale=4.825, offset=(6.25, 6.25), radii=0.9)
    ax.set_xlim(-0.5, QC23.T.shape[0] + 0.5)
    ax.set_ylim(-0.5, QC23.T.shape[1] - 0.5 - 1)


plot = False

beta = 45
n_segments = 4
kappa = 0
layer = 5
epsilon = 0.15

struc = read('C23.traj')

QC23 = np.load('Qvalues_01100.npy')
QC23 = QC23[layer, :, :, 0]
QC23 = np.roll(QC23, -5, axis=1)
QC23 = np.roll(QC23, 4, axis=0)
QC23 = QC23[4:-4, 3:-1]
mask = np.zeros(QC23.shape)
mask[np.logical_or(QC23 < -1, QC23 > 1)] = 1
QC23[mask == 1] = np.nan

M = np.zeros(mask.shape + (4,))
M[:, :, 3][mask == 1] = 0.4
######################

p_boltz = np.exp(QC23 * beta)
p_boltz[mask == 1] = 0
try:
    p_boltz = p_boltz / np.sum(p_boltz)
except FloatingPointError as e:
    print(np.sum(p_boltz))
    raise e

P_boltz = np.zeros(p_boltz.shape + (4,))
cmap = cm.get_cmap('viridis', 1)
val = 0.8
P_boltz[:, :, 0:3] = np.array([255, 0, 0]) / 255
P_boltz[:, :, 3] = (p_boltz - np.min(p_boltz)) / \
    (np.max(p_boltz) - np.min(p_boltz))

S = ims.watershed(QC23, pbc=True)
markers = ims.get_local_maxima(QC23, pbc=True)
Qmax = [QC23[m] for m in markers]
idxs = np.argsort(Qmax)[::-1]
markers = [markers[i] for i in idxs]
segments = [S == S[m] for m in markers[:n_segments]]

z = np.zeros(QC23.shape, dtype=float)
p = []
for segment in segments:
    p_tmp = z.copy()
    p_tmp[segment] += np.exp(QC23[segment] * beta)
    p.append(p_tmp)
p = [p / np.sum(p) for p in p]
p_per = [(n_segments - i)**kappa for i in range(n_segments)]
p_per = [p / sum(p_per) for p in p_per]
p = [p_per[i] * p[i] for i in range(n_segments)]
p_seg = np.zeros(QC23.shape, dtype=float)
for p_s in p:
    p_seg += p_s
p_seg[mask == 1] = 0
p_seg /= np.sum(p_seg)

p_eps = np.ones(QC23.shape, dtype=float)
p_eps[mask == 1] = 0
p_eps /= np.sum(p_eps)

p_tot = (1 - epsilon) * p_seg + epsilon * p_eps
p_tot /= np.sum(p_tot)

P_tot = np.zeros(p_tot.shape + (4,))
cmap = cm.get_cmap('viridis', 1)
val = 0.8
P_tot[:, :, 0:3] = np.array([255, 0, 0]) / 255
P_tot[:, :, 3] = (p_tot - np.min(p_tot)) / (np.max(p_tot) - np.min(p_tot))

cmapp = 'hot_r'

#########################

cmap = 'Blues'
cmapc = 'jet'

x, y = np.meshgrid(np.arange(p_tot.shape[0]), np.arange(p_tot.shape[1]))
s = 500
s_boltz = np.log10(1 + p_boltz[x, y]) * s
s_tot = np.log10(1 + p_tot[x, y]) * s

fig = plt.figure(figsize=(12, 6))

ax = plt.subplot(1, 2, 1)
ax.imshow(QC23.T, origin='lower', vmin=np.nanmin(
    QC23), vmax=np.nanmax(QC23), cmap=cmap)
ax.scatter(x, y, c='r', s=s_boltz)
plot_mask(ax)
ax.set_title('Boltzmann')
remove_ticks(ax)

ax = plt.subplot(1, 2, 2)
ax.imshow(QC23.T, origin='lower', vmin=np.nanmin(
    QC23), vmax=np.nanmax(QC23), cmap=cmap)
# ax.imshow(np.swapaxes(M, 0, 1), origin='lower')
plot_mask(ax)
for seg in segments:
    plot_world_outlines(seg, ax)

ax.scatter(x, y, c='r', s=s_tot)
ax.set_title('Segmentation Boltzmann')
remove_ticks(ax)

fig.tight_layout()

if not plot:
    fig.savefig('Boltzmann2D_dots.png')
#####################


def levels(p, ls=6):
    levs = np.exp(30 * np.linspace(np.min(p[p > 0]), np.max(p), ls)) - 1
    levs *= np.max(p) * 0.8 / np.max(levs)
    levs += 0.01 * np.max(p)
    return levs


x_new = np.linspace(0, p_tot.shape[0], 500)
y_new = np.linspace(0, p_tot.shape[1], 500)

spline_boltz = interpolate.RectBivariateSpline(
    np.arange(p_boltz.shape[0]), np.arange(p_boltz.shape[1]), p_boltz, kx=1, ky=1
)
spline_boltz = spline_boltz(x_new, y_new)
spline_boltz = spline_boltz - np.min(spline_boltz)

spline_tot = interpolate.RectBivariateSpline(
    np.arange(p_tot.shape[0]), np.arange(p_tot.shape[1]), p_tot, kx=1, ky=1
)
spline_tot = spline_tot(x_new, y_new)
spline_tot = spline_tot - np.min(spline_tot)

fig = plt.figure(figsize=(12, 6))

ax = plt.subplot(1, 2, 1)
ax.imshow(QC23.T, origin='lower', vmin=np.nanmin(
    QC23), vmax=np.nanmax(QC23), cmap=cmap)
# ax.imshow(np.swapaxes(M, 0, 1), origin='lower')
plot_mask(ax)
ax.contour(spline_boltz.T, levels=levels(spline_boltz),
           extent=(0, p_boltz.shape[0], 0, p_boltz.shape[1]))
ax.set_title('Boltzmann')
remove_ticks(ax)

ax = plt.subplot(1, 2, 2)
ax.imshow(QC23.T, origin='lower', vmin=np.nanmin(
    QC23), vmax=np.nanmax(QC23), cmap=cmap)
# ax.imshow(np.swapaxes(M, 0, 1), origin='lower')
plot_mask(ax)
for seg in segments:
    plot_world_outlines(seg, ax)
ax.contour(spline_tot.T, levels=levels(spline_tot),
           extent=(0, p_tot.shape[0], 0, p_tot.shape[1]))
ax.set_title('Segmentation Boltzmann')
remove_ticks(ax)

fig.tight_layout()

####################


if plot:
    plt.show()
else:
    fig.savefig('Boltzmann2D_contour.png')
