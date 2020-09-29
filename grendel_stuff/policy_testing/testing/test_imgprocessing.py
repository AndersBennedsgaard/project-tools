import numpy as np
import matplotlib.pyplot as plt
from lib.useful import remove_ticks
from asla.utils.plot_segments import plot_segments
from asla.modules.policies import ImageSegmentation as ims


def segments(ax, labels):
    for label in np.unique(labels):
        plot_segments((labels == label).T, ax=ax, correct=False, colors='k', linestyle='-', linewidths=0.5)


def plot(Qprocd):
    global Qvalues, ax_idx_i, ax_idx_j, axes, ylabel, markers, marker_labels
    ax = axes[ax_idx_i]

    if ax_idx_j == 2:
        labels = ims.watershed(Qprocd, pbc=True)
        ax[0].pcolor(Qprocd, vmin=-1, vmax=1, cmap='Blues')
        segments(ax[0], labels)

        ax[1].pcolor(Qvalues, vmin=-1, vmax=1, cmap='Blues')
        segments(ax[1], labels)

    labels = ims.watershed(Qprocd, pbc=True, markers=markers, marker_labels=marker_labels)
    ax[ax_idx_j].pcolor(Qvalues, vmin=-1, vmax=1, cmap='Blues')
    segments(ax[ax_idx_j], labels)

    if ax_idx_j == 2:
        ax[0].set_ylabel(ylabel, rotation=90)
        if ax_idx_i == 0:
            ax[0].set_title('Processed Q-values')
            ax[1].set_title('Real Q-values')
            ax[2].set_title('Real Q-values with markers ->')

    mx = [m[0] + 0.5 for m in markers]
    my = [m[1] + 0.5 for m in markers]
    ax[ax_idx_j].scatter(my, mx, s=20, marker='o', facecolors='none', edgecolors='r')
    ax[ax_idx_j].set_xlim(0, Qprocd.shape[1])
    ax[ax_idx_j].set_ylim(0, Qprocd.shape[0])

    ax_idx_i += 1


Q_all = np.load('files/Qvalues_02000.npy')
N = 3
fig, axes = plt.subplots(6, N + 2, figsize=(11, 15))

for ax in axes.flatten():
    ax.set_aspect('equal', 'box')
    remove_ticks(ax)

for layer in range(N):
    ax_idx_i = 0
    ax_idx_j = layer + 2

    Qvalues = Q_all[layer, :, :, 0]
    Qvalues[Qvalues < -1] = np.nan

    tmp_markers = ims.get_local_maxima(Qvalues, pbc=True)
    markers = []
    marker_labels = []
    i = 1
    for m in tmp_markers:
        if Qvalues[m] >= 0:
            markers.append(m)
            marker_labels.append(i)
            i += 1
        else:
            marker_labels.append(0)
            markers.append(m)

    # 1

    ylabel = r'$Q$'

    procQvalues = Qvalues.copy()
    plot(procQvalues)

    # 2

    ylabel = r'$-\nabla Q$'

    procQvalues_grad = -ims.morph_grad(Qvalues, pbc=True)
    plot(procQvalues_grad)

    # 3

    ylabel = r'Filtered $Q$'

    procQvalues = ims.close_operator(ims.open_operator(procQvalues, pbc=True))
    plot(procQvalues)

    # 4

    ylabel = r'Further smoothening of $Q$'

    procQvalues = ims.close_operator(ims.dilate_image(procQvalues, pbc=True), pbc=True)
    plot(procQvalues)

    # 5

    ylabel = r'$-\nabla$(smooth $Q$)'

    procQvalues_grad = -ims.morph_grad(procQvalues, pbc=True)
    plot(procQvalues_grad)

    # 6

    ylabel = r'Black tophat of smooth $Q$'

    procQvalues = ims.blackhat_transform(procQvalues, pbc=True)
    plot(procQvalues)

rect = [0, 0.03, 1, 0.95]
fig.tight_layout(rect=rect)
fig.savefig('images/procd_Q.png', dpi=150)
