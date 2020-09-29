import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as clt
from matplotlib.animation import FuncAnimation
from asla.modules import db as DB
from ase.data.colors import jmol_colors
from ase.data import covalent_radii
from lib.maths import get_structures


def init():
    print("Initializing movie", flush=True)
    line.set_data([], [])
    dots.set_offsets([])
    return line, dots


def update(i):
    print(f"Drawing frame {i}", flush=True)
    epi = i * step
    bepi = best_episodes[epi]
    ce = energies[epi]
    be = best_energies[epi]
    cs = strucs[epi]
    bs = best_strucs[epi]

    line.set_data(range(len(energies[:epi+1])), energies[:epi+1])
    cdot.set_offsets([epi, ce])
    bdot.set_offsets([bepi, be])

    collection_cs.set_paths(get_patches(cs))
    collection_bs.set_paths(get_patches(bs))

    ax_cs.set_title(f"Current structure ({ce:.2f}eV)", fontsize=12)
    ax_bs.set_title(f"Best structure ({be:.2f}eV)", fontsize=12)


def get_patches(struc):
    patches = []
    for num, p in zip(struc.numbers, struc.positions):
        circle = plt.Circle(
                [p[0], p[1]], radius=covalent_radii[num] * 0.7
                )
        patches.append(circle)
    return patches


def draw_borders(ax):
    kwargs = dict(linewidth=2, color='blue')
    ax.plot([0, grid_size[0]], [0, 0], **kwargs)
    ax.plot([0, 0], [0, grid_size[1]], **kwargs)
    ax.plot([grid_size[0], grid_size[0]], [0, grid_size[1]], **kwargs)
    ax.plot([0, grid_size[0]], [grid_size[1], grid_size[1]], **kwargs)


step = 5

fil = 'C7_EP_0/db1.db'
db = DB(fil)

grid_size = list(db.get_value('grid.size'))[:2]
energies = db.get_col('energy')
strucs = get_structures(db)
radius = covalent_radii[6]

best_strucs = []
best_energies = []
best_episodes = []
cbe = energies[0]

xy_ratio = grid_size[0] / grid_size[1]

xmin = ymin = 0
xmax, ymax = grid_size

for i, (energy, struc) in enumerate(zip(energies, strucs)):
    if len(best_strucs) == 0:
        best_strucs.append(struc)
        best_energies.append(energy)
        best_episodes.append(i)
    elif energy < cbe:
        best_strucs.append(struc)
        best_energies.append(energy)
        best_episodes.append(i)
        cbe = energy
    else:
        best_strucs.append(best_strucs[-1])
        best_energies.append(best_energies[-1])
        best_episodes.append(best_episodes[-1])

patches = get_patches(strucs[0])
collection_cs = clt.PatchCollection(
        patches, ec='k', lw=0.5, facecolors=[jmol_colors[num] for num in strucs[0].numbers]
        )
collection_bs = clt.PatchCollection(
        patches, ec='k', lw=0.5, facecolors=[jmol_colors[num] for num in strucs[0].numbers]
        )
################

fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(ncols=2, nrows=2, height_ratios=[1,2], left=0.15, right=0.95, top=0.925, bottom=0.1, hspace=0.05)

ax_cs = fig.add_subplot(gs[0, 0])
ax_bs = fig.add_subplot(gs[0, 1])
ax_e = fig.add_subplot(gs[1, :])

line, = ax_e.plot([], [], lw=1, zorder=0)
cdot = ax_e.scatter([0], [energies[0]], s=100, marker='o', edgecolors='orange', zorder=1, facecolors='none', lw=2)
bdot = ax_e.scatter([0], [energies[0]], s=100, marker='o', edgecolors='tomato', zorder=1, facecolors='none', lw=2)

ax_cs.add_collection(collection_cs)
ax_bs.add_collection(collection_bs)

ax_e.set_xlim(0, len(energies))
ax_e.set_ylim(min(energies) - 0.05 * (max(energies)-min(energies)), max(energies) + 0.05 * (max(energies)-min(energies)))

ax_e.set_xlabel("Episodes", fontsize=14)
ax_e.set_ylabel("Energy", fontsize=14)
ax_cs.set_title(f"Current structure ({energies[0]}eV)", fontsize=12)
ax_bs.set_title(f"Best structure ({energies[0]}eV)", fontsize=12)

for ax in [ax_cs, ax_bs]:
    ax.set_aspect('equal', 'box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

anim = FuncAnimation(fig, update, interval=150, frames=len(energies)//step, blit=False, repeat=False)
anim.save('C7_EP_E.gif', writer='imagemagick', dpi=150)

# plt.show()

