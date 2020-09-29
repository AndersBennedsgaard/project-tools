from plot_episode_slab import _get_diff_atoms, plot_slab
from ase.io import read


template = read('/home/abb/templates/MgO_fcc100_surface.traj')

added = read('bestStruc.traj')
added = _get_diff_atoms(template, added)

pseudo_heights = list(set(added.positions[:, 2]))

plot_slab(
        template, added, pseudo_heights, xlim=[2, 13], ylim=[1, 20], zlim=None, 
        filename='MgO_7Au.png', save=True, plot=False
        )

