import os
import matplotlib.pyplot as plt
from ase.io import write, read
from ase.visualize.plot import plot_atoms
from asla.modules import db
from lib.maths import get_db_info, validate_databases, get_dir_structures
from lib.useful import remove_ticks


directories = [
    # '/home/abb/projects/policy_testing/results/2020_04_07/C5O2H6_EP_1/',
    '/home/abb/projects/policy_testing/results/2020_04_07/C5O2H6_MP_a0.9_g20_0/',
    '/home/abb/projects/policy_testing/results/2020_04_07/C5O2H6_MP_a1.0_0/',
]
common_prefix = os.path.commonprefix(directories)
labels = [direc.rstrip('/').lstrip(common_prefix) for direc in directories]

cwd = os.getcwd()

energy_limit = -450
limit = 10

print("\nValidating ...")
databases = []
for directory in directories:
    for file in os.listdir(directory):
        if '.db' in file:
            databases.append(db(directory + '/' + file))
            break
validate_databases(databases)

atom_types, grid_info = get_db_info(databases[0])
scale, size, pseudo3D, anchor, template_pos = grid_info

print("\nGetting structures ...")
for direc in directories:
    print(direc)
    structures = get_dir_structures(direc, energy=energy_limit, limit=limit, unique=True)

    try:
        save_strucs = read('testing_strucs.traj', index=':')
    except FileNotFoundError:
        save_strucs = []
    for i in range(len(structures) // limit):
        strucs = structures[limit * i:limit * (i + 1)]
        fig, axes = plt.subplots(5, 2, figsize=(12, 12))
        for j, struc in enumerate(strucs):
            ax = axes.flatten()[j]
            plot_atoms(struc, ax)
            ax.set_title(j)
            ax.axis('equal')
            remove_ticks(ax)
        fig.tight_layout()
        plt.show()
        ans = input("Use structures [idxs/done]")
        if ans.lower() in ('save', 'done'):
            write('testing_strucs.traj', save_strucs)
        elif ans == '':
            save_strucs.append(strucs[0])
        else:
            for idx in ans.split(','):
                try:
                    save_strucs.append(strucs[int(idx)])
                except ValueError:
                    print("Input not understood - ignoring")
write("testing_strucs.traj", save_strucs)
