import os
import matplotlib.pyplot as plt
from lib.maths import validate_databases, get_db_info, get_dir_structures, get_struc_feature
from asla.modules import db
from ase.io import write


directories = [
    '/home/abb/projects/policy_testing/results/2020_04_07/C5O2H6_EP_1/',
    '/home/abb/projects/policy_testing/results/2020_04_07/C5O2H6_MP_a0.9_g20_0/',
    '/home/abb/projects/policy_testing/results/2020_04_07/C5O2H6_MP_a1.0_0/',
]
common_prefix = os.path.commonprefix(directories)
labels = [direc.rstrip('/').lstrip(common_prefix) for direc in directories]

cwd = os.getcwd()

print("\nInitializing ...")

databases = []
for directory in directories:
    for file in os.listdir(directory):
        if '.db' in file:
            databases.append(db(directory + '/' + file))
            break
validate_databases(databases)

# Features
r_cutoff = 6

# Radial
eta = 0.25
r_center = 1.7

# Angular
xi = 3

energy_limit = -450
limit = 10

atom_types, grid_info = get_db_info(databases[0])
scale, size, pseudo3D, anchor, template_pos = grid_info

print("\nCalculating features for")

full_features = []
n_dat = [0]
for direc in directories:
    print("\t", direc)
    structures = get_dir_structures(direc, energy=energy_limit, limit=limit)
    features = []
    features = [get_struc_feature(struc.get_positions(), eta, r_center, xi, r_cutoff) for struc in structures]

    n_dat.append(n_dat[-1] + len(features))
    full_features.extend(features)

print('\nn_dat: ', n_dat)

print("\nPlotting features for ...")

plt.figure()
for i, label in enumerate(labels):
    print("\t", label)
    dat = full_features[n_dat[i]: n_dat[i + 1]]
    plt.scatter([x[0] for x in dat], [x[1] for x in dat], s=5, alpha=1, label=label)
plt.xlabel(r'$\rho^I$')
plt.ylabel(r'$\rho^{II}$')
plt.title(f'Structures with energy < -450eV')
plt.legend()
# plt.savefig(os.path.abspath("feature_test.png"), dpi=200)
plt.show()
