import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ase.atoms import Atoms
from asla.modules.db import deblob, deblob_dict
from asla.modules import db as DB
from asla.modules.GPR_code.featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
matplotlib.use('Agg')


directories = [
    '/home/abb/projects/policy_testing/results/2020_04_07/C5O2H6_EP_1/',
    '/home/abb/projects/policy_testing/results/2020_04_07/C5O2H6_MP_a0.9_g20_0/',
    '/home/abb/projects/policy_testing/results/2020_04_07/C5O2H6_MP_a1.0_0/',
]
common_prefix = os.path.commonprefix(directories)
labels = [direc.rstrip('/').lstrip(common_prefix) for direc in directories]

cwd = os.getcwd()

# Fingerprint parameters
Rc1 = 6
binwidth1 = 0.2
sigma1 = 0.2

Rc2 = 4
Nbins2 = 30
sigma2 = 0.2
gamma = 2

# Radial/angular weighting
eta = 20
use_angular = True

# Initializing
numbers = None
atom_types = None
grid_info = None
fC = None
print("\nInitializing ...")
for direc in directories:
    for file in os.listdir(direc):
        if '.db' in file:
            db = DB(direc + '/' + file)
            break

    if numbers is None:
        numbers = db.get_nth('atom_numbers')
        numbers = deblob(numbers).astype(int)

    if atom_types is None:
        atom_types = db.get_nth('atom_types', table_name='Bob_the_Builder')
        atom_types = deblob(atom_types).astype(int)
    else:
        if set(atom_types) != set(deblob(db.get_nth('atom_types', table_name='Bob_the_Builder')).astype(int)):
            print(atom_types)
            print(deblob(atom_types).astype(int))
            raise ValueError("atom_types not equal")

    if grid_info is None:
        grid_info = db.get_nth('*', table_name='grid')
        scale, size, pseudo3D, anchor, template_pos = grid_info
        scale = float(scale)
        size = deblob(size)
        pseudo3D = deblob_dict(pseudo3D)
        anchor = deblob(anchor)
        template_pos = deblob(template_pos, shape=(-1, 3)).astype(int)
    else:
        _grid_info = db.get_nth('*', table_name='grid')
        _scale, _size, _pseudo3D, _, _ = _grid_info
        _scale = float(_scale)
        _size, _pseudo3D = deblob(_size), deblob_dict(_pseudo3D)
        if scale != _scale:
            print(scale, _scale)
            raise ValueError("grid_info scale not equal")
        if not (size == _size).all():
            print(size, _size)
            print(type(size), type(_size))
            raise ValueError("grid_info size not equal")
        if pseudo3D != _pseudo3D:
            print(pseudo3D, _pseudo3D)
            raise ValueError("grid_info pseudo3D not equal")

    if fC is None:
        positions = db.get_nth('positions')
        positions = deblob(positions, shape=(-1, 3)).astype(int)
        structure = Atoms([atom_types[n] for n in numbers], positions=positions, cell=size)

        fC = Angular_Fingerprint(
            structure,  # Anything with the right cell and number of atoms
            Rc1=Rc1,
            Rc2=Rc2,
            binwidth1=binwidth1,
            Nbins2=Nbins2,
            sigma1=sigma1,
            sigma2=sigma2,
            gamma=gamma,
            eta=eta,
            use_angular=use_angular
        )

print("\nGetting data ...")
n_dat = [0]
full_data = []
print("Calculating features for")
for direc in directories:
    print(direc)
    d = []
    for file in os.listdir(direc):
        if '.db' not in file:
            continue
        print("\t", file)
        db = DB(direc + '/' + file)
        cursor = db.con.cursor()
        cursor.execute(f"SELECT atom_numbers, positions FROM structures WHERE energy < -450")
        vals = cursor.fetchall()
        numbers, positions = list(map(lambda x: x[0], vals)), list(map(lambda x: x[1], vals))

        numbers = [deblob(n).astype(int) for n in numbers]
        positions = [deblob(pos, shape=(-1, 3)).astype(int) for pos in positions]

        structures = [
            Atoms([atom_types[n] for n in nums], positions=pos, cell=size) for nums, pos in zip(numbers, positions)
        ]
        for struc in structures:
            d.append(fC.get_feature(struc).tolist())
    n_dat.append(n_dat[-1] + len(d))
    full_data.extend(d)

print('n_dat: ', n_dat)

print("\nRunning PCA ...")
pca = PCA(n_components=2)
results = pca.fit_transform(full_data)
np.save(os.path.abspath('PCA_results.npy'), results)
print("Done")

variance_ratio = pca.explained_variance_ratio_
print(f'Variance: {variance_ratio[0]:.2f}, {variance_ratio[1]:.2f} (sum: {np.sum(variance_ratio):.2f})')

print("\nPlotting ...")
for i, label in enumerate(labels):
    dat = results[n_dat[i]: n_dat[i + 1], :]
    plt.scatter(dat[:, 0], dat[:, 1], s=10, alpha=0.5, label=label)
plt.xlabel('Principle component 1')
plt.ylabel('Principle component 2')
plt.title('Structures with energy <-450eV')
plt.legend()
plt.savefig(os.path.abspath("PCA.png"), dpi=200)
