import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from asla.modules import db
from asla.modules.GPR_code.featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from lib.maths import get_db_info, validate_databases, get_dir_structures, get_structures
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

energy_limit = -450
unique = True

print("\nInitializing ...")
databases = []
for directory in directories:
    for file in os.listdir(directory):
        if '.db' in file:
            databases.append(db(directory + '/' + file))
            break
validate_databases(databases)

atom_types, grid_info = get_db_info(databases[0])
scale, size, pseudo3D, anchor, template_pos = grid_info

struc = get_structures(databases[0], atom_types, size, limit=1)[0]
fC = Angular_Fingerprint(
    struc,  # Anything with the right cell and number of atoms
    Rc1=Rc1,
    Rc2=Rc2,
    binwidth1=binwidth1,
    Nbins2=Nbins2,
    sigma1=sigma1,
    sigma2=sigma2,
    gamma=gamma,
    eta=eta
)

print("Calculating features for")

n_dat = [0]
full_data = []
for direc in directories:
    print(direc)
    structures = get_dir_structures(direc, energy=energy_limit, unique=unique)
    d = [fC.get_feature(struc).tolist() for struc in structures]
    n_dat.append(n_dat[-1] + len(d))
    full_data.extend(d)
    print(f"\tNo. of features: {len(d)}")

print('n_dat: ', n_dat)
print("\nRunning PCA ...")

features = scale(full_data)

pca = PCA(n_components=2)
results = pca.fit_transform(full_data)
np.save(os.path.abspath('files/PCA_results.npy'), results)
print("Done")

variance_ratio = pca.explained_variance_ratio_
print(f'Variance: {variance_ratio[0]:.2f}, {variance_ratio[1]:.2f} (sum: {np.sum(variance_ratio):.2f})')

print("\nPlotting ...")
for i, label in enumerate(labels):
    dat = results[n_dat[i]: n_dat[i + 1], :]
    plt.scatter(dat[:, 0], dat[:, 1], s=10, alpha=0.5, label=label)
plt.xlabel('Principle component 1')
plt.ylabel('Principle component 2')
plt.title('Structures with energy < -450eV')
plt.legend()
plt.savefig(os.path.abspath("images/PCA.png"), dpi=200)
