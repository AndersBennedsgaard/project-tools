from datetime import datetime as dt
from ase.io import write
import matplotlib.pyplot as plt
from asla.modules.GPR_code.featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from lib.useful import PCA
from ase.atoms import symbols2numbers
import pickle
from asla.modules.policies.SegmentPolicies import SegmentBoltzmannPolicy, ChooseMaxs
import matplotlib
import numpy as np
import tensorflow as tf
from ase.io import read
from asla.asla import ASLA
from asla.modules.grid import Grid
from ase.calculators.dftb import Dftb
matplotlib.use('Agg')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Calculator

calc = Dftb(
    label='C2H4',
    Hamiltonian_SCC='No',
    kpts=(1, 1, 1),
    Hamiltonian_Eigensolver='Standard {}',
    Hamiltonian_MaxAngularMomentum_='',
    Hamiltonian_MaxAngularMomentum_C='"p"',
    Hamiltonian_MaxAngularMomentum_H='"s"',
    Hamiltonian_MaxAngularMomentum_O='"p"',
    Hamiltonian_MaxAngularMomentum_N='"p"',
    Hamiltonian_Charge='0.000000',
    Hamiltonian_Filling='Fermi {',
    Hamiltonian_Filling_empty='Temperature [Kelvin] = 0.000000'
)

# Path + Naming + Template

atoms_to_place = 'H4'

filename = __file__[:-3]
template = read('/home/abb/templates/C2.traj')

# General settings:
tf.reset_default_graph()

size = template.cell.lengths().tolist()
scale = size[0] / 100
grid_anchor = np.array([0, 0, 0])

n_rots = 4
use_rots = True
use_mirror = True
batch_size = 256 // (n_rots * use_rots + n_rots * use_mirror)

symbols = template.get_chemical_symbols()
positions = template.get_positions()
# Make sure Z = 0 isn't moved below 0, because of floating errors
assert (positions[:, 2] >= 0).all()

C_z = np.unique(positions[:, 2][np.array(symbols) == 'C'])
H_z = C_z.copy()

pseudo3D = {
    '1': H_z,
    '6': C_z,
}

choice = ChooseMaxs(
    alpha=1.,
    debug=False
)
policy = SegmentBoltzmannPolicy(
    atoms_to_place=atoms_to_place,
    epsilon=0.,
    beta=40,
    segment_choice=choice,
    sep_channels=True
)

kwargs_grid = {
    'pseudo3D': pseudo3D,
    'grid_pbc': [0, 0, 0],
    'size': size,
    'scale': scale,
    'grid_anchor': grid_anchor
}

kwargs_builder = {
    'atoms_to_place': atoms_to_place,
    'policy': policy,
}

kwargs_agent = {
    'learning_rate': 1e-3,
    'save_model_summaries': False
}

kwargs_trainer = {
    'use_rots': use_rots,
    'n_rots': n_rots,
    'use_mirror': use_mirror,
    'deltaE': 30,
    'spillover_radius': 1.0
}

kwargs_mask = {
    'c1': 0.5,
    'c2': None
}

asla = ASLA(
    kwargs_grid=kwargs_grid,
    kwargs_builder=kwargs_builder,
    kwargs_agent=kwargs_agent,
    kwargs_trainer=kwargs_trainer,
    kwargs_mask=kwargs_mask,
    template=template,
    calculator=calc,
    dbpath=None
)

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

# Initialize feature

feature_type_list = [
    'H-H',
    'H-C',
    'C-C',
    'H-H-H',
    'C-H-H',
    'C-H-C',
    'H-C-H',
    'C-C-H'
]
maindir = '/home/abb/projects/policy_testing/plot/files/'
strucs = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    asla.memory = pickle.load(open(maindir + 'memory000500.p', 'rb'))
    asla.agent.restore(
        sess, **{
            'postfix': '000500',
            'model_dir': maindir + 'model_checkpoint'
        }
    )
    N_runs = 1000
    dat = []
    print()
    for N_H in range(1, 5):
        now = dt.now()
        print("Current time: ", now.strftime("%H:%M:%S"))
        print(f"Calculating for {N_H} hydrogen atoms")

        atoms = read(maindir + 'C2H4_malthes_feature/C2H4_full.traj')
        while np.sum(atoms.numbers == 1) > N_H:
            del atoms[np.argmax(atoms.numbers == 1)]
        assert np.sum(atoms.numbers == 1) == N_H
        assert np.sum(atoms.numbers == 6) == 2

        fC = Angular_Fingerprint(
            atoms,  # Anything with the right cell and number of atoms
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
        atom_types = list(set(symbols2numbers(atoms_to_place)))
        for _ in range(N_runs):
            grid = Grid(template, **kwargs_grid)

            actions = []
            for _ in range(N_H):
                action, t, Qvalues, mask = policy.getAction(sess, grid)
                actions.append(action)
                grid.add_atom(action, atom_type=atom_types[t], t=t)
            strucs.append(grid)
            f = fC.get_feature(grid)
            dat.append(f.tolist())

write('trajs.traj', strucs)
try:
    dat = np.vstack(dat)
    dat = PCA(dat)
    np.save('PCA.npy', dat)

    plt.figure()
    for i in range(4):
        plt.scatter(dat[i * N_runs: (i + 1) * N_runs, 0], dat[i * N_runs: (i + 1) * N_runs, 0], label=f'H{i + 1}')
    plt.legend()
    plt.savefig('plot.png')
except ValueError:
    dat = np.array(dat)
    np.save('Data.npy', dat)
    print("Fingerprints of different size - saving fingerprints")
