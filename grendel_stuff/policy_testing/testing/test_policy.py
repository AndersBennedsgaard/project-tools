from ase.calculators.lj import LennardJones as LJ
import unittest
import pickle
from asla.modules.policies.SegmentPolicies import TestPolicy as TP
from asla.modules.policies.Policies import EpsilonPolicy as EP
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from ase.io import read
from asla.asla import ASLA
from asla.modules.grid import Grid
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.reset_default_graph()


calc = LJ()
atoms_to_place = 'H1'
template = read('files/H.traj')

size = template.cell.lengths().tolist()

symbols = template.get_chemical_symbols()
positions = template.get_positions()

H_z = np.unique(positions[:, 2][np.array(symbols) == 'H'])

policy = TP()

kwargs_grid = {
    'pseudo3D': {'1': H_z},
    'grid_pbc': [0] * 3,
    'size': size,
    'scale': size[0] / 10,
    'grid_anchor': np.array([0] * 3)
}

kwargs_builder = {
    'atoms_to_place': atoms_to_place,
    'policy': policy
}

asla = ASLA(
    kwargs_grid=kwargs_grid,
    kwargs_builder=kwargs_builder,
    template=template,
    calculator=calc,
    dbpath=None
)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
with open('files/memory001000.p', 'rb') as mem:
    asla.memory = pickle.load(mem)
asla.agent.restore(
    sess, **{
        'postfix': '001000',
        'model_dir': 'files/model_checkpoint'
    }
)
grid = Grid(template, **kwargs_grid)

output = policy.getAction(sess, grid)
