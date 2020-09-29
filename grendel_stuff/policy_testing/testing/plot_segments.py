from asla.modules.plot_episode_3D import plot_episode
import pickle
import numpy as np
import matplotlib as m
m.use('Agg')

grids = pickle.load(open('/home/abb/projects/policy_testing/testing/files/grids00100.pkl', 'rb'))
Qvalues = pickle.load(open('/home/abb/projects/policy_testing/testing/files/Qvalues00100.pkl', 'rb'))
masks = pickle.load(open('/home/abb/projects/policy_testing/testing/files/masks00100.pkl', 'rb'))

count_matrix = np.load('files/count_matrix.npy')
# count_matrix = [count_matrix] * len(Qvalues)

plot_episode(grids, Qvalues, masks, 0, pseudo3D=grids[0].pseudo3D, segmentation=True, count_matrix=count_matrix)
