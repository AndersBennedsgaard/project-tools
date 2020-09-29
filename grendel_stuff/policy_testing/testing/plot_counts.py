from asla.modules.policies.SegmentPolicies import SegmentationPolicy as SBP
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib as m
m.use('Agg')

grids = pickle.load(open('/home/abb/projects/policy_testing/testing/files/grids00100.pkl', 'rb'))
Qvalues = pickle.load(open('/home/abb/projects/policy_testing/testing/files/Qvalues00100.pkl', 'rb'))
masks = pickle.load(open('/home/abb/projects/policy_testing/testing/files/masks00100.pkl', 'rb'))

count_matrix = None
policy = SBP('C2O2H4')
for Q, mask, grid in zip(Qvalues, masks, grids):
    if count_matrix is None:
        count_matrix = np.zeros(Q.shape, dtype=int)
    for _ in range(1000):
        action, t, _, _ = policy.getAction(Q, mask, grid)
        count_matrix[action[0], action[1], t] += 1

np.save('count_matrix.npy', count_matrix)

fig, axes = plt.subplots(1, 3, figsize=(15, 8))
for i in range(3):
    axes[i].imshow(count_matrix[:, :, i], origin='lower')
    axes[i].axis('off')
fig.tight_layout()
fig.savefig('plot_first_actions.png')
