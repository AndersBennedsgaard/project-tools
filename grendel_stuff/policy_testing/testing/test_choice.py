import numpy as np
from asla.modules.policies.SegmentPolicies import SegmentBoltzmannPolicy


mask = np.load('mask.npy')
Qvalues = np.load('Qvalues.npy')
Qvalues[mask == 1] = np.nan

kappa = 2
n_segments = 1
policy = SegmentBoltzmannPolicy(n_segments=n_segments, epsilon=0., kappa=kappa)
labels = policy.get_labels(Qvalues, mask)

N = []
for i in range(1000):
    seg_mask = policy.choose_segment(Qvalues, labels)
    dist = policy.get_boltzmann(Qvalues, mask, seg_mask)
    action, t = policy.draw_action(dist)

    seg_label = labels[action[0], action[1], t]
    seg_max = np.nanmax(Qvalues[labels == seg_label])
    N.append(seg_max == np.nanmax(Qvalues))

print(np.unique(N))

