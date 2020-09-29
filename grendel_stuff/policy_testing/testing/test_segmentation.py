import numpy as np
from asla.modules.policies import ImageSegmentation as ims


def get_markers(Q):
    tmp_markers = ims.get_local_maxima(Q, pbc=True)
    markers = []
    marker_labels = []
    i = 1
    for m in tmp_markers:
        if Q[m] >= 0:
            markers.append(m)
            marker_labels.append(i)
            i += 1
        else:
            marker_labels.append(0)
            markers.append(m)
    if i == 1:  # If no Q-values are above 0
        print("Warning: all Q-values below 0 - using the maximum Q-value action as foreground marker")
        Q_argmax = np.unravel_index(np.nanargmax(Q), Q.shape)
        marker_labels[markers.index(Q_argmax)] = 1
    return markers, marker_labels


Qvalues = np.load('files/test_segmentation_Qvalues.npy')

processing = []

markers, marker_labels = get_markers(Qvalues)

# labels = ims.watershed(Qvalues)
# labels = ims.watershed(-ims.morph_grad(Qvalues, pbc=True))

# labels = ims.watershed(Qvalues, pbc=True, markers=markers, marker_labels=marker_labels)
labels = ims.watershed(-ims.morph_grad(Qvalues, pbc=True), markers=markers, marker_labels=marker_labels)
