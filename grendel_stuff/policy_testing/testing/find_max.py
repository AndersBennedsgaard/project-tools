import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from asla.modules.policies import ImageSegmentation as ims
matplotlib.use('TkAgg')


Qvalues = np.load('Qvalues.npy')
mask = np.load('mask.npy')

Qvalues[mask == 1] = np.nan

Qarg_max_lists = []
for i in range(Qvalues.shape[2]):
    Qarg_max_lists.append(ims.get_local_minima(Qvalues[:, :, i]))

Qarg_max = []
ts = []
Qmax = []
for t, max_layer in enumerate(Qarg_max_lists):
    for m in max_layer:
        Qarg_max.append(m)
        ts.append(t)
        Qmax.append(Qvalues[m[0], m[1], t])
print("No. of Q maxima: ", len(Qmax))

probabilities = np.zeros(Qvalues.shape)
for i, t in enumerate(ts):
    local_argmax = Qarg_max[i]
    sigma = 1 / (Qmax[i] + 1 + 1e-06)
    x, y = np.meshgrid(np.arange(Qvalues.shape[0]), np.arange(Qvalues.shape[1]))
    probabilities[:, :, t] += 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(
            -(x - local_argmax[0])**2 / 2 / sigma**2 - (y - local_argmax[1])**2 / 2 / sigma**2
            ).T
probabilities[mask == 1] = 0
probabilities /= np.sum(probabilities)

mis = [np.finfo('float32').min] * 4
mis.append(-0.06)
mis.append(-0.06)
mas = [np.finfo('float32').min] * 4
mas.append(0.06)
mas.append(0.06)

for t in [1, 4, 5]:
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(Qvalues[:, :, t], vmin=mis[t], vmax=mas[t])
    axes[0].scatter(
            [x[1] for i, x in enumerate(Qarg_max) if ts[i] == t], 
            [x[0] for i, x in enumerate(Qarg_max) if ts[i] == t], 
            c='r', s=3
            )
    axes[1].imshow(probabilities[:, :, t])
    axes[1].scatter(
            [x[1] for i, x in enumerate(Qarg_max) if ts[i] == t], 
            [x[0] for i, x in enumerate(Qarg_max) if ts[i] == t], 
            c='r', s=3
            )
    fig.suptitle('t = {}'.format(t))

plt.show()
