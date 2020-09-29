import matplotlib.pyplot as plt
import numpy as np
from lib.maths import get_atomic_features


eta = 0.25
r_center = 1.7
xi = 1
r_cutoff = 10

points = [(1, 1), (4, 2), (4.5, 4.5), (2, 5)]
points = np.array(points)

features = get_atomic_features(points, eta, r_center, xi, r_cutoff)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

ax = axes[0]
ax.scatter([x[0] for x in points], [x[1] for x in points])
for i, pi in enumerate(points):
    for j, pj in enumerate(points):
        if j <= i:
            continue
        ax.annotate(
            '', xy=pj, xytext=pi, arrowprops=dict(arrowstyle='<|-|>', connectionstyle='arc3')
        )
        ax.text(
            (pj[0] + pi[0]) / 2, (pj[1] + pi[1]) / 2, rf'$r_{{{i}{j}}}$',
            {'fontsize': 16, 'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)}
        )
ax.set_xlabel('X')
ax.set_ylabel('Y')

ax = axes[1]
for i, f in enumerate(features):
    ax.scatter(f[0], f[1])
    ax.text(f[0], f[1], f'{i}')
ax.set_xlabel(r'$\rho_i^I$')
ax.set_ylabel(r'$\rho_i^{II}$')

plt.show()
