import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import pylint: disable=unused-import
from ase.io import read
from lib.maths import get_wACSF_features


def structure_features(structures, eta, mu, xi, r_cutoff, lamb=1):
    features = [
        np.sum(np.array(get_wACSF_features(struc, eta, mu, xi, r_cutoff, lamb)), axis=0) for struc in structures
    ]
    return np.array(features)


r_cutoff = 10

# Radial
eta = 0.05
mu = 1.5

# Angular
xi = 1

strucs = read('/home/abb/projects/policy_testing/plot/files/testing_strucs.traj', index=':')
print("Number of structures: ", len(strucs))
struc = strucs[0]
features1 = get_wACSF_features(struc, eta, mu, xi, r_cutoff)
features1 = np.array(features1)

features2 = get_wACSF_features(struc, eta, mu, xi, r_cutoff, lamb=-1)
features2 = np.array(features2)

positions = struc.get_positions()
numbers = struc.numbers

fig = plt.figure(figsize=(20, 10))

xi = 1
print("Computing first range of features ...")
sfeatures = structure_features(strucs, eta, mu, xi, r_cutoff, lamb='both')
kmeans = KMeans(n_clusters=6)
kmeans.fit(sfeatures)
labels = kmeans.labels_

ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.scatter(sfeatures[:, 0], sfeatures[:, 1], sfeatures[:, 2], c=labels.astype(np.float), edgecolor='k')
ax.set_title(r'$\xi=1$')
ax.set_xlabel(r'$W^{rad}$')
ax.set_ylabel(r'$W^{ang}_{\lambda=1}$')
ax.set_zlabel(r'$W^{ang}_{\lambda=-1}$')

print("Computing second range of features ...")
xi = 2
sfeatures = structure_features(strucs, eta, mu, xi, r_cutoff, lamb='both')
kmeans = KMeans(n_clusters=6)
kmeans.fit(sfeatures)
labels = kmeans.labels_

ax = fig.add_subplot(2, 3, 2, projection='3d')
ax.scatter(sfeatures[:, 0], sfeatures[:, 1], sfeatures[:, 2], c=labels.astype(np.float), edgecolor='k')
ax.set_title(r'$\xi=2$')
ax.set_xlabel(r'$W^{rad}$')
ax.set_ylabel(r'$W^{ang}_{\lambda=1}$')
ax.set_zlabel(r'$W^{ang}_{\lambda=-1}$')

print("Computing third range of features ...")
r_cutoff = 6
sfeatures = structure_features(strucs, eta, mu, xi, r_cutoff, lamb='both')
kmeans = KMeans(n_clusters=6)
kmeans.fit(sfeatures)
labels = kmeans.labels_

ax = fig.add_subplot(2, 3, 3, projection='3d')
ax.scatter(sfeatures[:, 0], sfeatures[:, 1], sfeatures[:, 2], c=labels.astype(np.float), edgecolor='k')
ax.set_title(r'$r_{cutoff}=6$')
ax.set_xlabel(r'$W^{rad}$')
ax.set_ylabel(r'$W^{ang}_{\lambda=1}$')
ax.set_zlabel(r'$W^{ang}_{\lambda=-1}$')

ax1 = fig.add_subplot(2, 3, 4)
ax2 = fig.add_subplot(2, 3, 5)
ax3 = fig.add_subplot(2, 3, 6)
for i in range(len(features1)):
    feature1 = features1[i, :]
    feature2 = features2[i, :]

    pos = positions[i, :2]
    n = numbers[i]

    con1 = ConnectionPatch(
        xyA=pos, coordsA=ax2.transData,
        xyB=feature1, coordsB=ax1.transData,
        arrowstyle="->", shrinkB=5, shrinkA=5
    )
    con2 = ConnectionPatch(
        xyA=feature2, coordsA=ax3.transData,
        xyB=pos, coordsB=ax2.transData,
        arrowstyle="<-", shrinkB=5, shrinkA=5
    )
    ax2.add_artist(con1)
    ax3.add_artist(con2)

    ax1.scatter(feature1[0], feature1[1], c='k', zorder=1)
    ax3.scatter(feature2[0], feature2[1], c='r', zorder=1)

    if n == 1:
        ax2.scatter(pos[0], pos[1], facecolors='none', edgecolors='k', s=100)
    elif n == 6:
        ax2.scatter(pos[0], pos[1], c='0.3', s=100)
    elif n == 8:
        ax2.scatter(pos[0], pos[1], c='r', s=100)
    else:
        raise ValueError(f"What? n={n}")
    ax2.axis('equal')
plt.show()
