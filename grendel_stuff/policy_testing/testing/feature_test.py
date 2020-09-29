import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from ase.io import read
from ase.visualize.plot import plot_atoms
from lib.features import molecular_ACSF_feature, molecular_wACSF_feature, molecular_coulomb_feature
from lib.useful import remove_ticks


def add_inset(axes, xanchor, yanchor, width=1 / 3, height=1 / 3, axis=False, box=False, limited=True, **kwargs):
    ax = axes.inset_axes((xanchor, yanchor, width, height), **kwargs)
    if limited:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    if not limited:
        ax.set_aspect('equal', 'box')
    if box:
        remove_ticks(ax)
        axis = True
    if not axis:
        ax.axis('off')
    return ax


def get_ax_pos(features):
    assert isinstance(features, np.ndarray), "Features must be a numpy array"
    assert features.ndim == 2, "Features are 2D"

    x_prio = np.argsort(np.argsort(features[:, 0]))
    x_mask = x_prio < n_clusters // 2

    ax_x = np.zeros(n_clusters)
    ax_x[x_mask] = x_min - ax_width / 2
    ax_x[~x_mask] = x_max - ax_width / 2

    y1 = features[x_mask, :]
    y1_prio = np.argsort(np.argsort(y1[:, 1]))
    y2 = features[~x_mask, :]
    y2_prio = np.argsort(np.argsort(y2[:, 1]))

    ax_y = np.zeros(n_clusters)
    ax_y[x_mask] = y1_prio * ax_height
    ax_y[~x_mask] = y2_prio * ax_height
    ax_y += y_min
    return np.vstack([ax_x, ax_y]).T


def features_strucs_closest_to_centroids(features, centroids):
    plot_strucs = []
    plot_features = []
    for label, centroid in enumerate(centroids):
        label_features = features[labels == label, :]
        label_strucs = [struc for i, struc in enumerate(strucs) if labels[i] == label]

        idx = np.argmin(np.sum((label_features - centroid)**2, axis=1))

        plot_strucs.append(label_strucs[idx])
        plot_features.append(label_features[idx])
    return np.array(plot_features), plot_strucs


# Features
r_cutoff = 6

# Radial
eta = 0.05
r_center = mu = 1.5

# Angular
xi = 2
lamb = 1

n_clusters = 10

cmap = plt.cm.get_cmap('Paired')

print("Reading structures ...")
strucs = read('files/testing_strucs.traj', index=':')

recalc = False
try:
    parameters = np.load('files/kmeans_parameters.npy').tolist()
    if [r_cutoff, eta, r_center, xi, lamb] != parameters:
        print("Parameters not equal - calculating new features")
        raise FileNotFoundError
except FileNotFoundError:
    recalc = True
    np.save('files/kmeans_parameters.npy', np.array([r_cutoff, eta, r_center, xi, lamb]))

########
# ACSF #
########

print("\nCalculating ACSF features ...")

try:
    if recalc:
        raise FileNotFoundError
    features = np.load('files/kmeans_ACSF_features.npy')
    print("\tLoaded ACSF features")
except FileNotFoundError:
    features = np.array(
        [molecular_ACSF_feature(struc, eta, r_center, xi, r_cutoff, lamb=1, separate=True) for struc in strucs]
    )
    np.save('files/kmeans_ACSF_features.npy', features)
    print("\tSaved ACSF features")
scaled_features = scale(features)
assert scaled_features.shape[0] == len(strucs), "ACSF features and strucs misaligned"

print("Clustering ...")
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(scaled_features)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

h = .2
x_min, x_max = scaled_features[:, 0].min() - 1, scaled_features[:, 0].max() + 1
y_min, y_max = scaled_features[:, 1].min() - 1, scaled_features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plot_features, plot_strucs = features_strucs_closest_to_centroids(scaled_features, centroids)

ax_width, ax_height = 1.75, (y_max - y_min) / n_clusters * 2
ax_positions = get_ax_pos(plot_features)

print("Predicting ...")
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print("Plotting ...")
fig = plt.figure()
ax = fig.gca()
for label, centroid in enumerate(centroids):
    ax_pos = ax_positions[label]
    ax2 = add_inset(
        ax, ax_pos[0], ax_pos[1], ax_width, ax_height, box=True, limited=False, zorder=10, transform=ax.transData
    )
    plot_atoms(plot_strucs[label], ax2)
    ax.add_artist(
        ConnectionPatch(
            (0.5, 0.5), plot_features[label], 'axes fraction', 'data', axesA=ax2, axesB=ax, zorder=2, arrowstyle='->'
        )
    )

ax.imshow(
    Z, interpolation='nearest',
    extent=(x_min, x_max, y_min, y_max),
    cmap=cmap,
    aspect='auto', origin='lower'
)

ax.plot(scaled_features[:, 0], scaled_features[:, 1], 'k.', markersize=2)
ax.scatter(
    centroids[:, 0], centroids[:, 1],
    marker='x', s=169, linewidths=3,
    color='w', zorder=1
)
ax.set_title("Clustering with ACSF features")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
remove_ticks(ax)
fig.savefig('images/kmeans_ACSF_result.png', dpi=150)

#########
# wACSF #
#########

print("\nCalculating wACSF features ...")

try:
    if recalc:
        raise FileNotFoundError
    features = np.load('files/kmeans_wACSF_features.npy')
    print("\tLoaded wACSF features")
except FileNotFoundError:
    features = np.array(
        [molecular_wACSF_feature(struc, eta, mu, xi, r_cutoff, lamb) for struc in strucs]
    )
    np.save('files/kmeans_wACSF_features.npy', features)
    print("\tSaved wACSF features")
scaled_features = scale(features)
assert scaled_features.shape[0] == len(strucs), "wACSF features and strucs misaligned"

print("Clustering ...")
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(scaled_features)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

h = .2
x_min, x_max = scaled_features[:, 0].min() - 1, scaled_features[:, 0].max() + 1
y_min, y_max = scaled_features[:, 1].min() - 1, scaled_features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plot_features, plot_strucs = features_strucs_closest_to_centroids(scaled_features, centroids)

ax_width, ax_height = 1.75, (y_max - y_min) / n_clusters * 2
ax_positions = get_ax_pos(plot_features)

print("Predicting ...")
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print("Plotting ...")
fig = plt.figure()
ax = fig.gca()
for label, centroid in enumerate(centroids):
    ax_pos = ax_positions[label]
    ax2 = add_inset(
        ax, ax_pos[0], ax_pos[1], ax_width, ax_height, box=True, limited=False, zorder=10, transform=ax.transData
    )
    plot_atoms(plot_strucs[label], ax2)
    ax.add_artist(
        ConnectionPatch(
            (0.5, 0.5), plot_features[label], 'axes fraction', 'data', axesA=ax2, axesB=ax, zorder=2, arrowstyle='->'
        )
    )

ax.imshow(
    Z, interpolation='nearest',
    extent=(x_min, x_max, y_min, y_max),
    cmap=cmap,
    aspect='auto', origin='lower'
)

ax.plot(scaled_features[:, 0], scaled_features[:, 1], 'k.', markersize=2)
ax.scatter(
    centroids[:, 0], centroids[:, 1],
    marker='x', s=169, linewidths=3,
    color='w', zorder=1
)
ax.set_title("Clustering with wACSF features")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
remove_ticks(ax)
fig.savefig('images/kmeans_wACSF_result.png', dpi=150)

###########
# Coulomb #
###########

print("\nCalculating Coulomb features ...")

try:
    if recalc:
        raise FileNotFoundError
    features = np.load('files/kmeans_coulomb_features.npy')
    print("\tLoaded Coulomb features")
except FileNotFoundError:
    features = np.array(
        [molecular_coulomb_feature(struc) for struc in strucs]
    )
    np.save('files/kmeans_coulomb_features.npy', features)
    print("\tSaved Coulomb features")

print("\nRunning PCA ...")
features = scale(features)
pca = PCA(n_components=2)
scaled_features = pca.fit_transform(features)
variance_ratio = pca.explained_variance_ratio_
print(f'Variance: {variance_ratio[0]:.2f}, {variance_ratio[1]:.2f} (sum: {np.sum(variance_ratio):.2f})')

assert scaled_features.shape[0] == len(strucs), "Coulomb features and strucs misaligned"

print("Clustering ...")
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(scaled_features)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

h = .2
x_min, x_max = scaled_features[:, 0].min() - 1, scaled_features[:, 0].max() + 1
y_min, y_max = scaled_features[:, 1].min() - 1, scaled_features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plot_features, plot_strucs = features_strucs_closest_to_centroids(scaled_features, centroids)

ax_width, ax_height = 1.75, (y_max - y_min) / n_clusters * 2
ax_positions = get_ax_pos(plot_features)

print("Predicting ...")
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print("Plotting ...")
fig = plt.figure()
ax = fig.gca()
for label, centroid in enumerate(centroids):
    ax_pos = ax_positions[label]
    ax2 = add_inset(
        ax, ax_pos[0], ax_pos[1], ax_width, ax_height, box=True, limited=False, zorder=10, transform=ax.transData
    )
    plot_atoms(plot_strucs[label], ax2)
    ax.add_artist(
        ConnectionPatch(
            (0.5, 0.5), plot_features[label], 'axes fraction', 'data', axesA=ax2, axesB=ax, zorder=2, arrowstyle='->'
        )
    )

ax.imshow(
    Z, interpolation='nearest',
    extent=(x_min, x_max, y_min, y_max),
    cmap=cmap,
    aspect='auto', origin='lower'
)

ax.plot(scaled_features[:, 0], scaled_features[:, 1], 'k.', markersize=2)
ax.scatter(
    centroids[:, 0], centroids[:, 1],
    marker='x', s=169, linewidths=3,
    color='w', zorder=1
)
ax.set_title("Clustering with Coulomb eigenvalues")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
remove_ticks(ax)
fig.savefig('images/kmeans_coulomb_result.png', dpi=150)

#############
# Histogram #
#############

ns = [np.sum(labels == label) for label in np.unique(labels)]
fig = plt.figure()
ax = fig.gca()
ax.bar(range(n_clusters), ns, color=[cmap(i / (n_clusters - 1)) for i in range(n_clusters)])
ax.set_xlabel("Clusters")
ax.set_title("Distribution of structures")
remove_ticks(ax)
fig.savefig("images/kmeans_coulomb_histogram.png", dpi=150)

print("Done!")
plt.show()
