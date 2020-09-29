import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from ase.visualize.plot import plot_atoms
from asla.modules import db
from lib.maths import get_db_info, validate_databases, get_dir_structures, get_wACSF_struc_feature
from lib.useful import remove_ticks
matplotlib.use('Agg')


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


directories = [
    '/home/abb/projects/policy_testing/results/2020_04_07/C5O2H6_EP_1/',
    '/home/abb/projects/policy_testing/results/2020_04_07/C5O2H6_MP_a0.9_g20_0/',
    '/home/abb/projects/policy_testing/results/2020_04_07/C5O2H6_MP_a1.0_0/',
]
common_prefix = os.path.commonprefix(directories)
legend_labels = [direc[:-3].lstrip(common_prefix) for direc in directories]

main_dir = '/home/abb/projects/policy_testing/plot/'

# Features
r_cutoff = 6

# Radial
eta = 0.05
r_center = mu = 1.5

# Angular
xi = 2
lamb = 1

n_clusters = 6

energy_limit = -450
unique = True
limit = None

print("\nInitializing ...")
databases = []
for directory in directories:
    for file in os.listdir(directory):
        if '.db' in file:
            databases.append(db(directory + '/' + file))
            break
validate_databases(databases)

atom_types, grid_info = get_db_info(databases[0])
# scale, size, pseudo3D, anchor, template_pos = grid_info

recalc = False
try:
    parameters = np.load(main_dir + 'files/kmeans_pca_parameters.npy', allow_pickle=True).tolist()
    if [r_cutoff, eta, r_center, xi, lamb, energy_limit, unique, limit] != parameters:
        print("Parameters not equal - calculating new features")
        raise FileNotFoundError
except FileNotFoundError:
    recalc = True
    np.save(main_dir + 'files/kmeans_pca_parameters.npy',
            np.array([r_cutoff, eta, r_center, xi, lamb, energy_limit, unique, limit]))

print("\nCalculating wACSF features ...")

try:
    if recalc:
        raise FileNotFoundError
    features = np.load(main_dir + 'files/kmeans_pca_features.npy', allow_pickle=True)
    print("\tLoaded features")
except FileNotFoundError:
    features = []
    strucs = []
    for direc in directories:
        structures = get_dir_structures(direc, energy=energy_limit, unique=unique, limit=limit)
        strucs.extend(structures)
        features.append(
            np.array(
                [
                    get_wACSF_struc_feature(
                        struc, eta, r_center, xi, r_cutoff, lamb=1, full=True
                    ) for struc in structures
                ]
            )
        )
        assert features[-1].shape[0] == len(structures), "wACSF features and strucs misaligned"
        print(f"\tNo. of features: {features[-1].shape[0]}")
    np.save(main_dir + 'files/kmeans_pca_features.npy', features, allow_pickle=True)
    print("\tSaved features")
n_dat = [0]
for feature in features:
    n_dat.append(feature.shape[0] + n_dat[-1])
features = np.vstack(features)

print('n_dat: ', n_dat)

print("\nRunning PCA ...")

features = scale(features)

pca = PCA(n_components=2)
results = pca.fit_transform(features)
np.save(main_dir + 'files/kmeans_pca_results.npy', results)
print("Done")

variance_ratio = pca.explained_variance_ratio_
print(f'Variance: {variance_ratio[0]:.2f}, {variance_ratio[1]:.2f} (sum: {np.sum(variance_ratio):.2f})')

print("\nClustering ...")
kmeans = KMeans(n_clusters=n_clusters)
scaled_features = scale(results)
kmeans.fit(scaled_features)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

h = .2
x_min, x_max = scaled_features[:, 0].min() - 1, scaled_features[:, 0].max() + 1
y_min, y_max = scaled_features[:, 1].min() - 1, scaled_features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plot_strucs = []
plot_features = []
for label, centroid in enumerate(centroids):
    label_features = scaled_features[labels == label, :]
    label_strucs = [struc for i, struc in enumerate(strucs) if labels[i] == label]

    idx = np.argmin(np.sum((label_features - centroid)**2, axis=1))

    plot_strucs.append(label_strucs[idx])
    plot_features.append(label_features[idx])

ax_width, ax_height = 1.5, (y_max - y_min) / n_clusters * 1.8
x1 = x_min - ax_width / 2
x2 = x_max - ax_width / 2

feature_priorities_y = np.argsort(np.array(plot_features)[:, 1])
feature_priorities_y = np.argsort(feature_priorities_y)

feature_priorities_x = np.argsort(np.array(plot_features)[:, 0])
feature_priorities_x = np.argsort(feature_priorities_x)
mask = feature_priorities_x < n_clusters // 2

feature_priorities_y1 = np.argsort(feature_priorities_y[mask])
feature_priorities_y2 = np.argsort(feature_priorities_y[~mask])

ax_positions = []
label_y1 = label_y2 = 0
for label, feature in enumerate(plot_features):
    if mask[label]:
        ax_positions.append((x1, feature_priorities_y1[label_y1] * ax_height + y_min))
        label_y1 += 1
    else:
        ax_positions.append((x2, feature_priorities_y2[label_y2] * ax_height + y_min))
        label_y2 += 1

print("Predicting ...")
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print("\nPlotting ...")
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
    cmap=plt.cm.Paired,  # pylint: disable=no-member
    aspect='auto', origin='lower'
)

markers = ['.', 'x', '^']
for i, label in enumerate(legend_labels):
    dat = scaled_features[n_dat[i]:n_dat[i + 1], :]
    ax.plot(dat[:, 0], dat[:, 1], 'k' + markers[i], markersize=2, label=label)
ax.scatter(
    centroids[:, 0], centroids[:, 1],
    marker='x', s=169, linewidths=3,
    color='w', zorder=1
)
ax.set_title("Clustering with PCA")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
remove_ticks(ax)
ax.set_xlabel('Principle component 1')
ax.set_ylabel('Principle component 2')
ax.set_title('Structures with energy < -450eV')
ax.legend()
fig.savefig(main_dir + "images/kmeans_pca.png", dpi=200)
