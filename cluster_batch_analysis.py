import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from ase.visualize.plot import plot_atoms
from lib.features import molecular_coulomb_feature
from lib.useful import remove_ticks
from lib.maths import validate_databases, get_structures, get_db_info
from asla.modules import Grid, db as DB


class Plot:
    """Plotting class meant to be combined with combinatorial analysis
    """

    def __init__(self, centroids, features, labels, cmap=None, lim_ratio=1, limits=None, print_ratio=False):
        assert features.shape[0] == len(labels), "Features and labels must be of same length"
        assert 0 <= lim_ratio <= 1, f"lim_ratio must be between 0 and 1, is {lim_ratio}"

        self._centroids = centroids
        self._features = features
        self.labels = labels
        self.cmap = cmap or plt.cm.get_cmap('Paired')

        self.centroids, self.features = self._to2d(self._centroids, self._features)
        self.print_ratio = print_ratio

        if limits:
            self.limits = limits
        else:
            x_feats = np.sort(self.features[:, 0])
            y_feats = np.sort(self.features[:, 1])
            lim_min, lim_max = int(len(x_feats) * (1 - lim_ratio)), int(len(x_feats) * lim_ratio)
            x_min, x_max = x_feats[lim_min], x_feats[lim_max]
            y_min, y_max = y_feats[lim_min], y_feats[lim_max]

            x_min = min(x_min, self.centroids[:, 0].min())
            x_max = max(x_max, self.centroids[:, 0].max())
            y_min = min(y_min, self.centroids[:, 1].min())
            y_max = max(y_max, self.centroids[:, 1].max())
            
            self.limits = (x_min, x_max, y_min, y_max)

    @staticmethod
    def _split(data, split_idx):
        features, centroids = np.split(data, [split_idx], axis=0)  # pylint: disable=unbalanced-tuple-unpacking
        return centroids, features

    @classmethod
    def _to2d(cls, centroids, features):
        data = np.append(features, centroids, axis=0)
        data = scale(data)  # when not normalizing in Analysis

        pca = PCA(2)
        data = pca.fit_transform(data)
        return cls._split(data, -len(centroids))

    def histogram(self, startstop=None, colors=None, ax=None):
        n_clusters = len(self.centroids)
        if not ax:
            fig = plt.figure()
            ax = fig.gca()
        if not colors:
            colors = [self.cmap(i / (len(self.centroids) - 1)) for i in range(len(self.centroids))]
        if startstop:
            if isinstance(startstop[0], int):
                labels = self.labels[startstop[0]:startstop[1]]
                counts = np.array([np.sum(labels == label) for label in range(n_clusters)])
            else:
                counts = []
                for idxs in startstop:
                    labels = self.labels[idxs[0]:idxs[1]]
                    counts.append([np.sum(labels == label) for label in range(n_clusters)])
                counts = np.sum(counts, axis=0)

        probs = counts / np.sum(counts)

        x = np.arange(n_clusters)
        width = 1
        
        recs = ax.bar(x, probs, width, color=colors, edgecolor='k', linewidth=1)

        autolabel(ax, recs, rotation=0, fontsize=14)
        ax.set_ylim(0.005, 1.25)
        remove_ticks(ax)

    def plot(self, startstop=None, ax=None, centroid_strucs=None, **kwargs):
        ylabel = kwargs.get('ylabel', '')
        xlabel = kwargs.get('xlabel', '')
        title = kwargs.get('title', '')
        marker = kwargs.get('marker', 'o')
        colors = kwargs.get('colors', None)
        
        x_min, x_max, y_min, y_max = self.limits
        if not colors:
            colors = [self.cmap(i / (len(self.centroids) - 1)) for i in range(len(self.centroids))]
        if not ax:
            fig = plt.figure()
            ax = fig.gca()
        if startstop:
            if isinstance(startstop[0], int):
                features = self.features[startstop[0]:startstop[1], :]
                labels = self.labels[startstop[0]:startstop[1]]
            else:
                # Recursion - if a list of indices are given, features in these indices are plotted together
                for idxs in startstop:
                    self.plot(idxs, ax=ax, centroid_strucs=centroid_strucs, **kwargs)
                return
        else:
            features = self.features.copy()
            labels = self.labels.copy()

        for label, _ in enumerate(self.centroids):
            label_features = features[labels == label, :]
            ax.scatter(label_features[:, 0], label_features[:, 1], marker=marker, color=colors[label], s=15, alpha=0.5, edgecolor=None)

        if centroid_strucs:
            self._plot_atoms(ax, centroid_strucs, self.limits)

        for label, centroid in enumerate(self.centroids):
            ax.scatter(
                centroid[0], centroid[1],
                marker='o', s=60, linewidths=2, edgecolor='k',
                color=colors[label], zorder=1
            )
        ox = np.sum(features[:, 0] < self.limits[0]) + np.sum(features[:, 0] > self.limits[1]) 
        oy = np.sum(features[:, 1] < self.limits[2]) + np.sum(features[:, 1] > self.limits[3])
        ratio_o = (ox + oy) / len(features[:, 0])

        ax.set_title(title, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.text(
                0.95, 0.95, rf'$p={{{(1 - ratio_o):.2f}}}$', ha='right', va='top', 
                transform=ax.transAxes, fontsize=14, bbox=dict(boxstyle='round', alpha=1, facecolor='w')
                )
        ax.set_xlim(self.limits[:2])
        ax.set_ylim(self.limits[2:])
        remove_ticks(ax)

    def _plot_atoms(self, ax, strucs, limits):
        x_min, x_max, y_min, y_max = limits
        ax_width, ax_height = (x_max - x_min) / 5, (y_max - y_min) / len(strucs) * 1.5
        ax_positions = get_ax_pos(self.centroids, len(strucs), x_min, x_max, y_min, y_max, ax_width, ax_height)

        for label, centroid in enumerate(self.centroids):
            struc = strucs[label]
            com = struc.get_center_of_mass()
            ax_pos = ax_positions[label]

            ax2 = add_inset(
                ax, ax_pos[0], ax_pos[1], ax_width, ax_height,
                box=True, limited=False, zorder=10, transform=ax.transData
            )
            plot_atoms(struc, ax2)
            ax2.set_xlim(com[0] - 5, com[0] + 3)
            ax2.set_ylim(com[1] - 5, com[1] + 3)
            ax.add_artist(
                mpatches.ConnectionPatch(
                    (0.5, 0.5), centroid, 'axes fraction', 'data',
                    axesA=ax2, axesB=ax, zorder=2, arrowstyle='->'
                )
            )


class Analysis:
    """Analysis class for combinatorial analysis
    """

    def __init__(self, directories, centroid_strucs, **kwargs):
        if isinstance(directories, str):
            self.directories = [directories]
        else:
            self.directories = directories
        self.centroid_strucs = centroid_strucs
        self.n_clusters = len(self.centroid_strucs)

        self._feature_kwargs = kwargs

        lst_of_args = ['energy', 'ordered', 'unique', 'limit', 'skip', 'direction']
        for arg in lst_of_args:
            assert arg in self._feature_kwargs, f"{arg} not in kwargs"

        self.feature_indices = None
        self.n_runs = kwargs.get('n_runs', np.inf)
        self.dbs = self.get_dbs()
        validate_databases([db for dir_dbs in self.dbs for db in dir_dbs])

    def run(self):
        t0 = time()
        features = self.get_features()
        centroids = self.get_centroids()
        if True:
            # Not normalizing features
            norm_centroids, norm_features = centroids.copy(), features.copy()
        else:
            norm_centroids, norm_features = self.normalize(centroids, features)

        if not True:
            # PCA on centroids, transform features relative to this
            pca = PCA(2)
            pca.fit(norm_centroids)
            norm_centroids = pca.transform(norm_centroids)
            norm_features = pca.transform(norm_features)

        t1 = time()
        labels = self.cluster(norm_centroids, norm_features)
        print("Time to collect features: ", t1 - t0)
        return norm_centroids, norm_features, labels

    def get_dbs(self):
        dbs = []
        for direc in self.directories:
            n = 0
            dir_dbs = []
            for file in os.listdir(direc):
                if '.db' not in file:
                    continue
                dir_dbs.append(DB(direc + '/' + file))
                n += 1
                if self.n_runs and n >= self.n_runs:
                    break
            dbs.append(dir_dbs)
        return dbs

    def get_features(self):
        """Calculates the features of the structures in all the databases

        Returns:
            np.ndarray -- The flattened numpy array of features of shape (n_dirs*n_runs*n_strucs, n_interactions)
        """
        # Calculation of features
        features = []
        dir_indices = []

        n_features = 0
        for dir_dbs in self.dbs:
            dir_features = []
            dir_indices.append([n_features])
            for db in dir_dbs:
                db_features = [molecular_coulomb_feature(struc) for struc in get_structures(db, **self._feature_kwargs)]
                dir_features.append(db_features)

                n_features += len(db_features)
                dir_indices[-1].append(n_features)
            features.append(dir_features)

        self.feature_indices = dir_indices
        # Flattened features - if n_strucs*n_runs = 1000, from [0:1000, :] is first dir, [1000:2000,:] the next etc.
        return np.vstack([np.vstack(dir_feats) for dir_feats in features])

    def get_centroids(self):
        _, grid_info = get_db_info(self.dbs[0][0])
        grid_scale, size, pseudo3D, anchor, _ = grid_info

        # Snapping to the grid
        centroid_strucs = [Grid(struc, size, grid_scale, anchor, [0] * 3, pseudo3D) for struc in self.centroid_strucs]
        centroids = np.array([molecular_coulomb_feature(struc) for struc in centroid_strucs])
        return centroids

    @classmethod
    def normalize(cls, centroids, features):
        dat = scale(np.append(features, centroids, axis=0))
        return cls._split(dat, -(len(centroids)))

    @staticmethod
    def cluster(centroids, features):
        dists = np.array(
            [np.sqrt(np.sum((features - centroid)**2, axis=1)) for centroid in centroids]
        )
        return np.argmin(dists, axis=0)

    @staticmethod
    def _split(data, split_idx):
        features, centroids = np.split(data, [split_idx], axis=0)  # pylint: disable=unbalanced-tuple-unpacking
        return centroids, features


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


def get_ax_pos(features, n_clusters, x_min, x_max, y_min, y_max, ax_width, ax_height):
    assert isinstance(features, np.ndarray), "Features must be a numpy array"
    assert features.ndim == 2, "Features are 2D"
    x_prio = np.argsort(np.argsort(features[:, 0]))
    x_mask = x_prio < n_clusters // 2

    ax_x = np.zeros(n_clusters)
    ax_x[x_mask] = x_min - ax_width * 1 / 4
    ax_x[~x_mask] = x_max - ax_width * 3 / 4

    y1 = features[x_mask, :]
    y1_prio = np.argsort(np.argsort(y1[:, 1])).astype(float)
    y2 = features[~x_mask, :]
    y2_prio = np.argsort(np.argsort(y2[:, 1])).astype(float)

    y1_prio[np.argmax(y1_prio)] *= 1.5
    y2_prio[np.argmax(y2_prio)] *= 1.5

    ax_y = np.zeros(n_clusters)
    ax_y[x_mask] = y1_prio * ax_height
    ax_y[~x_mask] = y2_prio * ax_height
    ax_y += y_min
    return np.vstack([ax_x, ax_y]).T

def get_4ax_pos(features, n_clusters, x_min, x_max, y_min, y_max, ax_width, ax_height):
    assert isinstance(features, np.ndarray), "Features must be a numpy array"
    assert features.ndim == 2, "Features are 2D"

    x_prio = np.argsort(np.argsort(features[:, 0]))
    x_mask = x_prio < n_clusters // 2
    
    ax_x = np.zeros(n_clusters)
    ax_x[x_mask] = x_min
    ax_x[~x_mask] = x_max - ax_width
    
    y_prio = np.argsort(np.argsort(features[:, 1]))
    y_mask = y_prio < n_clusters // 2

    ax_y = np.zeros(n_clusters)
    ax_y[y_mask] = y_min - ax_height / 4
    ax_y[~y_mask] = y_max - ax_height / 4
    return np.vstack([ax_x, ax_y]).T



def autolabel(ax, rects, **kwargs):
    """Attach a text label above each bar in *rects*, displaying its height."""
    fontsize = kwargs.get('fontsize', 15)
    rotation = kwargs.get('rotation', 0)
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f'{height*100:.0f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
            rotation=rotation, fontsize=fontsize
        )
def autolabel(ax, rects, **kwargs):
    """Attach a text label above each bar in *rects*, displaying its height."""
    fontsize = kwargs.get('fontsize', 15)
    rotation = kwargs.get('rotation', 0)
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f'{height*100:.0f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
            rotation=rotation, fontsize=fontsize
        )
