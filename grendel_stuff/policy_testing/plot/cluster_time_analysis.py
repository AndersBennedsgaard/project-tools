import os
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from cluster_batch_analysis import Analysis, Plot

centroid_files = [
    '/home/abb/templates/ethenol_CO.traj',
    '/home/abb/templates/ethenyl_ester.traj',
    '/home/abb/templates/ethylene_CO2.traj',
    '/home/abb/templates/propanoic_acid.traj',
]
centroid_strucs = [read(file) for file in centroid_files]

directories = [
    '/home/abb/projects/policy_testing/results/C3H4O2_EP_pena_0/',
    '/home/abb/projects/policy_testing/results/C3H4O2_greedy_0/',
    '/home/abb/projects/policy_testing/results/C3H4O2_random_0/',
    '/home/abb/projects/policy_testing/results/C3H4O2_BP_skip5_0/',
    '/home/abb/projects/policy_testing/results/C3H4O2_SBP_skip5_0/',
    '/home/abb/projects/policy_testing/results/C3H4O2_MMP_skip5_0/',
]
titles = [
    r"Mod. $\epsilon$ greedy",
    "Greedy",
    "Random",
    "Boltzmann policy",
    "Segmented Boltzmann policy",
    "History policy"
]
n_episodes = 5000
skip = 5
n_runs = 2
n_analysis = 200
n_plots = (n_episodes // skip + n_analysis - 1) // n_analysis

for direc, title in zip(directories, titles):
    A = Analysis(
        direc, centroid_strucs,
        energy=None, ordered=False, unique=False, limit=None, n_runs=n_runs, skip=skip, direction='beginning'
    )
    centroids, features, labels = A.run()
    feature_indices = A.feature_indices

    P = Plot(centroids, features, labels)
    labels = labels.reshape(n_runs, -1)

    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 3, 4))
    if axes.ndim == 1:
        axes = np.array([axes])
    for i, dir_idxs in enumerate(feature_indices):
        idxs = [list(range(dir_idxs[i], dir_idxs[i + 1], n_analysis)) + [dir_idxs[i + 1]]
                for i in range(len(dir_idxs) - 1)]
        idxs = [[sublist[i:i + 2] for sublist in idxs if len(sublist[i:i + 2]) == 2] for i in range(n_plots)]
        for j, idx in enumerate(idxs):
            if i == j == 0:
                plot_strucs = centroid_strucs
            else:
                plot_strucs = None

            P.histogram(startstop=idx, ax=axes[i, j], errorbar=True)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.01, 0.04, 0.99, 0.92])
    fig.savefig('images/timing_' + os.path.basename(direc.rstrip('/')) + '.png', dpi=150)
