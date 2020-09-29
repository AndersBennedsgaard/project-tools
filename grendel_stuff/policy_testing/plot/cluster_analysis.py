import os
import matplotlib.pyplot as plt
from ase.io import read
from cluster_batch_analysis import Plot, Analysis

directories = [
    '/home/abb/projects/policy_testing/results/C3H4O2_greedy_0/',
    '/home/abb/projects/policy_testing/results/C3H4O2_random_0/',
    '/home/abb/projects/policy_testing/results/C3H4O2_EP_pena_e0.1_0/',
    '/home/abb/projects/policy_testing/results/C3H4O2_BP_b20_0/',
    '/home/abb/projects/policy_testing/results/C3H4O2_SBP_e0.1_b20_0/',
    '/home/abb/projects/policy_testing/results/C3H4O2_MMP_g5_b20_0/',
]
print("Analysing directories:")
print("\t" + "\n\t".join([os.path.basename(direc.rstrip('/')) for direc in directories]))

ylabels = [
    "Greedy",
    "Random",
    r'Mod. $\epsilon$-greedy',
    "Boltzmann",
    "Partition",
    "Retrospective",
]

n_episodes = 5000

skip = 1
n_plots = 5
n_analysis = n_episodes // skip // n_plots

main_dir = '/home/abb/projects/policy_testing/plot/'

structure_search = {
    'energy': None,
    'ordered': False,
    'unique': False,
    'limit': None,
    'n_runs': 50,
    'skip': skip,
    'direction': 'beginning'
}

centroid_files = [
    '/home/abb/templates/ethenol_CO.traj',
    '/home/abb/templates/ethenyl_ester.traj',
    '/home/abb/templates/ethylene_CO2.traj',
    '/home/abb/templates/propanoic_acid.traj',
]
centroid_strucs = [read(file) for file in centroid_files]

print("\nParameters:")
print("\tNumber of episodes per run: \t", n_episodes)
print("\tNumber of plots: \t\t", n_plots)
print("\tNumbers of features per plot: \t", n_analysis * structure_search['n_runs'])
print("\tNumber of ASLA runs: \t\t", structure_search['n_runs'])
print(f"\tEvery {skip} structure used")

############
# Analysis #
############

A = Analysis(directories, centroid_strucs, **structure_search)
centroids, features, labels = A.run()
feature_indices = A.feature_indices

P = Plot(centroids, features, labels, lim_ratio=99.9 / 100, print_ratio=True)
for i, dir_idxs in enumerate(feature_indices):
    fig, axes = plt.subplots(
            2, n_plots, figsize=(20, 6), 
            gridspec_kw={'height_ratios': [1, 4], 'hspace': 0, 'wspace': 0.045, 'left': 0.025, 'right': 0.975, 'top': 0.975, 'bottom': 0.07}
            )

    idxs = [list(range(dir_idxs[i], dir_idxs[i + 1], n_analysis)) + [dir_idxs[i + 1]] for i in range(len(dir_idxs) - 1)]
    # list in a list, to use features from many runs
    didxs = [[sublist[i:i + 2] for sublist in idxs] for i in range(n_plots)]
    cidxs = [[[sublist[0], sublist[i + 1]] for sublist in idxs] for i in range(n_plots)]
    for j, (cidx, didx) in enumerate(zip(cidxs, didxs)):
        # All of the structures in all databases
        if j == 0:
            title = f"Episodes {(didx[0][0] * skip) % n_episodes} - {((didx[0][1] * skip) - 1) % n_episodes + 1}"
        else:
            title = f"{(didx[0][0] * skip) % n_episodes} - {((didx[0][1] * skip) - 1) % n_episodes + 1}"

        if i == j == 0:
            plot_strucs = centroid_strucs
        else:
            plot_strucs = None

        P.plot(startstop=didx, ax=axes[1, j], xlabel=title, centroid_strucs=plot_strucs)
        P.histogram(startstop=didx, ax=axes[0, j])
    fig.savefig(main_dir + 'images/clustering_' + os.path.basename(directories[i].rstrip('/')) + '.png', dpi=200)
plt.show()
