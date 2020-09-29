import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS, LogNorm


def plot_segments(ax, minima):
    for x in minima:
        ax.plot([x / nA * maxA, x / nA * maxA], [-1, 1], ls='--', c='0.4')


def remove_ticks(ax):
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks_position('none')


center0 = 0
center1 = 2
center2 = 4
center3 = 5
center4 = 7
center5 = 8
center6 = 10.5

numQ = 3

width = 0.2
As = [-0.1, 0.2, 0.4]

maxA = 10
nA = 1000

lambdas = [0.5, 0.9]
n_probs = 5
a = np.linspace(0, maxA, nA)
Q = [np.zeros(a.shape) for i in range(numQ)]

for i, amplitude in enumerate(As):
    Q[i] -= 0.1 * np.exp(-(a - center0)**2 / 2 / width)
    Q[i] -= 0.2 * np.exp(-(a - center1)**2 / 2 / width) 
    Q[i] -= 0.3 * np.exp(-(a - center3)**2 / 2 / width / 6)
    Q[i] += amplitude * np.exp(-(a - center4)**2 / 2 / width)
    Q[i] -= 0.3 * np.exp(-(a - center5)**2 / 2 / width)
    Q[i] -= 0.3 * np.exp(-(a - center6)**2 / 2 / width / 6)

minima = [np.where(np.r_[True, Q[i][1:] <= Q[i][:-1]] & np.r_[Q[i][:-1] <= Q[i][1:], True]) for i in range(numQ)]
borders = []
for i in range(numQ):
    borders.append([0, maxA])
    for m in minima[i][0]:
        m = m / nA * maxA
        if m < center4:
            borders[i][0] = m
        else:
            borders[i][1] = m
            break

mask = []
for i in range(numQ):
    mask.append(np.logical_and(a > borders[i][0], a < borders[i][1]))

probs_seg = [[] for x in range(n_probs)]
probs_tot = [[] for x in range(n_probs)]
for j in range(len(Q)):
    q = Q[j][mask[j]]
    P = np.zeros(Q[j].shape)

    # e^(Q/T)
    p = P.copy() 
    T = 0.1
    p[mask[j]] += np.exp(q / T)
    p /= np.sum(p)
    probs_seg[0].append(p)

    p = P.copy()
    p += np.exp(Q[j] / T)
    p /= np.sum(p)
    probs_tot[0].append(p)

    # e^(Q/Qmax)
    p = P.copy()
    p[mask[j]] += np.exp(q / max(q))
    p /= np.sum(p)
    probs_seg[1].append(p)

    p = P.copy()
    p += np.exp(Q[j] / max(q))
    p /= np.sum(p)
    probs_tot[1].append(p)

    # e^(2 * (Q + 1)/(Qmax + 1) - 1)
    p = P.copy()
    p[mask[j]] += np.exp(2 * (q + 1) / (max(q) + 1) - 1)
    p /= np.sum(p)
    probs_seg[2].append(p)

    p = P.copy()
    p += np.exp(2 * (Q[j] + 1) / (max(q) + 1) - 1)
    p /= np.sum(p)
    probs_tot[2].append(p)

    # e^(2 * ((Q + 1)/(Qmax + 1))^kappa - 1)
    p = P.copy()
    kappa1 = 3
    p[mask[j]] += np.exp(2 * ((q + 1) / (max(q) + 1))**kappa1 - 1)
    p /= np.sum(p)
    probs_seg[3].append(p)

    p = P.copy()
    p += np.exp(2 * ((Q[j] + 1) / (max(q) + 1))**kappa1 - 1)
    p /= np.sum(p)
    probs_tot[3].append(p)

    # test distribution
    p = P.copy()
    kappa2 = 5
    p[mask[j]] += np.log(np.exp((2 * (q + 1) / (max(q) + 1))**kappa2 - 1))
    p /= np.sum(p)
    probs_seg[4].append(p)

    p = P.copy()
    p += np.log(np.exp(2 * ((Q[j] + 1) / (max(q) + 1))**kappa2 - 1))
    p /= np.sum(p)
    probs_tot[4].append(p)

maxP = [max([max(x) for x in probs_seg[i]]) for i in range(len(probs_seg))]

ylabels = [
        r'$exp\left(\dfrac{Q}{T}\right), T = $' + '{}'.format(T), 
        r'$exp\left(\dfrac{Q}{Q_{max}}\right)$', 
        r'$exp\left(2\dfrac{Q + 1}{Q_{max} + 1} - 1\right)$',
        r'$exp\left(2\left(\dfrac{{Q + 1}}{{Q_{{max}} + 1}}\right)^{{{}}} - 1\right)$'.format(kappa1),
        r'$ln\left(exp\left(\left(2\dfrac{{Q + 1}}{{Q_{{max}} + 1}} - 1\right)^{{{}}}\right)\right)$'.format(kappa2)  # change this after testing
        ]

fig, axes = plt.subplots(n_probs + 1, len(Q), figsize=(5 * len(Q), 3 * n_probs), sharex=True)
alpha1 = 0.6
c = CSS4_COLORS['royalblue']
alpha2 = 0.3
for i in range(len(Q)):
    ax = axes[0, i]
    amask1 = a < borders[i][0]
    amask2 = a > borders[i][1]

    ax.plot(a[amask1], Q[i][amask1], alpha=alpha1, c=c)
    ax.plot(a[amask2], Q[i][amask2], alpha=alpha1, c=c)
    ax.plot(a[mask[i]], Q[i][mask[i]], c=c, label=r"$Q_{max}$ = " + "{:.1f}".format(np.max(Q[i][mask[i]])))
    plot_segments(ax, minima[i])
    ax.fill_between(a[amask1], Q[i][amask1], -1, alpha=alpha2, color=c)
    ax.fill_between(a[amask2], Q[i][amask2], -1, alpha=alpha2, color=c)
    ax.fill_between(a[mask[i]], Q[i][mask[i]], -1, alpha=0.5, color=c)

    ax.set_xlim(min(a), max(a))
    ax.set_ylim(-1, 1)
    ax.legend(loc='upper left', fontsize=12)
    if i == 0:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
    else:
        remove_ticks(ax)

for i in range(len(probs_seg)):
    for j in range(len(Q)):
        ax = axes[i + 1, j]

        ax.plot(a, probs_seg[i][j], c=CSS4_COLORS['forestgreen'])
        ax.plot(a, probs_tot[i][j], c=CSS4_COLORS['deepskyblue'])
        plot_segments(ax, minima[j])

        ax.set_xlim(min(a), max(a))
        ax.set_ylim(0, round(maxP[i] + 0.005, 2))
        if j == 0:
            ax.set_ylabel(ylabels[i], rotation=60, fontsize=12, labelpad=20)
            ax.xaxis.set_ticklabels([])
            ax.xaxis.set_ticks_position('none')
        else:
            remove_ticks(ax)
fig.tight_layout()
# fig.savefig('Boltzmann_policy.png')

### 2nd plot ###
N = 100
M = 100

center = 0
qwidth = 0.3
a = np.linspace(-3, 3, M)
Q0 = np.linspace(-1, 1, M)

A = np.linspace(0., 2., 6)
Q = []
for aa in A:
    q = aa * np.exp(-(a - center)**2 / 2 / qwidth) - 1
    Q.append(q)

P = []
for q in Q:
    p = np.exp(q / max(q))
    P.append(p / np.sum(p))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes[0, 0]
for i, q in enumerate(Q):
    ax.plot(a, q)
ax.set_xlabel('Action space')
ax.set_ylabel('Q')

ax = axes[0, 1]
for i, p in enumerate(P):
    ax.plot(a, p, label="A = {:.2f}".format(A[i]))
ax.set_xlabel('Action space')
ax.set_ylabel('Probability')
ax.legend(loc='upper left')

ax = axes[1, 0]
for i, p in enumerate(P):
    dpda = np.diff(p) / np.diff(a)
    ax.plot(a[:-1], dpda)
ax.set_xlabel('Action space')
ax.set_ylabel('dp/da')

A = np.linspace(0, 2, N)
P = np.zeros((N, M))
for i in range(N):
    q = A[i] * np.exp(-(a - center)**2 / 2 / qwidth) - 1
    p = np.exp(q / Q0)
    p /= np.sum(p)
    P[i, :] += p.copy()

ax = axes[1, 1]
im = ax.imshow(P, extent=[min(Q0), max(Q0), min(A), max(A)], origin='lower', norm=LogNorm(vmin=0.00001, vmax=1))
fig.colorbar(im)
ax.set_ylabel(r'$A$')
ax.set_xlabel(r'$Q_0$')
ax.set_title(r'$exp(A * gauss(center) / Q_0)$')
fig.tight_layout()
# fig.savefig('Analysis.png')
plt.show()
