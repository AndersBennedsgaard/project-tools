import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def linear_probs(N):
    distr = [N - i for i in range(N)]
    return [x / sum(distr) for x in distr]


def squared_probs(N):
    distr = [(N - i)**2 for i in range(N)]
    return [x / sum(distr) for x in distr]


n_segments = 10

fig = plt.figure(figsize=(8, 8))

ax = plt.subplot(221)
colors = cm.rainbow(np.linspace(0, 1, n_segments))
for n in range(1, n_segments + 1):
    idx = n - 1
    x = [x + 1 for x in range(n)]
    y = linear_probs(n)
    ax.plot(x, y, c=colors[idx], label='n={}'.format(n))
    ax.scatter(x, y, c=[colors[idx]] * len(x), s=10)
ax.set_xlabel('Segment')
ax.set_ylim(0, 1 + 0.02)
ax.set_xlim(1 - 0.2, n_segments + 0.2)
ax.set_title(r'$(N - i)$')
ax.grid()
ax.set_ylabel('Probability')

ax = plt.subplot(222)
for n in range(1, n_segments + 1):
    idx = n - 1
    x = [x + 1 for x in range(n)]
    y = squared_probs(n)
    ax.plot(x, y, c=colors[idx], label='n={}'.format(n))
    ax.scatter(x, y, c=[colors[idx]] * len(x), s=10)
ax.set_xlabel('Segment')
ax.set_ylim(0, 1 + 0.02)
ax.set_yticklabels([])
ax.set_xlim(1 - 0.2, n_segments + 0.2)
ax.set_title(r'$(N - i)^2$')
ax.grid()
ax.legend()

x = np.linspace(0, 2, n_segments)

ax = plt.subplot(212)
for alpha in [0.5, 1, 2]:
    y = np.max(x) - alpha * x
    y[y < 0] = 0
    y /= np.sum(y)
    # distr = [max(x[0] - alpha * x, 0) for m in x]
    # distr = [x / sum(distr) for x in distr]

    plt.plot(x, y, label=fr'$alpha={{{alpha}}}$')
ax.legend()
ax.set_xlabel(r'$Q_{max}^1 - Q_{max}^i$')
ax.set_ylabel(r'$Q_{max}^1 - \alpha \left(Q_{max}^1 - Q_{max}^i\right)$')

plt.tight_layout()
plt.show()
