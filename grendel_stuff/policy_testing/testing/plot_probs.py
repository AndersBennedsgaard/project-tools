import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-2, 2)
a = np.linspace(0, 1, 6)

fig, ax = plt.subplots()
ax2 = ax.twinx()
maxP = 0
for amp in a:
    y = amp * np.exp(-(x)**2 / 2 / 2)
    p = np.exp(2 * ((y + 1) / (max(y) + 1))**2 - 1)
    p /= np.sum(p)
    if max(p) > maxP:
        maxP = max(p)
    ax.plot(x, y, label=str(amp))
    ax2.plot(x, p, '--', label=str(amp))
ax.legend()
ax2.legend()
ax.set_ylim(-1, 1)
ax2.set_ylim(0, maxP)

plt.show()
