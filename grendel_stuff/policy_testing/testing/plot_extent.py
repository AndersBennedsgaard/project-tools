import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.gca()
ax.set_aspect('equal', 'box')

height = 0
xmax = ymax = 0
Qs = []
for i in range(3):
    x, y = np.random.randint(10, 15, 2)
    if x > xmax:
        xmax = x
    if y > ymax:
        ymax = y
    Q = np.random.rand(x, y) * 2 - 1
    Qs.append(Q)


def plot_sequence(sequence, ax, text=None):
    xmax = ymax = 0
    cdots = len(sequence)
    for i, entry in enumerate(sequence):
        if isinstance(entry, np.ndarray):
            y, x = entry.shape
            if x > xmax:
                xmax = x
            if y > ymax:
                ymax = y
        if entry == 'cdots':
            if cdots is not None:
                cdots = i
            else:
                raise ValueError('Only a single "cdots" are possible')

    ypos = 0
    for i, entry in enumerate(sequence[:cdots]):
        height, width = entry.shape
        ax.imshow(entry, extent=(0, width, ypos, ypos + height), vmin=-1, vmax=1, cmap='Blues')
        if text is not None:
            txt, text = text[0], text[1:]
            t = ax.text(xmax * 1.5, ypos + height / 2, txt, ha='left')
        ypos += height + 1

    if cdots < len(sequence):
        ypos += 5
        ax.scatter([xmax / 2] * 3, [ypos + i for i in range(-1, 2)], s=2, c='k')
        ypos += 5

    for i, entry in enumerate(sequence[cdots + 1:]):
        if cdots == len(sequence):
            break
        height, width = entry.shape
        ax.imshow(entry, extent=(0, width, ypos, ypos + height), vmin=-1, vmax=1, cmap='Blues')
        if text is not None:
            txt, text = text[0], text[1:]
            t = ax.text(xmax * 1.5, ypos + height / 2, txt, ha='left')
        ypos += height + 1
    inv = ax.transData.inverted()
    text_width, _ = inv.transform((t.clipbox.width, 0))

    ax.set_xlim(0, xmax * 1.5 + text_width)
    ax.set_ylim(ypos, 0)


Qs.insert(-1, 'cdots')
text = [rf'$m = {{{m}}}$' for m in range(1, len(Qs))]
text[-1] = r'$m = M_Z$'
plot_sequence(Qs, ax, text)

plt.show()
