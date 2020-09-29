import matplotlib.pyplot as plt


Qdat = []
with open('Qvalues', 'r') as f:
    for row in f:
        Qdat.append([float(x) for x in row.split()])

Pdat = []
with open('Pvalues', 'r') as f:
    for row in f:
        Pdat.append([float(x) for x in row.split()])

QEPdat = []
with open('QEPvalues', 'r') as f:
    for row in f:
        QEPdat.append([float(x) for x in row.split()])

Qx = []
Qmean = []
Qstd = []
Qmin = []
Qmax = []
for i, d in enumerate(Qdat):
    Qx.append(i)
    Qmean.append(d[0])
    Qstd.append(d[1])
    Qmin.append(d[2])
    Qmax.append(d[3])

Px = []
Pmean = []
Pstd = []
Pmin = []
Pmax = []
for i, d in enumerate(Pdat):
    Px.append(i)
    Pmean.append(d[0])
    Pstd.append(d[1])
    Pmin.append(d[2])
    Pmax.append(d[3])

QEPx = []
QEPmean = []
QEPstd = []
QEPmin = []
QEPmax = []
for i, d in enumerate(QEPdat):
    QEPx.append(i)
    QEPmean.append(d[0])
    QEPstd.append(d[1])
    QEPmin.append(d[2])
    QEPmax.append(d[3])

# Plotting

fig, axes = plt.subplots(2, 1, figsize=(18, 10))

ax = axes[0]
ax.plot(Px, Pmean, 'k', lw=0.5, label='Pmean')
ax.plot(Px, Pmin, 'b', lw=0.5, label='Pmin/max')
ax.plot(Px, Pmax, 'b', lw=0.5)
ax.fill_between(
        Px, [Pmean[i] + Pstd[i] for i in range(len(Pmean))],
        [Pmean[i] - Pstd[i] for i in range(len(Pmean))], alpha=0.6, label='Pstd'
        )
ax.set_xlim(0, max(Px))
ax.set_ylim(0, max(Pmax))
ax.legend()

ax = axes[1]
ax.plot(Qx, Qmean, 'k', lw=0.5, label='Qmean')
ax.plot(Qx, Qmin, 'b', lw=0.5, label='Qmin/max')
ax.plot(Qx, Qmax, 'b', lw=0.5)
ax.fill_between(
        Qx, [Qmean[i] + Qstd[i] for i in range(len(Qmean))],
        [Qmean[i] - Qstd[i] for i in range(len(Qmean))], alpha=0.6, label='Qstd'
        )
ax.set_xlim(0, max(Qx))
ax.set_ylim(min(Qmin), max(Qmax))
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots(1, 1, figsize=(18, 5))
ax.plot(QEPx, QEPmean, 'k', lw=0.5, label='QEPmean')
ax.plot(QEPx, QEPmin, 'b', lw=0.5, label='QEPmin/max')
ax.plot(QEPx, QEPmax, 'b', lw=0.5)
ax.fill_between(
        QEPx, [QEPmean[i] + QEPstd[i] for i in range(len(QEPmean))],
        [QEPmean[i] - QEPstd[i] for i in range(len(QEPmean))], alpha=0.6, label='QEPstd'
        )
ax.set_xlim(0, max(QEPx))
ax.set_ylim(min(QEPmin), max(QEPmax))
ax.legend()
fig.tight_layout()

plt.show()

