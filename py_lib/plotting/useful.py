def remove_ticks(ax):
    """Removes the ticks of both x- and y-axis

    Args:
        ax (matplotlib.axes.Axes): axes where ticks should be removed
    """
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])


def autolabel(ax, rects, **kwargs):
    """Attach a text label above each bar, displaying its height

    Args:
        ax (matplotlib.axes.Axes): axes that contains barplot
        rects (matplotlib.container.BarContainer): Container for the rectangles in a bar-plot: rects = ax.bar(...)
    """
    fontsize = kwargs.get('fontsize', 15)
    rotation = kwargs.get('rotation', 0)
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f'{height*100:.0f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
            rotation=rotation, fontsize=fontsize
        )
