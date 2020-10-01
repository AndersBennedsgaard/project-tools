import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def get_all_boundary_edges(bool_img):
    """
    Get a list of all edges (where the value changes from 'True' 
    to 'False') in the image.
    Return the list as indices of the image
    """
    assert len(bool_img.shape) == 2
    grid_size_x, grid_size_y = bool_img.shape

    ij_boundary = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == grid_size_y - 1 or not bool_img[i, j + 1]:
            ij_boundary.append(
                np.array([[i, j + 1], [i + 1, j + 1]])
                )
        # East
        if i == grid_size_x - 1 or not bool_img[i + 1, j]:
            ij_boundary.append(
                np.array([[i + 1, j], [i + 1, j + 1]])
                )
        # South
        if j == 0 or not bool_img[i, j - 1]:
            ij_boundary.append(
                np.array([[i, j], [i + 1, j]])
                )
        # West
        if i == 0 or not bool_img[i - 1, j]:
            ij_boundary.append(
                np.array([[i, j], [i, j + 1]])
                )

    if not ij_boundary:
        return np.zeros((0, 2, 2))
    else:
        return np.array(ij_boundary)


def close_loop_boundary_edges(xy_boundary, clean=True):
    """
    Connect all edges defined by 'xy_boundary' 
    to closed boundaries around a object.
    If not all edges are part of the surface of one object a list of closed 
boundaries is returned (one for every
    object).
    """

    boundary_loop_list = []
    while xy_boundary.size != 0:
        # Current loop
        xy_cl = [xy_boundary[0, 0], xy_boundary[0, 1]]  # Start with first edge
        xy_boundary = np.delete(xy_boundary, 0, axis=0)

        while xy_boundary.size != 0:
            # Get next boundary edge (edge with common node)
            ij = np.nonzero((xy_boundary == xy_cl[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                xy_cl.append(xy_cl[0])
                break

            xy_cl.append(xy_boundary[i, (j + 1) % 2, :])
            xy_boundary = np.delete(xy_boundary, i, axis=0)

        xy_cl = np.array(xy_cl)

        boundary_loop_list.append(xy_cl)

    return boundary_loop_list


def plot_world_outlines(bool_img, ax=None, **kwargs):
    """
    Plot the image by drawing the outlines of the areas where the values are True.
    """
    if ax is None:
        ax = plt.gca()

    ij_boundary = get_all_boundary_edges(bool_img=bool_img)
    xy_boundary = ij_boundary - 0.5
    xy_boundary = close_loop_boundary_edges(xy_boundary=xy_boundary)
    cl = LineCollection(xy_boundary, **kwargs)
    ax.add_collection(cl)
