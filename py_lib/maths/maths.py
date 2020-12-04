import math
import numpy as np


def rotate_3d(array, yaw, pitch, roll):
    """Rotates 3D points around (0, 0, 0)

    Args:
        array ([type]): [description]
        yaw ([type]): [description]
        pitch ([type]): [description]
        roll ([type]): [description]

    Returns:
        [type]: [description]
    """
    yaw = yaw * np. pi / 180
    pitch = pitch * np. pi / 180
    roll = roll * np. pi / 180
    yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    rot_mat = yaw @ pitch @ roll
    return (rot_mat @ array.T).T


def ceil(var, d=0):
    return round(var + 0.5 * 10**(-d), d)


def cutoff_function(distances, r_cutoff):
    """Cutoff function for a numpy array

    0.5 + 0.5 cos(pi * r_ij / r_cutoff) for r_ij < r_cutoff
    0 otherwise

    Args:
        distances (np.ndarray): euclidian distances as either a 1- or 2-dimensional array
        r_cutoff (float): cutoff distance

    Raises:
        ValueError: If the number of dimensions of the input array is larger than 2

    Returns:
        np.ndarray: Calculated cutoff function
    """
    if distances.ndim == 1:
        if np.sqrt(np.sum(distances ** 2)) > r_cutoff:
            f_c = 0
        else:
            f_c = 0.5 + 0.5 * np.cos(math.pi * np.sqrt(np.sum(distances ** 2)) / r_cutoff)
    elif distances.ndim == 2:
        f_c = 0.5 + 0.5 * np.cos(math.pi * distances / r_cutoff)
        f_c[distances > r_cutoff] = 0
    else:
        raise ValueError("Distances-matrix too high-dimensional")
    return f_c
