import math
import numpy as np


def ceil(var, d=0):
    return round(var + 0.5 * 10**(-d), d)


def remove_diagonal(arr):
    """Removes the diagonal of a numpy array

    Args:
        arr (np.ndarray): square n*n array

    Returns:
        np.ndarray: rectangular n*(n-1) array
    """
    final_arr = arr[~np.eye(arr.shape[0], dtype=bool)].reshape(arr.shape[0], -1)
    return final_arr


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


def list_of_tuples(*args):
    assert all([isinstance(arg, list) for arg in args]), "arguments given must be lists"
    if len(args) == 2:
        return list(map(lambda x, y: (x, y), args[0], args[1]))
    if len(args) == 3:
        return list(map(lambda x, y, z: (x, y, z), args[0], args[1], args[2]))
    raise ValueError("Length of arguments not implemented")
