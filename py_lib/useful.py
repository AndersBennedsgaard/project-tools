import numpy as np


def list_of_tuples(*args):
    assert all([isinstance(arg, list) for arg in args]), "arguments given must be lists"
    if len(args) == 2:
        return list(map(lambda x, y: (x, y), args[0], args[1]))
    if len(args) == 3:
        return list(map(lambda x, y, z: (x, y, z), args[0], args[1], args[2]))
    raise ValueError("Length of arguments not implemented")


def remove_diagonal(arr):
    """Removes the diagonal of a numpy array

    Args:
        arr (np.ndarray): square n*n array

    Returns:
        np.ndarray: rectangular n*(n-1) array
    """
    final_arr = arr[~np.eye(arr.shape[0], dtype=bool)].reshape(arr.shape[0], -1)
    return final_arr
