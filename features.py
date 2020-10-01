import numpy as np
from scipy.spatial import distance
from ase.atoms import Atoms
from .maths import list_of_tuples, remove_diagonal, cutoff_function


def _ACSF_radial(positions, eta, r_center, r_cutoff):
    distances = distance.cdist(positions, positions)
    distances = remove_diagonal(distances)
    f_c = cutoff_function(distances, r_cutoff)
    feature = np.sum(np.exp(-eta * ((distances - r_center) / r_cutoff) ** 2) * f_c, axis=1)
    return feature.tolist()


def _ACSF_angular(positions, xi, eta, r_cutoff, lamb=1):
    assert lamb in (-1, 1), "Lambda must be -1 or 1"
    angular = []
    for i, v_i in enumerate(positions):
        angular.append(0)
        for j, v_j in enumerate(positions):
            if j == i:
                continue
            for k, v_k in enumerate(positions):
                if k == i:
                    continue
                v_ij = v_i - v_j
                v_ik = v_i - v_k
                v_jk = v_j - v_k
                cos_to_angle = np.dot(v_ij / np.sqrt(np.sum(v_ij**2)), v_ik / np.sqrt(np.sum(v_ik**2)))
                angular[-1] += (1 + lamb * cos_to_angle)**xi * np.exp(
                    -eta * (np.sum(v_ij**2) + np.sum(v_ik**2) + np.sum(v_jk**2))
                ) * cutoff_function(v_ij, r_cutoff) * cutoff_function(v_ik, r_cutoff) * cutoff_function(v_jk, r_cutoff)
        angular[-1] *= 2**(1 - xi)
    return angular


def atomic_ACSF_features(positions, eta, r_center, xi, r_cutoff, lamb=1):
    radial = _ACSF_radial(positions, eta, r_center, r_cutoff)
    angular = _ACSF_angular(positions, xi, eta, r_cutoff, lamb)
    return np.array([radial, angular]).T


def molecular_ACSF_feature(structure, eta, r_center, xi, r_cutoff, lamb=1, separate=False):
    assert isinstance(structure, Atoms), "structure should be an ase.atom.Atoms object"
    if separate:
        feature = np.array([0] * 2)
        numbers = structure.numbers
        positions = structure.get_positions()
        for n in np.unique(numbers):
            poss = positions[numbers == n]
            feature = feature + np.sum(atomic_ACSF_features(poss, eta, r_center, xi, r_cutoff, lamb), axis=0)
    else:
        feature = np.sum(atomic_ACSF_features(structure.get_positions(), eta, r_center, xi, r_cutoff, lamb), axis=0)
    return tuple(feature)


def _wACSF_radial(structure, eta, mu, r_cutoff):
    numbers = structure.numbers
    g = np.array([numbers] * len(numbers))
    g = remove_diagonal(g)
    bonds = distance.cdist(structure.get_positions(), structure.get_positions())
    bonds = remove_diagonal(bonds)
    radial = np.sum(g * np.exp(-eta * (bonds - mu)**2) * cutoff_function(bonds, r_cutoff), axis=1)
    return radial.tolist()


def _wACSF_angular(structure, eta, xi, mu, r_cutoff, lamb):
    def vec_length(vector):
        return np.sqrt(np.sum(vector**2))
    assert lamb in (-1, 1), "Lambda must be -1 or 1"

    numbers = structure.numbers
    g = np.array([numbers] * len(numbers))
    g = remove_diagonal(g)
    positions = structure.get_positions()

    angular = []
    for i, v_i in enumerate(positions):
        angular.append(0)
        for j, v_j in enumerate(positions):
            if j == i:
                continue
            Z_j = numbers[j]
            for k, v_k in enumerate(positions):
                if k == i:
                    continue
                Z_k = numbers[k]
                v_ij = v_i - v_j
                v_ik = v_i - v_k
                v_jk = v_j - v_k
                cos_to_angle = np.dot(v_ij / np.sqrt(np.sum(v_ij**2)), v_ik / np.sqrt(np.sum(v_ik**2)))
                angular[-1] += Z_j * Z_k * (1 + lamb * cos_to_angle)**xi * np.exp(
                    -eta * ((vec_length(v_ij) - mu)**2 + (vec_length(v_ik) - mu)**2 + (vec_length(v_jk) - mu)**2)
                ) * cutoff_function(v_ij, r_cutoff) * cutoff_function(v_ik, r_cutoff) * cutoff_function(v_jk, r_cutoff)
        angular[-1] *= 2**(1 - xi)
    return angular


def atomic_wACSF_features(structure, eta, mu, xi, r_cutoff, lamb=1):
    """Computes the Weighted Atom-Centered Symmetry Function descriptors

    Arguments:
        structure {ase.atoms.Atoms} -- The structure that needs to have its feature calculated
        eta {float} -- Half of the inverse of the Guassian width
        mu {float} -- Center of Gaussian
        xi {float} -- Weight of linear molecules
        r_cutoff {float} -- Cutoff distance

    Keyword Arguments:
        lamb {int,str} -- Phase. Possibility using both phase -1 and 1, to make 3D features, with 'both' (default: {1})
    """
    assert lamb in (-1, 1, 'both'), "lambda must be -1, 1 or 'both'"
    radial = _wACSF_radial(structure, eta, mu, r_cutoff)
    if lamb == 'both':
        angular1 = _wACSF_angular(structure, eta, xi, mu, r_cutoff, lamb=-1)
        angular2 = _wACSF_angular(structure, eta, xi, mu, r_cutoff, lamb=1)
        features = list_of_tuples(radial, angular1, angular2)
    else:
        angular = _wACSF_angular(structure, eta, xi, mu, r_cutoff, lamb=lamb)
        features = list_of_tuples(radial, angular)
    return features


def molecular_wACSF_feature(structure, eta, mu, xi, r_cutoff, lamb=1, full=False):
    if full:
        features = atomic_wACSF_features(structure, eta, mu, xi, r_cutoff, lamb)
        idxs = np.argsort(structure.numbers)
        features = [features[i] for i in idxs]
        radial = list(map(lambda x: x[0], features))
        angular = list(map(lambda x: x[1], features))
        if lamb == 'both':
            angular2 = list(map(lambda x: x[2], features))
            return tuple(radial + angular + angular2)
            # rad1, rad2, ..., radN, ang1(lambda=1), ang2, ..., angN, ang1(lambda=-1), ang2, ..., angN
            # 3 * len(numbers) dimensional
        return tuple(radial + angular)
        # rad1, rad2, ..., radN, ang1, ang2, ..., angN
        # 2 * len(numbers) dimensional
        # return list(sum(features), ())
        # rad1, ang1, rad2, ang2, ..., radN, angN
        # 2 * len(numbers) dimensional
    return tuple(np.sum(atomic_wACSF_features(structure, eta, mu, xi, r_cutoff, lamb), axis=0))
    # sum(rad_i), sum(ang_i)
    # 2 dimensional


def molecular_coulomb_feature(structure):
    bonds = distance.cdist(structure.get_positions(), structure.get_positions())
    numbers = structure.numbers.astype(float)

    eye = np.eye(bonds.shape[0], dtype=bool)
    charges = np.outer(numbers, numbers)
    charges[eye] = 0.5 * numbers**2.4

    coulomb = np.ones_like(bonds)
    coulomb[~eye] = 1 / bonds[~eye]
    coulomb *= charges
    eigenvals = np.linalg.eigvals(coulomb)
    return np.sort(eigenvals)[::-1]
