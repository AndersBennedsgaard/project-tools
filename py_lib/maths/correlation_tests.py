# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
from scipy import stats
import numpy as np


def pearson(data1, data2, p=0.05):
    """Pearson correlation test.
    Tests whether two datasets have a linear relationship between them
    Assumes IID behavior for both datasets
    Assumes normal distributions in each sample, and same variance in both

    Args:
        data1 (list or np.ndarray): Array of sample data
        data2 (list or np.ndarray): Array of sample data
        p (float, optional): P-value. Defaults to 0.05.
    """
    stat, pval = stats.pearsonr(data1, data2)
    print(f'stat={stat:.3f}, p-value={pval:.3f}')
    if pval > p:
        print('Probably independent')
    else:
        print('Probably dependent')


def spearman(data1, data2, p=0.05):
    """Spearman's rank correlation test.
    Tests whether two datasets have a monotonic relationship betwen them
    Assumes IID behavior for both datasets
    Assumes observations in each sample can be ranked

    Args:
        data1 (list or np.ndarray): Array of sample data
        data2 (list or np.ndarray): Array of sample data
        p (float, optional): P-value. Defaults to 0.05.
    """
    stat, pval = stats.spearmanr(data1, data2)
    print(f'stat={stat:.3f}, p-value={pval:.3f}')
    if pval > p:
        print('Probably independent')
    else:
        print('Probably dependent')


def kendall(data1, data2, p=0.05):
    """Kendall's rank correlation test.
    Tests whether two datasets have a monotonic relationship betwen them
    Assumes IID behavior for both datasets
    Assumes observations in each sample can be ranked

    Args:
        data1 (list or np.ndarray): Array of sample data
        data2 (list or np.ndarray): Array of sample data
        p (float, optional): P-value. Defaults to 0.05.
    """
    stat, pval = stats.kendalltau(data1, data2)
    print(f'stat={stat:.3f}, p-value={pval:.3f}')
    if pval > p:
        print('Probably independent')
    else:
        print('Probably dependent')


def chi_squared(data1, data2, p=0.05):
    """Chi-squared correlation test.
    Tests whether two categorical variables are related using a frequency/contingency table
    Assumes obervations used in the calculation of the contingency table independent

    Args:
        data1 (list): Array of categorical sample data
        data2 (list): Array of categorical sample data
        p (float, optional): P-value. Defaults to 0.05.
    """
    table = np.array([data1, data2])
    stat, pval, dof, expected = stats.chi2_contingency(table)
    print(f'stat={stat:.3f}, p-value={pval:.3f}, DoF={dof}')
    if pval > p:
        print('Probably independent')
    else:
        print('Probably dependent')

    if not all([obs > 5 for obs in np.array(table).flatten()]):
        print("Warning: all observed frequencies not above 5")
    if not all([expect > 5 for expect in expected.flatten()]):
        print("Warning: all expected frequencies not above 5")


def correlation_test(data1, data2):
    """Tests whether two datasets are correlated.

    Args:
        data1 (list): Array sample data
        data2 (list): Array sample data
    """
    print_len = 60

    print("\n" + " Pearson correlation test ".center(print_len, '*'))
    pearson(data1, data2)

    print("\n" + " Spearman's rank correlation test ".center(print_len, '*'))
    spearman(data1, data2)

    print("\n" + " Kendall's rank correlation test ".center(print_len, '*'))
    kendall(data1, data2)

    print("\n" + " Chi-squared correlation test for categorical data ".center(print_len, '*'))
    if isinstance(data1[0], int) and isinstance(data2[0], int):
        chi_squared(data1, data2)
    else:
        print("Not categorical data. Skipping ...\n")
